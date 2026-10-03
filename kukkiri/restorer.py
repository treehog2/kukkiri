from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from kukkiri.basicvsrpp_net import BasicVSRPlusPlusNet
from kukkiri.trt_primary import STEP_BATCHES, TrtPrimary
from kukkiri.geometry import CLIP


def load_state(weights: Path) -> dict:
    if Path(weights).suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(weights))
    else:
        state = torch.load(str(weights), map_location="cpu")
        state = state.get("net", state.get("state_dict", state))
    return {
        k.removeprefix("generator."): v
        for k, v in state.items()
        if not k.startswith("data_preprocessor") and k != "step_counter"
    }


def infer_num_blocks(state: dict) -> int:
    idx = [
        int(k.split(".")[4])
        for k in state
        if k.startswith("backbone.backward_1.main.2.") and k.endswith("conv1.weight")
    ]
    return max(idx) + 1 if idx else 7


def _load_generator(weights: Path, device: str):
    state = load_state(weights)
    net = BasicVSRPlusPlusNet(mid_channels=64, num_blocks=infer_num_blocks(state), is_low_res_input=False)
    net.load_state_dict(state, strict=True)
    return net.eval().to(device)


class VideoRestorer:
    def __init__(
        self,
        weights: str | Path,
        device: str = "cuda",
        fp16: bool = True,
        compile_clip: int = 0,
        cuda_graph: bool = True,
        trt_steps: bool = False,
        engine_dir: str | Path | None = None,
    ):
        self.device = device
        self.fp16 = fp16 and device.startswith("cuda")
        self.net = _load_generator(Path(weights), device)
        if self.fp16:
            self.net = self.net.half()
        self.compile_clip = compile_clip
        if compile_clip:
            self.net = torch.compile(self.net, mode="reduce-overhead")
        self.cuda_graph = cuda_graph and device.startswith("cuda")
        self.graph_clip = CLIP
        self.graph_batch = 1
        self._graphs: dict = {}
        self.trt_steps = trt_steps and device.startswith("cuda")
        self.engine_dir = Path(engine_dir) / "primary_steps" if engine_dir is not None else None
        self._primaries: dict = {}

    def warm(self, S: int, batches=STEP_BATCHES) -> None:
        if self.trt_steps:
            for B in batches:
                self._primary(B, S)

    def _primary(self, B: int, S: int):
        p = self._primaries.get((B, S))
        if p is None:
            p = self._primaries[(B, S)] = TrtPrimary(
                self.net, B, S // 4, S // 4, mode="torch_tensorrt", cache_dir=self.engine_dir
            )
        return p

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.cuda_graph:
            return self.net(x)
        key = tuple(x.shape)
        g = self._graphs.get(key)
        if g is None:
            static_in = x.clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self.net(static_in)
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                static_out = self.net(static_in)
            graph.replay()
            ref = self.net(static_in)
            err = (static_out.float() - ref.float()).abs().max().item()
            if err > 1e-2:
                self._graphs[key] = None
                return ref
            g = self._graphs[key] = (graph, static_in, static_out)
        if g is None:
            return self.net(x)
        graph, static_in, static_out = g
        static_in.copy_(x)
        graph.replay()
        return static_out

    @torch.no_grad()
    def restore_canvases(self, clips: list[np.ndarray]) -> list[np.ndarray]:
        out: list = [None] * len(clips)
        by_len: dict = {}
        for i, c in enumerate(clips):
            by_len.setdefault(c.shape[0], []).append(i)
        for T, idx in by_len.items():
            if self.trt_steps:
                group = [clips[i] for i in idx]
                B = next(b for b in STEP_BATCHES if b >= len(group))
                group = group + [group[0]] * (B - len(group))
                x = self._prep(group)
                y = self._primary(len(group), x.shape[-1]).forward(x)
                y = (
                    (y.float().clamp(0, 1) * 255.0)
                    .round()
                    .clamp(0, 255)
                    .byte()
                    .flip(2)
                    .permute(0, 1, 3, 4, 2)
                    .contiguous()
                )
                y = y.cpu().numpy()
                for b, i in enumerate(idx):
                    out[i] = y[b]
                continue
            if len(idx) == 1 and not (self.cuda_graph and T == self.graph_clip and self.graph_batch > 1):
                out[idx[0]] = self.restore_canvas(clips[idx[0]])
                continue
            group = [clips[i] for i in idx]
            if self.cuda_graph and T == self.graph_clip and len(group) < self.graph_batch:
                group = group + [group[0]] * (self.graph_batch - len(group))
            x = self._prep(group)
            y = self._forward(x) if T == self.graph_clip else self.net(x)
            y = (y.float().clamp(0, 1) * 255.0).round().clamp(0, 255).byte().flip(2).permute(0, 1, 3, 4, 2).contiguous()
            y = y.cpu().numpy()
            for b, i in enumerate(idx):
                out[i] = y[b]
        return out

    def _prep(self, group: list) -> torch.Tensor:
        x = (
            (torch.stack(group) if isinstance(group[0], torch.Tensor) else torch.from_numpy(np.stack(group)))
            .to(self.device)
            .flip(-1)
            .permute(0, 1, 4, 2, 3)
        )
        return (x.half() if self.fp16 else x.float()).div_(255.0).contiguous()

    @torch.no_grad()
    def restore_canvas(self, crops: np.ndarray) -> np.ndarray:
        n = crops.shape[0]
        if self.compile_clip and n < self.compile_clip:
            crops = np.concatenate([crops, np.repeat(crops[-1:], self.compile_clip - n, 0)])
        x = (
            (crops if isinstance(crops, torch.Tensor) else torch.from_numpy(crops))
            .to(self.device)
            .flip(-1)
            .permute(0, 3, 1, 2)[None]
        )
        x = (x.half() if self.fp16 else x.float()).div_(255.0).contiguous()
        if self.trt_steps:
            y = self._primary(1, x.shape[-1]).forward(x)
        else:
            y = self._forward(x) if x.shape[1] == self.graph_clip else self.net(x)
        y = (
            (y[0, :n].float().clamp(0, 1) * 255.0).round().clamp(0, 255).byte().flip(1).permute(0, 2, 3, 1).contiguous()
        )
        return y.cpu().numpy()

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

PREFIX = "model.diffusion_model."
SIGMAS = (1.0, 0.725, 0.0)
FPS = 30.0
SEED = 42


class Int8Linear(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool) -> None:
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16), False) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.reshape(-1, x.shape[-1])
        if rows.shape[0] <= 16:
            y = rows.float() @ (self.weight.float() * self.weight_scale[:, None]).t()
        else:
            s = rows.abs().amax(-1, keepdim=True).float().clamp_min(1e-8) / 127.0
            q = (rows.float() / s).round().clamp(-127, 127).to(torch.int8)
            y = torch._int_mm(q, self.weight.t()).float() * s * self.weight_scale
        y = y.to(x.dtype)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*x.shape[:-1], self.out_features)


_DTYPES = {"I8": torch.int8, "U8": torch.uint8, "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
           "I32": torch.int32, "I64": torch.int64, "BOOL": torch.bool}


def _header(f) -> tuple[dict, dict, int]:
    n = int.from_bytes(f.read(8), "little")
    header = json.loads(f.read(n))
    return header, header.pop("__metadata__", None) or {}, 8 + n


def _read_to_device(path: Path, device: torch.device, on_host=lambda name: False) -> dict[str, torch.Tensor]:
    out = {}
    with open(path, "rb") as f:
        header, _, base = _header(f)
        for name, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            a, b = info["data_offsets"]
            f.seek(base + a)
            buf = bytearray(f.read(b - a))
            t = torch.frombuffer(buf, dtype=_DTYPES[info["dtype"]]) if b > a else torch.empty(0, dtype=_DTYPES[info["dtype"]])
            t = t.reshape(info["shape"])
            out[name] = t.pin_memory() if on_host(name) else t.to(device)
            del buf, t
    return out


def _stream_blocks(blocks: torch.nn.ModuleList, keep: int, device: torch.device) -> None:
    for block in list(blocks)[keep:]:
        tensors = list(block.parameters()) + list(block.buffers())
        parked = [(t, t.data if t.device.type == "cpu" else t.data.to("cpu").pin_memory()) for t in tensors]
        for t, cpu in parked:
            t.data = cpu

        def load(_mod, _args, parked=parked):
            for t, cpu in parked:
                t.data = cpu.to(device, non_blocking=True)

        def unload(_mod, _args, _out, parked=parked):
            for t, cpu in parked:
                t.data = cpu

        block.register_forward_pre_hook(load)
        block.register_forward_hook(unload)
    torch.cuda.empty_cache()


def _load_transformer(path: Path, device: torch.device, keep: int | None = None) -> torch.nn.Module:
    from ltx_core.model.transformer import LTXVideoOnlyModelConfigurator
    from ltx_core.model.transformer.model import X0Model

    with open(path, "rb") as f:
        meta = _header(f)[1]
    with torch.device("meta"):
        model = LTXVideoOnlyModelConfigurator.from_metadata({"config": json.loads(meta["config"])})
    def on_host(name: str) -> bool:
        parts = name.removeprefix(PREFIX).split(".")
        return keep is not None and parts[0] == "transformer_blocks" and int(parts[1]) >= keep

    state = {k.removeprefix(PREFIX): v for k, v in _read_to_device(path, device, on_host).items()}
    for name, mod in list(model.named_modules()):
        if isinstance(mod, torch.nn.Linear) and f"{name}.weight_scale" in state:
            parent, _, child = name.rpartition(".")
            with torch.device("meta"):
                q = Int8Linear(mod.in_features, mod.out_features, mod.bias is not None)
            setattr(model.get_submodule(parent), child, q)
    model.load_state_dict(state, strict=True, assign=True)
    for t in list(model.parameters()) + list(model.buffers()):
        if t.is_meta:
            raise RuntimeError("transformer tensor missing from the weights file")
    return X0Model(model).eval()


def _fp32_conv3d(module: torch.nn.Module, frames: int = 8) -> None:
    import torch.nn.functional as F

    for m in module.modules():
        if not isinstance(m, torch.nn.Conv3d):
            continue

        def forward(x, m=m):
            w, b = m.weight.float(), None if m.bias is None else m.bias.float()

            def conv(v):
                return F.conv3d(v.float(), w, b, m.stride, m.padding, m.dilation, m.groups).to(x.dtype)

            kt, st = m.kernel_size[0], m.stride[0]
            if m.padding[0] or m.dilation[0] != 1:
                return conv(x)
            n = (x.shape[2] - kt) // st + 1
            return torch.cat([conv(x[:, :, a * st:(min(a + frames, n) - 1) * st + kt]) for a in range(0, n, frames)], 2)

        m.forward = forward


def _chunk_attention(module: torch.nn.Module, rows: int = 1024) -> None:

    def chunked(fn):
        def call(q, k, v, heads, *mask):
            if q.shape[1] <= rows:
                return fn(q, k, v, heads, *mask)
            parts = []
            for a in range(0, q.shape[1], rows):
                m = mask
                if mask and mask[0] is not None and mask[0].shape[-2] == q.shape[1]:
                    m = (mask[0][..., a:a + rows, :],)
                parts.append(fn(q[:, a:a + rows], k, v, heads, *m))
            return torch.cat(parts, 1)

        return call

    for m in module.modules():
        for name in ("attention_function", "masked_attention_function"):
            if callable(getattr(m, name, None)):
                setattr(m, name, chunked(getattr(m, name)))


def pad_len(n: int) -> int:
    return 8 * -(-(n - 1) // 8) + 1


class LtxRestorer:
    canvas = 512

    def __init__(self, weights_dir: str | Path, device: str = "cuda", compile: bool = True,
                 resident_blocks: int | None = None) -> None:
        from ltx_core.model.video_vae.memory_efficient_decode import disable_memory_efficient_decode
        from ltx_pipelines.utils.blocks import ImageConditioner, VideoDecoder

        from kukkiri.weights import ltx_paths

        tr, vae, prompt = ltx_paths(Path(weights_dir))
        self.weights_dir, self.compiled, self.resident_blocks = weights_dir, compile, resident_blocks
        self.device, self.dtype = torch.device(device), torch.bfloat16
        self.context = load_file(str(prompt))["video_encoding"].to(self.device, self.dtype)
        self.model = _load_transformer(tr, self.device, resident_blocks)
        self.encoder = ImageConditioner(str(vae), self.dtype, self.device)._encoder_builder.build(
            device=self.device, dtype=self.dtype
        ).eval()
        dec = VideoDecoder(str(vae), self.dtype, self.device)._prepared_builder().build(device=self.device, dtype=self.dtype)
        self.decoder = disable_memory_efficient_decode(dec.eval())
        if torch.cuda.get_device_capability(self.device)[0] < 8:
            _chunk_attention(self.model)
            _fp32_conv3d(self.encoder)
            _fp32_conv3d(self.decoder)
        if compile:
            from ltx_core.model.transformer.compiling import CompilationConfig, compile_transformer

            compile_transformer(self.model.velocity_model, CompilationConfig())
            self.encoder = torch.compile(self.encoder)
            self.decoder.forward = torch.compile(self.decoder.forward)
        if resident_blocks is not None and resident_blocks < len(self.model.velocity_model.transformer_blocks):
            _stream_blocks(self.model.velocity_model.transformer_blocks, resident_blocks, self.device)

    def warm(self, n_frames: int) -> None:
        try:
            self.restore_canvas(np.zeros((n_frames, self.canvas, self.canvas, 3), np.uint8))
            failed = None
        except torch.OutOfMemoryError:
            raise
        except Exception as e:
            if not self.compiled:
                raise
            failed = str(e).strip().splitlines()[-1][:300]
        if failed is not None:
            import logging

            logging.getLogger(__name__).warning(f"kiyo-ltx: compiling failed, continuing uncompiled (slower): {failed}")
            torch._dynamo.reset()
            for k in ("model", "encoder", "decoder", "context"):
                setattr(self, k, None)
            gc.collect()
            torch.cuda.empty_cache()
            self.__init__(self.weights_dir, str(self.device), compile=False, resident_blocks=self.resident_blocks)
            self.restore_canvas(np.zeros((n_frames, self.canvas, self.canvas, 3), np.uint8))
        torch.cuda.synchronize()

    def restore_clip(self, tr, sp, cc) -> np.ndarray:
        return self.restore_canvas(cc.crops)

    @torch.inference_mode()
    def restore_canvas(self, crops) -> np.ndarray:
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.components.patchifiers import VideoLatentPatchifier
        from ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
        from ltx_core.tools import VideoLatentTools
        from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape, VideoPixelShape
        from ltx_pipelines.utils.blocks.diffusion_stage import _build_state
        from ltx_pipelines.utils.denoisers import SimpleDenoiser
        from ltx_pipelines.utils.helpers import create_initial_video_latent
        from ltx_pipelines.utils.samplers import _step_state
        from ltx_pipelines.utils.types import ModalitySpec

        x = crops if isinstance(crops, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(crops))
        x = x.to(self.device).flip(-1)
        n, h, w = x.shape[:3]
        m = pad_len(n)
        if m > n:
            x = torch.cat([x, x[-1:].expand(m - n, -1, -1, -1)])
        video = (x.permute(3, 0, 1, 2)[None].float() / 127.5 - 1.0).to(self.dtype)
        ref = self.encoder(video)
        if self.resident_blocks is not None:
            del video, x
            torch.cuda.empty_cache()
        sf = SpatioTemporalScaleFactors.default()
        latent = create_initial_video_latent(
            pixel_shape=VideoPixelShape(batch=1, frames=m, height=h, width=w, fps=FPS),
            device=self.device,
            dtype=self.dtype,
            scale_factors=sf,
        )
        spec = ModalitySpec(
            latent=latent, conditioning_fps=FPS, context=self.context, conditionings=[VideoConditionByReferenceLatent(ref)]
        )
        tools = VideoLatentTools(
            VideoLatentPatchifier(patch_size=1), VideoLatentShape.from_torch_shape(latent.shape), FPS, scale_factors=sf
        )
        g = torch.Generator(device=self.device).manual_seed(SEED)
        state = _build_state(spec, tools, GaussianNoiser(generator=g), self.dtype, self.device)
        sigmas = torch.tensor(SIGMAS)
        for i in range(len(SIGMAS) - 1):
            res, _ = SimpleDenoiser()(self.model, state, None, sigmas.to(self.device), i, self.context, None)
            state = _step_state(state, res.denoised, EulerDiffusionStep(), sigmas, i)
        out = tools.unpatchify(tools.clear_conditioning(state)).latent
        if self.resident_blocks is not None:
            del state, res, spec
            torch.cuda.empty_cache()
        frames = torch.cat(list(self.decoder.decode_video(out, None, g)))
        out = (frames[:n].float().clamp(0, 1) * 255 + 0.5).to(torch.uint8).flip(-1).cpu().numpy()
        if self.resident_blocks is not None:
            del frames
            torch.cuda.empty_cache()
        return out

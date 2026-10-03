
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from kukkiri.basicvsrpp_net import BasicVSRPlusPlusNet, flow_warp

BRANCHES = ["backward_1", "forward_1", "backward_2", "forward_2"]
STEP_BATCHES = (1, 2, 4, 6, 8)


class PropStep(nn.Module):

    def __init__(self, net: BasicVSRPlusPlusNet, branch: str):
        super().__init__()
        self.align = net.deform_align[branch]
        self.backbone = net.backbone[branch]
        self.has_prefix = BRANCHES.index(branch) > 0

    def forward(self, feat_prop, feat_n2, feat_current, flow_n1, flow_n2, prefix=None):
        cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
        cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
        cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
        aligned = self.align(torch.cat([feat_prop, feat_n2], dim=1), cond, flow_n1, flow_n2)
        feat = [feat_current] + ([prefix] if prefix is not None else []) + [aligned]
        return aligned + self.backbone(torch.cat(feat, dim=1))


class _GraphedStep:

    def __init__(self, fn, example: list[torch.Tensor]):
        self.fn = fn
        self.inp = [e.clone() for e in example]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fn(*self.inp)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
            self.out = fn(*self.inp)

    def __call__(self, *args):
        for b, a in zip(self.inp, args):
            b.copy_(a)
        self.graph.replay()
        return self.out


def _step_inputs(B: int, P: int, h: int, w: int, device, dtype) -> list[torch.Tensor]:
    def z(c):
        return torch.zeros(B, c, h, w, device=device, dtype=dtype)

    ex = [z(64), z(64), z(64), z(2), z(2)]
    if P:
        ex.append(z(P))
    return ex


class TrtPrimary:

    def __init__(
        self, net: BasicVSRPlusPlusNet, B: int, h: int, w: int, mode: str = "graph", cache_dir: str | Path | None = None
    ):
        self.net = net
        self.B, self.h, self.w = B, h, w
        self.mode = mode
        self.device = next(net.parameters()).device
        self.dtype = next(net.parameters()).dtype
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.steps = {}
        for i, br in enumerate(BRANCHES):
            self.steps[br] = self._build(br, P=64 * i)

    def _build(self, br: str, P: int):
        step = PropStep(self.net, br).eval()
        ex = _step_inputs(self.B, P, self.h, self.w, self.device, self.dtype)
        if self.mode == "eager":
            return step
        if self.mode == "graph":
            return _GraphedStep(step, ex)
        if self.mode == "torch_tensorrt":
            import tensorrt
            import torch_tensorrt

            path = None
            if self.cache_dir is not None:
                name = engine_names(tensorrt.__version__.split(".")[0], (self.B,))[BRANCHES.index(br)]
                path = self.cache_dir / Path(name).name
            if path is not None and path.exists():
                return _GraphedStep(torch_tensorrt.load(str(path)).module(), ex)
            with torch.no_grad():
                trt_mod = torch_tensorrt.compile(
                    step,
                    ir="dynamo",
                    inputs=ex,
                    enabled_precisions={self.dtype},
                    truncate_double=True,
                    min_block_size=1,
                    optimization_level=5,
                )
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                torch_tensorrt.save(trt_mod, str(path), inputs=ex)
            return _GraphedStep(trt_mod, ex)
        raise ValueError(self.mode)

    @torch.no_grad()
    def forward(self, lqs: torch.Tensor) -> torch.Tensor:
        net = self.net
        n, t, c, h, w = lqs.size()
        lqs_down = F.interpolate(lqs.view(-1, c, h, w), scale_factor=0.25, mode="bicubic").view(n, t, c, h // 4, w // 4)
        feats_ = net.feat_extract(lqs.view(-1, c, h, w))
        fh, fw = feats_.shape[2:]
        feats_ = feats_.view(n, t, -1, fh, fw)
        feats = {"spatial": [feats_[:, i] for i in range(t)]}
        flows_forward, flows_backward = net.compute_flow(lqs_down)
        for br in BRANCHES:
            feats[br] = []
            feats = self._propagate(feats, flows_backward if "backward" in br else flows_forward, br)
        return net.upsample(lqs, feats)

    def _propagate(self, feats, flows, br):
        net = self.net
        n, t, _, h, w = flows.size()
        frame_idx = list(range(0, t + 1))
        flow_idx = list(range(-1, t))
        mapping_idx = list(range(0, len(feats["spatial"])))
        mapping_idx += mapping_idx[::-1]
        if "backward" in br:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx
        earlier = [k for k in feats if k not in ("spatial", br)]
        step = self.steps[br]
        feat_prop = flows.new_zeros(n, net.mid_channels, h, w)
        zeros = feat_prop
        zflow = flows.new_zeros(n, 2, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats["spatial"][mapping_idx[idx]]
            prefix = torch.cat([feats[k][idx] for k in earlier], dim=1) if earlier else None
            if i == 0:
                feat = [feat_current] + ([prefix] if prefix is not None else []) + [feat_prop]
                feat_prop = feat_prop + net.backbone[br](torch.cat(feat, dim=1))
            else:
                flow_n1 = flows[:, flow_idx[i]]
                if i > 1:
                    feat_n2 = feats[br][-2]
                    flow_n2 = flows[:, flow_idx[i - 1]]
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                else:
                    feat_n2, flow_n2 = zeros, zflow
                args = [feat_prop, feat_n2, feat_current, flow_n1, flow_n2] + ([prefix] if prefix is not None else [])
                feat_prop = step(*args).clone()
            feats[br].append(feat_prop)
        if "backward" in br:
            feats[br] = feats[br][::-1]
        return feats


def engine_names(trt_major: str, batches=STEP_BATCHES) -> list[str]:
    return [f"primary_steps/step_{br}_b{B}_trt{trt_major}.ep" for B in batches for br in BRANCHES]

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

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


class _HostPark:

    RESERVE = 3 * 2**29

    def __init__(self) -> None:
        self.pin, self.bytes, self.reserve = True, 0, None

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if not self.pin:
            return t
        try:
            if self.reserve is None:
                self.reserve = torch.empty(self.RESERVE, dtype=torch.uint8).pin_memory()
            pinned = t.pin_memory()
            self.bytes += t.nbytes
            return pinned
        except RuntimeError as e:
            self.pin = False
            self.release()
            try:
                torch.zeros(1, device="cuda").add_(1)
            except RuntimeError:
                pass
            logger.warning(f"kiyo-ltx: no more pinned memory after {self.bytes / 2**30:.1f} GB, the rest uses ordinary "
                           f"memory (slower): {str(e).splitlines()[0]}")
            return t

    def release(self) -> None:
        self.reserve = None
        torch._C._host_emptyCache()


def _read_to_device(path: Path, device: torch.device, on_host=lambda name: False) -> dict[str, torch.Tensor]:
    out, park = {}, _HostPark()
    with open(path, "rb") as f:
        header, _, base = _header(f)
        for name, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            a, b = info["data_offsets"]
            f.seek(base + a)
            buf = bytearray(f.read(b - a))
            t = torch.frombuffer(buf, dtype=_DTYPES[info["dtype"]]) if b > a else torch.empty(0, dtype=_DTYPES[info["dtype"]])
            t = t.reshape(info["shape"])
            out[name] = park(t) if on_host(name) else t.to(device)
            del buf, t
    park.release()
    return out


def _stream_blocks(blocks: torch.nn.ModuleList, keep: int, device: torch.device) -> None:
    park = _HostPark()
    for block in list(blocks)[keep:]:
        tensors = list(block.parameters()) + list(block.buffers())
        parked = [(t, t.data if t.device.type == "cpu" else park(t.data.to("cpu"))) for t in tensors]
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
    park.release()
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


FP16_ENCODER_FP32_FROM = 5


def _fp32_norms(module: torch.nn.Module) -> None:
    for m in module.modules():
        if "norm" in type(m).__name__.lower():
            m.float()
            m.forward = (lambda f: lambda x, *a, **k: f(x.float(), *a, **k).to(x.dtype))(m.forward)


def _fp16_vae(encoder: torch.nn.Module, decoder: torch.nn.Module) -> None:
    _fp32_norms(encoder)
    _fp32_norms(decoder)
    for i, b in enumerate(encoder.down_blocks):
        if i >= FP16_ENCODER_FP32_FROM:
            b.float()
            if i == FP16_ENCODER_FP32_FROM:
                b.register_forward_pre_hook(lambda m, a: tuple(t.float() if torch.is_tensor(t) else t for t in a))
    for name in ("per_channel_statistics", "conv_norm_out", "conv_act", "conv_out"):
        getattr(encoder, name).float()


def _fp16_attention(module: torch.nn.Module) -> None:

    def half(fn):
        def call(q, k, v, heads, *mask):
            m = tuple(x.half() if torch.is_tensor(x) and x.is_floating_point() else x for x in mask)
            return fn(q.half(), k.half(), v.half(), heads, *m).to(q.dtype)

        return call

    for m in module.modules():
        for name in ("attention_function", "masked_attention_function"):
            if callable(getattr(m, name, None)):
                setattr(m, name, half(getattr(m, name)))


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
        older = torch.cuda.get_device_capability(self.device)[0] < 8
        self.vae_dtype = torch.float16 if older else self.dtype
        self.context = load_file(str(prompt))["video_encoding"].to(self.device, self.dtype)
        try:
            self.model = _load_transformer(tr, self.device, resident_blocks)
        except MemoryError as e:
            raise MemoryError("kiyo-ltx ran out of system memory: on this GPU it keeps part of the model in RAM (up to about "
                              "12 GB). Close other programs, or use kiyo-v4") from e
        self.encoder = ImageConditioner(str(vae), self.vae_dtype, self.device)._encoder_builder.build(
            device=self.device, dtype=self.vae_dtype
        ).eval()
        dec = VideoDecoder(str(vae), self.vae_dtype, self.device)._prepared_builder().build(
            device=self.device, dtype=self.vae_dtype
        )
        self.decoder = disable_memory_efficient_decode(dec.eval())
        if older:
            _fp16_attention(self.model)
            _fp16_vae(self.encoder, self.decoder)
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
            torch._C._host_emptyCache()
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
        video = (x.permute(3, 0, 1, 2)[None].float() / 127.5 - 1.0).to(self.vae_dtype)
        ref = self.encoder(video).to(self.dtype)
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
        frames = torch.cat(list(self.decoder.decode_video(out.to(self.vae_dtype), None, g)))
        out = (frames[:n].float().clamp(0, 1) * 255 + 0.5).to(torch.uint8).flip(-1).cpu().numpy()
        if self.resident_blocks is not None:
            del frames
            torch.cuda.empty_cache()
        return out

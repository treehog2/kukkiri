from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from kukkiri import trt_compat
from kukkiri.constants import DETECT_BATCH_SIZE, DETECT_CONF_THRESHOLD


class RFDETRSize(str, Enum):
    NANO = "nano"
    SMALL = "small"
    MEDIUM = "medium"
    BASE = "base"
    LARGE = "large"
    SEG_SMALL = "seg_small"
    SEG_MEDIUM = "seg_medium"


@dataclass
class Detection:
    box: tuple[int, int, int, int]
    score: float
    label: int = 1
    mask: np.ndarray | None = None


def _model_class(size: RFDETRSize):
    import rfdetr

    return {
        RFDETRSize.NANO: rfdetr.RFDETRNano,
        RFDETRSize.SMALL: rfdetr.RFDETRSmall,
        RFDETRSize.MEDIUM: rfdetr.RFDETRMedium,
        RFDETRSize.BASE: rfdetr.RFDETRBase,
        RFDETRSize.LARGE: rfdetr.RFDETRLarge,
        RFDETRSize.SEG_SMALL: rfdetr.RFDETRSegSmall,
        RFDETRSize.SEG_MEDIUM: rfdetr.RFDETRSegMedium,
    }[size]


def _fp16_onnx(onnx_path: Path) -> Path:
    out = onnx_path.with_name(onnx_path.stem + ".h16.onnx")
    if not out.exists():
        import onnx
        from onnxconverter_common import float16

        m = float16.convert_float_to_float16(onnx.load(str(onnx_path)), keep_io_types=False, disable_shape_infer=True)
        for n in m.graph.node:
            if n.op_type == "Cast" and n.attribute[0].i == onnx.TensorProto.FLOAT:
                n.attribute[0].i = onnx.TensorProto.FLOAT16
        onnx.save(m, str(out))
    return out


def _checkpoint_resolution(weights: str | Path) -> int | None:
    import torch

    args = torch.load(str(weights), map_location="cpu", weights_only=False).get("args") or {}
    res = (args if isinstance(args, dict) else vars(args)).get("resolution")
    return int(res) if res else None


class CensorDetector:
    def __init__(
        self,
        weights: str | Path,
        size: RFDETRSize = RFDETRSize.SMALL,
        threshold: float = DETECT_CONF_THRESHOLD,
        device: str = "cuda",
        grid_masks: bool = True,
    ):
        self.threshold = threshold
        self.weights = str(weights)
        self.size = RFDETRSize(size)
        res = _checkpoint_resolution(weights)
        kw = {"resolution": res} if res else {}
        self.model = _model_class(size)(pretrain_weights=str(weights), device=device, **kw)
        if grid_masks:
            pp = getattr(getattr(self.model, "model", None), "postprocess", None)
            if pp is not None and hasattr(pp, "upsample_masks_to_image_size"):
                pp.upsample_masks_to_image_size = False

    def detect_frame(self, frame_bgr: np.ndarray) -> list[Detection]:
        dets = self.model.predict(frame_bgr[..., ::-1].copy(), threshold=self.threshold)
        out = []
        ids = dets.class_id if dets.class_id is not None else [1] * len(dets.xyxy)
        masks = getattr(dets, "mask", None)
        for i, ((x0, y0, x1, y1), score, cid) in enumerate(zip(dets.xyxy, dets.confidence, ids)):
            m = np.asarray(masks[i]).astype(bool) if masks is not None else None
            out.append(Detection((int(x0), int(y0), int(x1), int(y1)), float(score), int(cid), m))
        return out

    def onnx_for_engine(self, batch: int = 32, half: bool = True) -> Path:
        exp = Path(self.weights).parent / (f"export_b{batch}_fp16" if half else f"export_b{batch}")
        onnx_path = exp / f"rfdetr-{self.size.value.replace('_', '-')}.onnx"
        shipped = onnx_path.with_name(onnx_path.stem + ".h16.onnx")
        if half and shipped.exists():
            return shipped
        if not onnx_path.exists():
            m = self.model
            m.predict(np.zeros((1080, 1920, 3), np.uint8), threshold=self.threshold, include_source_image=False)
            m.export(
                output_dir=str(onnx_path.parent), output_name=onnx_path.name, batch_size=batch, format="onnx", fp16=half,
                verbose=False, opset_version=17,
            )
        return _fp16_onnx(onnx_path) if half else onnx_path

    def _trt_init(self, batch: int = 32) -> None:
        import tensorrt as trt
        import torch

        wdir = Path(self.weights).parent
        half = trt_compat.strongly_typed(trt)
        exp = wdir / (f"export_b{batch}_fp16" if half else f"export_b{batch}")
        eng_path = exp / f"det_b{batch}_h16_trt{trt_compat.trt_major(trt)}.engine"
        logger = trt.Logger(trt.Logger.WARNING)
        if not eng_path.exists():
            build_from = self.onnx_for_engine(batch, half)
            builder = trt.Builder(logger)
            network = builder.create_network(trt_compat.network_flags(trt))
            parser = trt.OnnxParser(network, logger)
            if not parser.parse(build_from.read_bytes()):
                raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            config = builder.create_builder_config()
            trt_compat.configure_fp16(trt, config)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
            eng_path.write_bytes(bytes(builder.build_serialized_network(network, config)))
        self._trt_engine = trt.Runtime(logger).deserialize_cuda_engine(eng_path.read_bytes())
        self._trt_ctx = self._trt_engine.create_execution_context()
        self._trt_bufs = trt_compat.io_buffers(trt, self._trt_engine, "cuda")
        for n, buf in self._trt_bufs.items():
            self._trt_ctx.set_tensor_address(n, buf.data_ptr())
        self._trt_batch = batch
        self._trt_res = int(self.model.model.resolution)
        self._trt_mean = torch.tensor(self.model.means, device="cuda").view(1, 3, 1, 1)
        self._trt_std = torch.tensor(self.model.stds, device="cuda").view(1, 3, 1, 1)

    def detect_frames_trt(self, frames_bgr: list, batch_size: int = 0) -> list[list[Detection]]:
        import torch
        import torch.nn.functional as F

        if not hasattr(self, "_trt_ctx"):
            self._trt_init()
        B, res, bufs = self._trt_batch, self._trt_res, self._trt_bufs
        results: list[list[Detection]] = []
        with torch.no_grad():
            for i in range(0, len(frames_bgr), B):
                fr = frames_bgr[i : i + B]
                x = torch.stack(
                    [
                        (f if isinstance(f, torch.Tensor) else torch.from_numpy(f)).to("cuda", non_blocking=True)
                        for f in fr
                    ]
                )
                H, W = x.shape[1:3]
                x = x.flip(-1).permute(0, 3, 1, 2).float().div_(255.0)
                x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=False)
                x = (x - self._trt_mean) / self._trt_std
                bufs["input"][: x.shape[0]].copy_(x)
                self._trt_ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
                n = x.shape[0]
                scores, cls = bufs["labels"][:n].float().sigmoid().max(-1)
                keep = scores > self.threshold
                boxes = bufs["dets"][:n].float()
                xyxy = torch.stack(
                    [
                        (boxes[..., 0] - boxes[..., 2] / 2) * W,
                        (boxes[..., 1] - boxes[..., 3] / 2) * H,
                        (boxes[..., 0] + boxes[..., 2] / 2) * W,
                        (boxes[..., 1] + boxes[..., 3] / 2) * H,
                    ],
                    -1,
                )
                keep_c, xyxy_c, sc_c, cls_c = (
                    keep.cpu().numpy(),
                    xyxy.cpu().numpy(),
                    scores.cpu().numpy(),
                    cls.cpu().numpy(),
                )
                sel = [np.nonzero(keep_c[b])[0] for b in range(n)]
                flat = [(b, q) for b in range(n) for q in sel[b]]
                if flat:
                    bi = torch.as_tensor([b for b, _ in flat], device=keep.device)
                    qi = torch.as_tensor([q for _, q in flat], device=keep.device)
                    mk_sel = (bufs["masks"][:n][bi, qi] > 0).cpu().numpy()
                else:
                    mk_sel = None
                k = 0
                for b in range(n):
                    idx = sel[b]
                    results.append(
                        [
                            Detection(
                                tuple(int(v) for v in xyxy_c[b, q]),
                                float(sc_c[b, q]),
                                int(cls_c[b, q]),
                                mk_sel[k + j].copy(),
                            )
                            for j, q in enumerate(idx)
                        ]
                    )
                    k += len(idx)
        return results

    def detect_frames_fast(self, frames_bgr: list, batch_size: int = DETECT_BATCH_SIZE) -> list[list[Detection]]:
        import torch
        import torch.nn.functional as F

        m = self.model
        if not getattr(self, "_warm", False):
            m.predict(np.zeros((1080, 1920, 3), np.uint8), threshold=self.threshold, include_source_image=False)
            self._warm = True
        net, pp, dev = m.model.model, m.model.postprocess, m.model.device
        res = int(m.model.resolution)
        mean = torch.tensor(m.means, device=dev).view(1, 3, 1, 1)
        std = torch.tensor(m.stds, device=dev).view(1, 3, 1, 1)
        results: list[list[Detection]] = []
        with torch.no_grad():
            for i in range(0, len(frames_bgr), batch_size):
                fr = frames_bgr[i : i + batch_size]
                x = torch.stack(
                    [(f if isinstance(f, torch.Tensor) else torch.from_numpy(f)).to(dev, non_blocking=True) for f in fr]
                )
                H, W = x.shape[1:3]
                x = x.flip(-1).permute(0, 3, 1, 2).float().div_(255.0)
                x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=False)
                x = (x - mean) / std
                with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    pred = net(x)
                if isinstance(pred, tuple):
                    d = {"pred_logits": pred[1], "pred_boxes": pred[0]}
                    if len(pred) == 3:
                        d["pred_masks"] = pred[2]
                    pred = d
                sizes = torch.tensor([[H, W]] * x.shape[0], device=dev)
                outs = pp(pred, target_sizes=sizes, score_threshold=self.threshold)
                for r in outs:
                    keep = r["scores"] > self.threshold
                    boxes = r["boxes"][keep].cpu().numpy()
                    scores = r["scores"][keep].cpu().numpy()
                    labels = r["labels"][keep].cpu().numpy()
                    masks = r.get("masks")
                    masks = masks[keep].cpu().numpy() if masks is not None else None
                    results.append(
                        [
                            Detection(
                                (int(b[0]), int(b[1]), int(b[2]), int(b[3])),
                                float(s),
                                int(c),
                                (np.squeeze(masks[k]).astype(bool) if masks is not None else None),
                            )
                            for k, (b, s, c) in enumerate(zip(boxes, scores, labels))
                        ]
                    )
        return results

    def detect_frames(self, frames_bgr: list[np.ndarray], batch_size: int = DETECT_BATCH_SIZE) -> list[list[Detection]]:
        import torch

        if getattr(self, "trt", False):
            return self.detect_frames_trt(frames_bgr)
        if getattr(self, "fast", True):
            return self.detect_frames_fast(frames_bgr, batch_size)
        results: list[list[Detection]] = []
        dev = self.model.model.device
        for i in range(0, len(frames_bgr), batch_size):
            batch = torch.stack(
                [
                    (f if isinstance(f, torch.Tensor) else torch.from_numpy(f)).to(dev, non_blocking=True)
                    for f in frames_bgr[i : i + batch_size]
                ]
            )
            batch = batch.flip(-1).permute(0, 3, 1, 2).float().div_(255.0)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                dets_list = self.model.predict(list(batch), threshold=self.threshold, include_source_image=False)
            if not isinstance(dets_list, list):
                dets_list = [dets_list]
            for dets in dets_list:
                masks = getattr(dets, "mask", None)
                results.append(
                    [
                        Detection(
                            (int(a), int(b), int(c), int(d)),
                            float(s),
                            int(cid),
                            np.asarray(masks[i]).astype(bool) if masks is not None else None,
                        )
                        for i, ((a, b, c, d), s, cid) in enumerate(
                            zip(
                                dets.xyxy,
                                dets.confidence,
                                dets.class_id if dets.class_id is not None else [1] * len(dets.xyxy),
                            )
                        )
                    ]
                )
        return results

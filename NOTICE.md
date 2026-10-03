# Third-party notices

kukkiri, Copyright (c) 2026 kukkiri authors, is licensed under the GNU General Public License v3.0 only
(`LICENSE`), with the additional permission for NVIDIA libraries in `LICENSE-EXCEPTION.md`. It includes the
third-party code listed under "Vendored code", whose license text is in `licenses/`, and uses the libraries listed
under "Libraries used", which keep their own licenses whether installed separately or bundled in the Docker image or the Windows
installer.

## Vendored code

Upstream: OpenMMLab MMagic v1.2.0 (https://github.com/open-mmlab/mmagic, tag `v1.2.0`, commit
`c749dcc7172d198ac2a27c3e5a4d2181640f0fd5`): `mmagic/models/editors/basicvsr_plusplus_net/basicvsr_plusplus_net.py`,
`mmagic/models/editors/basicvsr/basicvsr_net.py`, `mmagic/models/archs/` and `mmagic/models/utils/flow_warp.py`.
License: Apache-2.0; Copyright (c) OpenMMLab, Copyright 2023 MMagic Authors, Copyright 2020 MMEditing Authors. The
upstream license file is reproduced unchanged in `licenses/APACHE-2.0.txt`.

| kukkiri file | derived from | modifications |
|---|---|---|
| `kukkiri/basicvsrpp_net.py` | BasicVSR++ network, SPyNet, flow warp, PixelShufflePack, second-order deformable alignment | mmengine/registry/logging removed, ConvModule replaced by conv+ReLU with identical parameter names, deformable conv on `torchvision.ops.deform_conv2d`, same-resolution output, fp16/CUDA-graph friendly forward; see the module docstring |
| `kukkiri/trt_primary.py` | BasicVSR++ forward pass and second-order propagation (`BasicVSRPlusPlusNet.forward` / `propagate`) | propagation loop re-driven from Python, each per-frame propagation step compiled as a TensorRT engine and replayed under a CUDA graph; see the module docstring |

## Libraries used (not vendored)

| component | license | note |
|---|---|---|
| RF-DETR | Apache-2.0 | detector library; the detector is fine-tuned from its published COCO checkpoint |
| PyTorch, torchvision | BSD-3 | |
| NumPy | BSD-3 | |
| OpenCV | Apache-2.0 | |
| ONNX / onnxconverter-common | Apache-2.0 / MIT | detector export |
| safetensors | Apache-2.0 | reading the published weights |
| PyNvVideoCodec | MIT, over the NVIDIA Video Codec SDK (NVIDIA SDK license) | GPU decode / encode |
| TensorRT | NVIDIA TensorRT license (proprietary) | |
| CUDA runtime (cudart64_12.dll, Windows installer) | NVIDIA CUDA EULA (redistributable) | loaded by PyNvVideoCodec |
| Microsoft Visual C++ runtime (msvcp140*.dll, Windows installer) | Microsoft Visual C++ Redistributable license | |
| Torch-TensorRT | BSD-3 | |
| PySide6 (Qt for Python) | LGPL-3.0 | desktop app only (`kukkiri[gui]`) |
| FFmpeg (system binary) | LGPL-2.1+, or GPL-2+ for builds with libx264 | called as an external process for probing, remuxing, audio, and re-encoding inputs the GPU decoder cannot read to H.264; the Windows installer bundles the GPL build ffmpeg n8.1.3 from github.com/BtbN/FFmpeg-Builds (release autobuild-2026-09-22-13-18), whose source is ffmpeg.org release 8.1.3 |

TensorRT engines are built locally on first run and contain converted model weights; they fall under the terms of
the weights they were built from.

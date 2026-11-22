# kukkiri CLI on CUDA 13 (torch 2.14, TensorRT 11, NVDEC/NVENC). The host needs an NVIDIA driver >= 580 and the
# NVIDIA Container Toolkit.
#
#   docker build -t kukkiri .
#   docker run --rm --gpus all -v "$PWD:/data" -v "$PWD/weights:/weights" kukkiri \
#       restore /data/input.mp4 -o /data/restored.mp4 --weights /weights
#
# TensorRT engines are built into the weights directory on the first run, so mount it writable to keep them.
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1 NVIDIA_DRIVER_CAPABILITIES=all
# build-essential + python3-dev: torch.compile (kiyo-ltx) builds its kernels with the system C compiler
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.14.0 torchvision==0.29.0
RUN pip install numpy==2.3.5 opencv-python-headless==5.0.0.93 rfdetr==1.10.1 PyNvVideoCodec==2.2.2 \
    tensorrt-cu13==11.3.0.99 onnx==1.22.0 onnxconverter-common==1.16.0 safetensors==0.8.0
# torch_tensorrt pins tensorrt<11.2 but works with 11.3; --no-deps keeps pip from downgrading it
RUN pip install --no-deps --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple \
    torch_tensorrt==2.14.0+cu130 dllist==2.0.0 psutil==7.2.2

# kiyo-ltx: Lightricks' LTX-2 packages at a fixed commit (--no-deps: their pins are older) and what they import
# torchaudio 2.11 declares torch==2.11 but imports fine on 2.14 (the LTX packages only import it): --no-deps
RUN pip install --no-deps --index-url https://download.pytorch.org/whl/cu130 torchaudio==2.11.0 \
    && pip install transformers==5.17.0 accelerate==1.15.0 einops==0.8.2 scipy==1.18.1 colour-science==0.4.7 av==18.1.0 \
       OpenImageIO==3.2.2.0 pillow tqdm \
    && pip install --no-deps "ltx-core @ https://github.com/Lightricks/LTX-2/archive/f0f2f8607189c78a72420b2b63cb1975b5eca179.zip#subdirectory=packages/ltx-core" \
       "ltx-pipelines @ https://github.com/Lightricks/LTX-2/archive/f0f2f8607189c78a72420b2b63cb1975b5eca179.zip#subdirectory=packages/ltx-pipelines"

WORKDIR /src
COPY pyproject.toml README.md LICENSE LICENSE-EXCEPTION.md NOTICE.md ./
COPY licenses/ licenses/
COPY kukkiri/ kukkiri/
RUN pip install --no-deps . && rm -rf /src

WORKDIR /data
ENTRYPOINT ["kukkiri"]
CMD ["--help"]

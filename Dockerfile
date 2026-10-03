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
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.14.0 torchvision==0.29.0
RUN pip install numpy==2.3.5 opencv-python-headless==5.0.0.93 rfdetr==1.10.1 PyNvVideoCodec==2.2.2 \
    tensorrt-cu13==11.3.0.99 onnx==1.22.0 onnxconverter-common==1.16.0 safetensors==0.8.0
# torch_tensorrt pins tensorrt<11.2 but works with 11.3; --no-deps keeps pip from downgrading it
RUN pip install --no-deps --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple \
    torch_tensorrt==2.14.0+cu130 dllist==2.0.0 psutil==7.2.2

WORKDIR /src
COPY pyproject.toml README.md LICENSE LICENSE-EXCEPTION.md NOTICE.md ./
COPY licenses/ licenses/
COPY kukkiri/ kukkiri/
RUN pip install --no-deps . && rm -rf /src

WORKDIR /data
ENTRYPOINT ["kukkiri"]
CMD ["--help"]

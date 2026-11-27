# kukkiri CLI on CUDA 13 (torch 2.14, TensorRT 11, NVDEC/NVENC), published as ghcr.io/treehog2/kukkiri. The host needs
# an NVIDIA driver >= 580 and the NVIDIA Container Toolkit.
#
#   docker run --rm --gpus all -v "$PWD:/data" -v "$PWD/weights:/weights" ghcr.io/treehog2/kukkiri \
#       restore /data/input.mp4 -o /data/restored.mp4 --weights /weights
#
# To build it yourself: docker build -t kukkiri .
#
# TensorRT engines are built into the weights directory on the first run, so mount it writable to keep them.
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1 NVIDIA_DRIVER_CAPABILITIES=all
# build-essential + python3-dev: torch.compile (kiyo-ltx) builds its kernels with the system C compiler
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# the same pinned set as install.sh (packaging/linux)
COPY packaging/linux/requirements-torch.txt /tmp/req/
RUN pip install -r /tmp/req/requirements-torch.txt
COPY packaging/linux/requirements-base.txt /tmp/req/
RUN pip install -r /tmp/req/requirements-base.txt
COPY packaging/linux/requirements-nodeps.txt /tmp/req/
RUN pip install --no-deps -r /tmp/req/requirements-nodeps.txt

WORKDIR /src
COPY pyproject.toml README.md LICENSE LICENSE-EXCEPTION.md NOTICE.md ./
COPY licenses/ licenses/
COPY kukkiri/ kukkiri/
RUN pip install --no-deps . && rm -rf /src

WORKDIR /data
ENTRYPOINT ["kukkiri"]
CMD ["--help"]

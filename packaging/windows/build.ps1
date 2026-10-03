# Assemble the Windows install folder (dist\stage): a private Python with the pinned GPU stack, kukkiri and ffmpeg.
# The model weights are not included: kukkiri downloads them into the per-user folder on first use. Then kukkiri.iss
# wraps it into the installer. Run from the repo root on Windows (CI: windows-latest):
#   powershell -File packaging\windows\build.ps1
param(
    [string]$Out = "dist"
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# pinned inputs (same GPU stack as the Dockerfile)
$PythonUrl = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip"  # last 3.12 with an embeddable build
$FfmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-09-22-13-18/ffmpeg-n8.1.3-win64-gpl-8.1.zip"

$Stage = Join-Path $Out "stage"
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
New-Item -ItemType Directory -Force -Path $Stage, "$Out\dl" | Out-Null

# 1. Python (embeddable), with site-packages enabled so pip can install into it
Invoke-WebRequest $PythonUrl -OutFile "$Out\dl\python.zip"
Expand-Archive "$Out\dl\python.zip" -DestinationPath $Stage
$pth = Get-ChildItem $Stage -Filter "python*._pth" | Select-Object -First 1
Set-Content $pth.FullName "python312.zip`r`n.`r`nLib\site-packages`r`nimport site"
Invoke-WebRequest "https://bootstrap.pypa.io/get-pip.py" -OutFile "$Out\dl\get-pip.py"
& "$Stage\python.exe" "$Out\dl\get-pip.py" --no-warn-script-location
if ($LASTEXITCODE) { throw "get-pip failed" }
$py = "$Stage\python.exe"
$pip = @("-m", "pip", "install", "--no-cache-dir", "--no-warn-script-location")

# 2. the GPU stack and kukkiri. Wheels are fetched / built with the regular Python on the PATH (same 3.12): the
# embeddable Python ignores PYTHONPATH, so pip cannot build source-only packages (tensorrt-cu13) inside it.
$wh = Join-Path $Out "wheels"
$idx = @("--index-url", "https://pypi.org/simple", "--extra-index-url", "https://download.pytorch.org/whl/cu130",
         "--extra-index-url", "https://pypi.nvidia.com")
$pins = @("torch==2.14.0+cu130", "torchvision==0.29.0+cu130", "numpy==2.3.5", "opencv-python-headless==5.0.0.93",
          "rfdetr==1.10.1", "PyNvVideoCodec==2.2.2", "tensorrt-cu13==11.3.0.99", "onnx==1.22.0",
          "onnxconverter-common==1.16.0", "safetensors==0.8.0", "PySide6-Essentials==6.11.2")
# torch_tensorrt pins tensorrt<11.2 but works with 11.3; --no-deps keeps pip from downgrading it
$nodeps = @("torch_tensorrt==2.14.0+cu130", "dllist==2.0.0", "psutil==7.2.2")
& python -m pip wheel --wheel-dir $wh @idx @pins
if ($LASTEXITCODE) { throw "fetching wheels failed" }
& python -m pip wheel --no-deps --wheel-dir $wh @idx @nodeps .
if ($LASTEXITCODE) { throw "fetching no-deps wheels failed" }
& $py @pip --no-index --find-links $wh @pins
if ($LASTEXITCODE) { throw "dependency install failed" }
& $py @pip --no-index --find-links $wh --no-deps @nodeps kukkiri
if ($LASTEXITCODE) { throw "kukkiri install failed" }
Remove-Item -Recurse -Force $wh
# TensorRT's engine builder carries one resource DLL per GPU generation; drop the datacenter-only ones (A100 sm80,
# H100 sm90, B200 sm100: ~950 MB) that no Windows desktop has. The PTX one stays as the fallback for newer GPUs.
foreach ($sm in "80", "90", "100") { Remove-Item "$Stage\Lib\site-packages\tensorrt_libs\nvinfer_builder_resource_sm${sm}_11.dll" }

# DLLs a clean Windows lacks. PyNvVideoCodec's CUDA 13 module still links the CUDA 12 runtime (cudart64_12.dll; torch
# only ships CUDA 13's), and the Visual C++ runtime (msvcp140*.dll) comes with the Visual C++ Redistributable, which
# a fresh install may not have. Both go next to what loads them; app-local copies are Microsoft- and NVIDIA-sanctioned.
& python -m pip download --no-cache-dir --no-deps --only-binary=:all: --platform win_amd64 -d "$Out\dl" nvidia-cuda-runtime-cu12==12.9.79
if ($LASTEXITCODE) { throw "cuda runtime download failed" }
Copy-Item (Get-ChildItem "$Out\dl\nvidia_cuda_runtime_cu12-*.whl").FullName "$Out\dl\cudart.zip"  # Expand-Archive only takes .zip
Expand-Archive "$Out\dl\cudart.zip" -DestinationPath "$Out\dl\cudart"
Copy-Item (Get-ChildItem "$Out\dl\cudart" -Recurse -Filter "cudart64_12.dll").FullName "$Stage\Lib\site-packages\PyNvVideoCodec\"
# the runtime as one current set (the embeddable Python's vcruntime140 may be older): torch 2.14 also needs
# vcruntime140_threads.dll, which only recent redistributables install, so an older Windows fails to load torch
foreach ($f in "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "vcruntime140.dll", "vcruntime140_1.dll",
               "vcruntime140_threads.dll") {
    Copy-Item "$env:SystemRoot\System32\$f" "$Stage\" -Force
}

# 3. ffmpeg + ffprobe (GPL build: libx264)
Invoke-WebRequest $FfmpegUrl -OutFile "$Out\dl\ffmpeg.zip"
Expand-Archive "$Out\dl\ffmpeg.zip" -DestinationPath "$Out\dl\ffmpeg"
$ff = Get-ChildItem "$Out\dl\ffmpeg" -Directory | Select-Object -First 1
New-Item -ItemType Directory -Force -Path "$Stage\ffmpeg" | Out-Null
Copy-Item "$($ff.FullName)\bin\ffmpeg.exe", "$($ff.FullName)\bin\ffprobe.exe" "$Stage\ffmpeg\"
Copy-Item "$($ff.FullName)\LICENSE.txt" "$Stage\ffmpeg\LICENSE.txt"

# 4. launchers, licenses
Copy-Item "packaging\windows\kukkiri-gui.pyw" "$Stage\"
Set-Content "$Stage\kukkiri.cmd" "@echo off`r`nsetlocal`r`nset PATH=%~dp0ffmpeg;%PATH%`r`n`"%~dp0python.exe`" -m kukkiri %*"
Copy-Item "LICENSE", "LICENSE-EXCEPTION.md", "NOTICE.md", "README.md" "$Stage\"
Copy-Item -Recurse "licenses" "$Stage\licenses"
Write-Output ("stage: {0:N0} MB" -f ((Get-ChildItem $Stage -Recurse -File | Measure-Object Length -Sum).Sum / 1MB))

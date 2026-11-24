<img src="docs/media/logo.png" width="96" alt="kukkiri logo">

# kukkiri

Restore mosaic-censored video. kukkiri finds the pixelated regions in every frame, tracks them through the scene and
reconstructs what is under them with a video restoration model that uses many frames at once, then writes the result
back into the original video with the source audio.

It runs as one streaming pass on an NVIDIA GPU: decoding, detection, restoration and encoding overlap, and only a
bounded window of frames is ever held in memory, so a two-hour film needs no more memory than a two-minute clip.

It also works the other way: [`kukkiri add-mosaic`](#add-mosaics) adds a block mosaic over the bare genitals in a video.

[![original, mosaic and kukkiri side by side](docs/media/strawberry.webp)](docs/media/strawberry.mp4)

*Left to right: original, mosaic, kukkiri (kiyo-ltx-v1, `--sensitivity high`). An AI-generated clip with a synthetic mosaic; click for the full-quality video.*

**On Windows:** download the installer, [kukkiri-setup.exe](https://huggingface.co/treehog/kukkiri/resolve/main/windows/kukkiri-setup.exe) (about 3 GB), and follow the
[Windows guide](docs/windows.md), which walks through installing and using the app with screenshots.

## Requirements

- An NVIDIA GTX 16-series / RTX 20-series or newer GPU (compute capability 7.5+) and driver 580 or newer
- Python 3.10+
- `ffmpeg` on the `PATH`

## Install

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # match your CUDA
pip install .
```

Or with Docker (Linux, NVIDIA Container Toolkit, driver 580 or newer):

```
docker build -t kukkiri .
docker run --rm --gpus all -v "$PWD:/data" -v "$PWD/weights:/weights" kukkiri \
    restore /data/input.mp4 -o /data/restored.mp4 --weights /weights
```

The image pins the tested versions of the main dependencies, kiyo-ltx's included. Keep the weights directory mounted writable so the TensorRT
engines built on the first run are reused.

## Weights

The kiyo weights are licensed under the same terms as the code (GPL-3.0-only with the NVIDIA exception below); the
kiyo-ltx weights are under the LTX-2.x Community License (see kiyo-ltx below).
They are published at [treehog/kukkiri](https://huggingface.co/treehog/kukkiri) on Hugging Face, one folder per version.

The first restore downloads them: the detector (RF-DETR seg-medium, 141 MB, plus its 65 MB ONNX export) and the
restorer (BasicVSR++, 39 MB) as safetensors, checked against the SHA-256 in `kukkiri/weights.py`. They go to a per-user folder
(`%LOCALAPPDATA%\kukkiri\weights\<version>` on Windows, `~/.cache/kukkiri/weights/<version>` elsewhere), or to the folder in
`KUKKIRI_WEIGHTS` or `--weights`. The Windows app downloads them on first use into the `models` folder inside the
folder you installed it to (kiyo-ltx needs about 15 GB there); versions already downloaded to the per-user folder by an
older install stay where they are. To fetch them ahead of time:

```
kukkiri download-weights [folder]
```

TensorRT engines are built into the same folder on the first run, which takes a few minutes and is cached afterwards.

## kiyo-ltx

`kiyo-ltx-v1` restores with Lightricks' [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) video diffusion model,
fine-tuned to restore mosaics, for clearer detail on larger mosaics; small ones still go to kiyo. It needs:

- a GPU with at least **8 GB** of memory. Below 32 GB it keeps part of the model in system memory and streams it to
  the GPU as needed: the same result, but slower (about 1.5x on a 24 GB card, 1.75x on 16 GB, in our tests). Below
  15 GB it also runs in three passes (find the mosaics, restore the large ones with LTX, then the small ones and the
  video), so the models are never on the GPU together; it needs about 8 GB free, so close other GPU programs on an
  8 GB card. It stops with a message on less than 7.5 GB.
  Cards older than the RTX 30 series (RTX 20, GTX 16) work too, but slowly: they lack bf16 support, so kiyo-ltx
  computes some steps in smaller pieces (about 90 s per 3 s of video on a T4 in our tests).
- a **15 GB** download on first use (the folder `ltx-v1/` on Hugging Face)
- a few minutes on the first run on a machine, to build its GPU kernels (cached in the weights folder afterwards)

```
kukkiri restore input.mp4 --model kiyo-ltx-v1
```

Videos above 30 fps are restored at 30 fps by default (59.94 becomes 29.97), which halves the work; `--keep-fps` keeps
the full frame rate at about twice the time. The Windows app and the Docker image include everything it needs. With
pip, also install its packages:

```
pip install ".[ltx]"
pip install --no-deps --index-url https://download.pytorch.org/whl/cu130 torchaudio==2.11.0
pip install --no-deps "ltx-core @ https://github.com/Lightricks/LTX-2/archive/f0f2f8607189c78a72420b2b63cb1975b5eca179.zip#subdirectory=packages/ltx-core"
pip install --no-deps "ltx-pipelines @ https://github.com/Lightricks/LTX-2/archive/f0f2f8607189c78a72420b2b63cb1975b5eca179.zip#subdirectory=packages/ltx-pipelines"
```

The kiyo-ltx weights are a derivative of LTX-2.x and are licensed under the **LTX-2.x Community License**, not the GPL
(see `ltx-v1/LICENSE-LTX-2.x.txt` and `ltx-v1/NOTICE.md` on Hugging Face). That licence includes use restrictions, and
it asks that content made with the model be disclosed as machine-generated wherever it is published.

## VR

Side-by-side 180° VR videos (two square eyes next to each other) are restored with `--vr`, with any model:

```
kukkiri restore input.mp4 --model kiyo-ltx-v1 --vr sbs
kukkiri restore input.mp4 --model kiyo-ltx-v1 --vr sbs --vr-lens fisheye
```

Each eye is searched at its full resolution with a detector trained on VR footage (downloaded on first use into the
weights folder's `vr/` subfolder, about 210 MB), and every restored area stays inside its own eye. Some studios pixelate
the camera's fisheye image before converting it: their mosaic blocks bend into fans towards the bottom of each eye.
For those add `--vr-lens fisheye`, which restores each area in the eye's fisheye view and maps it back. VR is encoded at up
to 80 Mbit/s and ignores `--max-1080p`. In the app: Advanced → VR video. VR with kiyo-ltx needs a GPU with 32 GB
(two 4K eyes do not fit next to the model on smaller cards yet); on smaller cards restore VR with kiyo-v4.

## Usage

```
kukkiri restore input.mp4 -o restored.mp4
```

Without `-o` the result goes next to the input as `<name>.kukkiri-kiyo-v4.mp4`, named after the model. `--model`
picks the model (the latest by default; `kiyo-v3` is also available) and `--previews` also writes up to three
watermarked before/after clips of the most restored moments.

Before restoring, kukkiri samples the file's keyframes and skips inputs with nothing to restore: videos that are
already uncensored, copies another tool has already processed, or footage with no censoring at all. Pass
`--skip-check` to restore anyway.

Output keeps the source resolution. `--max-1080p` serves larger sources (4K) at 1080p instead, which is faster and
gives smaller files.

`--sensitivity high` also detects faint or small mosaics (at the risk of touching things that are not mosaics);
`--sensitivity low` makes fewer false detections. The default, `normal`, suits most videos.

To restore many videos, pass several files and/or folders (folders are searched recursively) and an output folder:

```
kukkiri restore movies/ extra.mkv -o restored/
```

Files are restored one after another; a file that is declined or fails is reported and the batch goes on, with a
summary at the end. Earlier kukkiri outputs (`*.kukkiri*.mp4`) in those folders are left out.

To restore only part of a video, for example a 30-second preview:

```
kukkiri restore input.mp4 -o section.mp4 --start 37:45 --duration 30
```

The section is cut frame-accurately before restoring and the output contains only that section. The input check is
skipped for sections.

Inputs in containers or codecs the GPU decoder cannot read directly (AVI, WMV, MPEG-TS, MPEG-PS, ...) are first
remuxed or re-encoded to H.264 MP4 automatically.

## Add mosaics

kukkiri can also do the opposite: add a block mosaic over the bare genitals in a video, with square blocks of about a
36th of the frame height on a grid fixed to the frame.

```
kukkiri add-mosaic input.mp4
```

The result goes next to the input as `<name>.kukkiri-mosaic.mp4` (or to `-o`); several files and folders work as
with `restore`. `--block` sets the block size in pixels; `--sensitivity high` misses fewer (and may mosaic a few
things that are not genitals). It uses the detector only, so the kiyo-v4 weights (240 MB)
are enough. It is best effort: a frame the detector misses for longer than a few frames stays uncensored, so check
the result before relying on it.

## Desktop app

```
pip install ".[gui]"
kukkiri-gui
```

Add videos or whole folders with the buttons or by dropping them on the window, choose the weights folder once, and press Start.
Files are restored one after another; each row shows its progress, speed and time left, and a finished row opens on
double-click. A row's ✂ button restores only a section of that video. Stop ends the current file (its partial output
is removed) and pauses the rest of the queue. The window comes in English, Korean and Chinese (simplified and
traditional), shows GPU, CPU and memory use, and **System check** tests the GPU, driver, disk, model files and ffmpeg
(also `kukkiri check` on the command line). **Save log** writes the full log to a text file.

## How it works

1. **Detect.** An RF-DETR segmentation model marks mosaic regions in every frame.
2. **Track.** Overlapping detections are merged into regions and followed through the scene; short misses are
   bridged by interpolation.
3. **Restore.** Each region is cut out as a square window with a little context, scaled onto the model's canvas and
   restored in clips of 96 frames by BasicVSR++. Neighbouring clips overlap and are crossfaded, so there are no seams.
4. **Blend.** The restored region is feathered back into the frame just past the detected area, and the video is
   encoded on the GPU.

## Limitations

- Pixelated faces (identity mosaics) are recognised as such and left as they are; kukkiri never restores them.
- Content that another tool has already "decensored" into smooth shapes cannot be recovered; the input check
  declines such files instead.

## License

Copyright (c) 2026 kukkiri authors. The code is licensed under the GNU General Public License v3.0 only
(`LICENSE`); the vendored BasicVSR++ code keeps its Apache-2.0 license (`licenses/APACHE-2.0.txt`). Third-party
components and their licenses are listed in `NOTICE.md`. The model weights are released under the same license.
`LICENSE-EXCEPTION.md` grants an additional permission to combine kukkiri with NVIDIA's CUDA, TensorRT and Video
Codec SDK libraries. The kiyo-ltx weights are not GPL: they are a derivative of Lightricks' LTX-2.x, under the LTX-2.x
Community License. The kukkiri name and logo are not covered by the license; see `CONTRIBUTING.md`.

## Acknowledgements

- [Lada](https://github.com/ladaapp/lada), for showing what open mosaic restoration can do
- [BasicVSR++](https://arxiv.org/abs/2104.13371) (Chan et al., CVPR 2022), via OpenMMLab's
  [MMagic](https://github.com/open-mmlab/mmagic): the restoration network
- Roboflow's [RF-DETR](https://github.com/roboflow/rf-detr): the detector
- Lightricks' [LTX-2](https://github.com/Lightricks/LTX-2): the video diffusion model behind kiyo-ltx

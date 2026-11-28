<img src="docs/media/logo.png" width="96" alt="kukkiri logo">

# kukkiri

Restore mosaic-censored video. kukkiri finds the pixelated areas in every frame, follows them through the scene and
rebuilds what is under them, then writes the result back into the original video with its sound. It streams the
video, so a two-hour film needs no more memory than a two-minute clip.

It also works the other way: [`kukkiri add-mosaic`](#add-mosaics) censors a video with a block mosaic, the way a
censored release would.

[![original, mosaic and kukkiri side by side](docs/media/strawberry.webp)](docs/media/strawberry.mp4)

*Left to right: original, mosaic, kukkiri (kiyo-ltx-v1, `--sensitivity high`). An AI-generated clip with a synthetic mosaic; click for the full-quality video.*

## Requirements

- An NVIDIA GTX 16-series / RTX 20-series or newer GPU (compute capability 7.5+) and driver 580 or newer, or an
  Apple Silicon Mac ([experimental](#mac-apple-silicon-experimental))
- With pip: Python 3.10+ and `ffmpeg` on the `PATH` (the Windows app includes both)

## Install

### Windows

Download the installer, [kukkiri-setup.exe](https://huggingface.co/treehog/kukkiri/resolve/main/windows/kukkiri-setup.exe)
(about 3 GB), and follow the [Windows guide](docs/windows.md), which walks through installing and using the app with
screenshots.

### Linux

```
git clone https://github.com/treehog2/kukkiri && cd kukkiri
./install.sh            # add --gui for the desktop app
```

It checks the NVIDIA driver (580 or newer), Python (3.12 or newer) and ffmpeg, then installs kukkiri and kiyo-ltx's
packages into `.venv` (about 10 GB) and runs `kukkiri check`. kiyo-ltx is fastest with a C compiler and Python's
headers installed (`sudo apt install build-essential python3-dev`). To install by hand instead, the pinned packages
are in [packaging/linux](packaging/linux).

Or with Docker (NVIDIA Container Toolkit, driver 580 or newer); the image includes kiyo-ltx's packages:

```
docker run --rm --gpus all -v "$PWD:/data" -v "$PWD/weights:/weights" ghcr.io/treehog2/kukkiri \
    restore /data/input.mp4 -o /data/restored.mp4 --weights /weights
```

To build the image yourself: `docker build -t kukkiri .`

Keep the weights folder mounted writable, so the GPU engines built on the first run are reused.

The desktop app (`./install.sh --gui`, then `kukkiri-gui`) works as in the [Windows guide](docs/windows.md).

### Mac (Apple Silicon, experimental)

```
brew install ffmpeg
git clone https://github.com/treehog2/kukkiri && cd kukkiri
python3 -m venv .venv && source .venv/bin/activate    # Python 3.10 or newer
pip install .
kukkiri restore input.mp4
```

- `restore` with kiyo-v4 and `add-mosaic` work; kiyo-ltx and VR need an NVIDIA GPU.
- A 2-minute 720p film takes about 2 minutes on a high-end chip; smaller chips will be slower.
- It uses about 3 GB of memory at 1080p and 9 GB at 4K; `--max-1080p` lowers that.

### ComfyUI

[ComfyUI-kukkiri](https://github.com/treehog2/ComfyUI-kukkiri) adds Kukkiri Mosaic Restore and Kukkiri Mosaic Add
nodes (they run an installed kukkiri), plus a core-nodes workflow for the kiyo-ltx LoRA alone.

## Models

| | kiyo-v4 (default) | kiyo-ltx-v1 |
|---|---|---|
| Best for | speed | the clearest detail on larger mosaics |
| GPU memory | any supported GPU | 8 GB or more (slower below 32 GB) |
| First-use download | about 240 MB | about 15 GB |
| Speed on a high-end GPU | a 2-minute 720p film in about 20 seconds | a 10-second 1080p clip in about 45 seconds, including loading the model |

Smaller GPUs are slower. The weights are downloaded on first use ([where they go](docs/weights.md)), and the first run
on a machine takes a few extra minutes to prepare the GPU. kiyo-ltx sends small mosaics to kiyo and the larger ones to
Lightricks' LTX-2.5 video model, fine-tuned to restore mosaics; its memory modes, system memory needs and licence are
in [docs/kiyo-ltx.md](docs/kiyo-ltx.md).

## Usage

```
kukkiri restore input.mp4 -o restored.mp4
kukkiri restore input.mp4 --model kiyo-ltx-v1
kukkiri restore movies/ extra.mkv -o restored/                       # several files and folders, one after another
kukkiri restore input.mp4 -o section.mp4 --start 37:45 --duration 30   # only part of a video
```

Without `-o` the result goes next to the input as `<name>.kukkiri-<model>.mp4`. Before restoring, kukkiri samples the
file and skips it when there is nothing to restore (already uncensored or already processed). In a batch, a file that
is declined or fails is reported and the rest go on; earlier kukkiri outputs in those folders are left out. Inputs the
GPU decoder cannot read directly (AVI, WMV, MPEG-TS, ...) are converted to MP4 first.

The output can be played while it is written (`<name>.part`, renamed when done; in the app, **Watch** opens it in VLC,
except with kiyo-ltx on GPUs under 15 GB, which write the video only in their last pass). On an NVIDIA GPU, a video of 15
minutes or more is restored in parts, kept in a `<output>.kukkiri-job` folder until it is done: if the restore is
interrupted, run the same command again and it carries on from the last finished part. The folder needs up to about
twice the output's size on the output's disk.

| Option | |
|---|---|
| `--model` | `kiyo-v4` (default), `kiyo-ltx-v1` or `kiyo-v3` |
| `--sensitivity high` / `low` | `high` also finds faint or small mosaics (and may touch things that are not mosaics), `low` makes fewer false finds |
| `--max-1080p` | restore sources above 1080p (4K) at 1080p: faster, smaller files |
| `--keep-fps` | kiyo-ltx: keep sources above 30 fps at their full rate (about twice as slow; by default 59.94 becomes 29.97) |
| `--full-frame` | kiyo-ltx: redraw the whole picture, for mosaics that are not found otherwise (see Limitations) |
| `--vr sbs` | side-by-side 180° VR video ([details](docs/vr.md)) |
| `--codec hevc` | HEVC output: about a third smaller at the same quality (`h264`, the default, plays everywhere) |
| `--gpu 1` | restore on another NVIDIA GPU (numbered as `nvidia-smi` lists them) |
| `--gpus 0,1` / `all` | split each video across several NVIDIA GPUs |
| `--restart` | start an interrupted restore over instead of resuming it |
| `--previews` | also write up to three watermarked before/after clips |
| `--skip-check` | restore even if the input looks uncensored |
| `kukkiri check` | test the GPU, driver, disk, model files and ffmpeg |

## Add mosaics

```
kukkiri add-mosaic input.mp4
```

Adds a block mosaic (blocks of about a 36th of the frame height, on a grid fixed to the frame) where a censored
release would have one, written next to the input as `<name>.kukkiri-mosaic.mp4` (or to `-o`). `--block` sets the
block size in pixels; `--sensitivity high` misses fewer. It needs only the kiyo-v4 weights. It is best effort: a frame
the detector misses for longer than a few frames stays uncensored, so check the result before relying on it.

## Limitations

- Pixelated faces are recognised and left as they are; kukkiri never restores them.
- Content another tool has already "decensored" into smooth shapes cannot be recovered; the input check declines it.
- The restored detail is generated, not recovered.
- `--full-frame` is slow and needs a high-VRAM GPU, so leave it off for ordinary censored video. It also redraws faces
  and areas without a mosaic: do not use it to try to identify real people. A 5-second clip takes about 40 seconds on
  a high-end GPU.

How kukkiri works: [docs/how-it-works.md](docs/how-it-works.md).

## Help

Questions and results: [Discussions](https://github.com/treehog2/kukkiri/discussions). Bugs:
[Issues](https://github.com/treehog2/kukkiri/issues); please include your GPU, the model, and the log (the app's
**Save log**, or the command's output).

## License

Copyright (c) 2026 kukkiri authors. The code is licensed under the GNU General Public License v3.0 only
(`LICENSE`); the vendored BasicVSR++ code keeps its Apache-2.0 license (`licenses/APACHE-2.0.txt`). Third-party
components and their licenses are listed in `NOTICE.md`. The model weights are released under the same license.
`LICENSE-EXCEPTION.md` grants an additional permission to combine kukkiri with NVIDIA's CUDA, TensorRT and Video
Codec SDK libraries. The kiyo-ltx weights are not GPL: they are a derivative of Lightricks' LTX-2.x, under the LTX-2.x
Community License ([details](docs/kiyo-ltx.md#licence)). The kukkiri name and logo are not covered by the license;
see `CONTRIBUTING.md`.

## Acknowledgements

- [Lada](https://github.com/ladaapp/lada), for showing what open mosaic restoration can do
- [BasicVSR++](https://arxiv.org/abs/2104.13371) (Chan et al., CVPR 2022), via OpenMMLab's
  [MMagic](https://github.com/open-mmlab/mmagic): the restoration network
- Roboflow's [RF-DETR](https://github.com/roboflow/rf-detr): the detector
- Lightricks' [LTX-2](https://github.com/Lightricks/LTX-2): the video diffusion model behind kiyo-ltx

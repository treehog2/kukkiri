# kiyo-ltx

`kiyo-ltx-v1` restores with Lightricks' [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) video diffusion model,
fine-tuned to restore mosaics, for clearer detail on larger mosaics; small ones still go to kiyo.

```
kukkiri restore input.mp4 --model kiyo-ltx-v1
```

## What it needs

- A GPU with at least **8 GB** of memory. Below 32 GB it keeps part of the model in system memory and streams it to
  the GPU as needed: the same result, but slower (about 1.5x on a 24 GB card, 1.75x on 16 GB, in our tests). Below
  15 GB it also runs in three passes (find the mosaics, restore the large ones with LTX, then the small ones and the
  video), so the models are never on the GPU together; it needs about 8 GB free, so close other GPU programs on an
  8 GB card. It stops with a message on less than 7.5 GB.
- System memory (RAM): **32 GB recommended** on Windows with a GPU under 32 GB. With less than 24 GB of RAM there it
  runs in the three passes too, which worked with 16 GB of RAM but left nothing free: close other programs first; it
  may still run out.
- Cards older than the RTX 30 series (RTX 20, GTX 16) work too, but slowly: they lack bf16 support, so kiyo-ltx
  computes some steps in smaller pieces (about 90 s per 3 s of video on a T4 in our tests).
- A **15 GB** download on first use (the folder `ltx-v1/` on Hugging Face).
- A few minutes on the first run on a machine, to build its GPU kernels (cached in the weights folder afterwards).

After that, a 10-second 1080p clip takes about 45 seconds on a high-end GPU, including loading the model.

Videos above 30 fps are restored at 30 fps by default (59.94 becomes 29.97), which halves the work; `--keep-fps` keeps
the full frame rate at about twice the time. VR with kiyo-ltx needs a GPU with 32 GB ([VR](vr.md)).

## Install on Linux

The Windows app, `./install.sh` and the Docker image include everything it needs. By hand, install the files in
[packaging/linux](../packaging/linux) in order: `requirements-torch.txt`, `requirements-base.txt`, then
`requirements-nodeps.txt` with `--no-deps` (the LTX packages' own pins are older than the versions kiyo-ltx uses).

## Full frame

`--full-frame` (kiyo-ltx only, not with `--vr`) redraws the whole picture instead of finding the mosaics, for mosaics
that are not found otherwise (on objects, for example). It is slow and needs a high-VRAM GPU, so leave it off for
ordinary censored video. It also redraws faces and areas without a mosaic. The result is generated, not recovered: do
not use it to try to identify real people. A 5-second clip takes about 40 seconds on a high-end GPU.

## Licence

The kiyo-ltx weights are a derivative of LTX-2.x and are licensed under the **LTX-2.x Community License**, not the GPL
(see `ltx-v1/LICENSE-LTX-2.x.txt` and `ltx-v1/NOTICE.md` on Hugging Face). That licence includes use restrictions, and
it asks that content made with the model be disclosed as machine-generated wherever it is published.

# VR

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

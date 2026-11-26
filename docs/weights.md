# Weights

The kiyo weights are licensed under the same terms as the code (GPL-3.0-only with the NVIDIA exception in
`LICENSE-EXCEPTION.md`); the kiyo-ltx weights are under the LTX-2.x Community License ([kiyo-ltx](kiyo-ltx.md#licence)).
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

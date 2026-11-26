# How it works

kukkiri runs as one streaming pass on the GPU: decoding, detection, restoration and encoding overlap, and only a
bounded window of frames is ever held in memory, so a two-hour film needs no more memory than a two-minute clip.

1. **Detect.** An RF-DETR segmentation model marks mosaic regions in every frame.
2. **Track.** Overlapping detections are merged into regions and followed through the scene; short misses are
   bridged by interpolation.
3. **Restore.** Each region is cut out as a square window with a little context, scaled onto the model's canvas and
   restored in clips of 96 frames by BasicVSR++. Neighbouring clips overlap and are crossfaded, so there are no seams.
   With kiyo-ltx, clips of the larger regions go to the fine-tuned LTX-2.5 model instead ([kiyo-ltx](kiyo-ltx.md)).
4. **Blend.** The restored region is feathered back into the frame just past the detected area, and the video is
   encoded on the GPU.

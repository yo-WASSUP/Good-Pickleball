# Model Weights

The copied baseline weights are kept locally for Good-Pickleball testing.

Default files used by the project:

```text
weights/tennis-ball.pt      # copied Good-Tennis ball detector baseline
weights/yolo26s.pt          # YOLO person detector
weights/yolo11s-pose.pt     # YOLO pose model
weights/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx
weights/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.onnx
```

Replace `weights/tennis-ball.pt` with a fine-tuned pickleball detector when one is trained, or pass another model with `--ball-model`.

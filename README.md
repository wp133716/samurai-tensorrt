# samurai_tensorrt

This repository contains the TensorRT implementation of the [SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory](https://github.com/yangchris11/samurai). The codebase includes both Python and C++ implementations for running inference using TensorRT engines.

## 使用tensorRT python推理

requirements:
- tensorRT 10+
- pycuda = 2025.1
- onnx = 1.17.0

#### 安装 tensorRT

download [tensorrt 10.1.0](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.1.0/tars/TensorRT-10.1.0.27.Linux.x86_64-gnu.cuda-11.8.tar.gz), 解压后cd 到TensorRT-10.1.0/python目录下，执行
```shell
pip install tensorrt-10.1.0-cp310-none-linux_x86_64.whl
pip install tensorrt_dispatch-10.1.0-cp310-none-linux_x86_64.whl
pip install tensorrt_lean-10.1.0-cp310-none-linux_x86_64.whl
```

#### SAM 2.1 Checkpoint Download
https://github.com/facebookresearch/sam2/tree/main/checkpoints

#### export onnx
onnx模型可以在另一个项目 [samurai-onnx](https://github.com/wp133716/samurai-onnx) 中获取

#### 运行
```shell
cd python
python main.py --video_path <path_to_video> --trt_engine_path <path_to_trt_engines>

or
python main.py --image_path <path_to_image> --onnx_model_path <path_to_onnx_models> # 将重新构建trt engine
```

## tensorRT C++推理版本coming soon

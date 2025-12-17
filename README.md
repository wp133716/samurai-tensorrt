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

## tensorRT C++推理

- tensorRT 10+
- opencv 4.10+

#### 安装 tensorRT
download [tensorrt 10.1.0](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.1.0/tars/TensorRT-10.1.0.27.Linux.x86_64-gnu.cuda-11.8.tar.gz), 解压后配置环境变量

```shell
在.bashrc中添加
# tensorRT
export TENSORRT_DIR=/home/user/tensorRT/TensorRT-10.***.Linux.x86_64-gnu.cuda-12.5/TensorRT-10.***
# export TENSORRT_DIR=/home/user/3rd-party/tensorRT/TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8/TensorRT-10.9.0.34
# export TENSORRT_DIR=/home/user/3rd-party/tensorRT/TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-12.9/TensorRT-10.14.1.48
export PATH=$PATH:$TENSORRT_DIR/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TENSORRT_DIR/lib
export LIBRARY_PATH=$LIBRARY_PATH:$TENSORRT_DIR/lib
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$TENSORRT_DIR/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$TENSORRT_DIR/include
```

将4个onnx模型放在 samurai_tensorrt/onnx_model 文件夹下
#### 运行
```shell
cd cpp && mkdir build
cd build
cmake .. && make -j8
./sam2_tracker
```
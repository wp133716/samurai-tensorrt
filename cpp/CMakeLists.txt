cmake_minimum_required(VERSION 3.16)
project(sam2_tracker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC -O0 -ggdb -Wno-deprecated-declarations")

set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
# find_package(TensorRT REQUIRED)
# find_package(OpenMP REQUIRED)

# add_library(tensorrt_cpp_api SHARED
#             src/engine.cpp)

add_executable(${PROJECT_NAME}
            main.cpp
            src/engine.cpp
            src/sam2_tracker.cpp
            src/kalman_filter.cpp
            )
# message("-- TensorRT include dirs: ${TensorRT_INCLUDE_DIRS}")
# message("-- TensorRT libraries: ${TensorRT_LIBRARIES}")

target_include_directories(${PROJECT_NAME} PRIVATE
                            include
                            ${OpenCV_INCLUDE_DIRS}
                            # ${CUDA_INCLUDE_DIRS}
                            )

# target_link_directories(${PROJECT_NAME} PRIVATE
#                         /home/user/3rd-party/tensorRT/TensorRT-10.2.0.19.Linux.x86_64-gnu.cuda-12.5/TensorRT-10.2.0.19/lib)

target_link_libraries(${PROJECT_NAME} PRIVATE
                        ${OpenCV_LIBS}
                        nvinfer
                        nvonnxparser
                        onnxruntime
                        ${CUDA_LIBRARIES}
                        # OpenMP::OpenMP_CXX
                        )

# -- CUDA include dirs: /usr/local/cuda/include
# -- CUDA libraries: /usr/local/cuda/lib64/libcudart_static.a;Threads::Threads;dl;/usr/lib/x86_64-linux-gnu/librt.a
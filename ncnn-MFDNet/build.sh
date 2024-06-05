export NDK_ROOT=/opt/sdk/ndk/25.2.9519653
export CMAKE=/home/jl/cmake-3.10.2/cmake-3.10.2-Linux-x86_64/bin

echo "NDK_ROOT is ${NDK_ROOT}"

# configure
ARM_ABI=arm64-v8a
ARM_TARGET_LANG=clang
NCNN_LITE_DIR="/media/Work/jl/Package/ncnn_tools/ncnn-20230816-android-vulkan-shared/"
OPENCV_LITE_DIR="/media/Work/jl/Package/opencv_tools/android_opencv4.1.0"
PROJECT_NAME=ecbsr_mfdnet

if [ "x$1" != "x" ]; then
    ARM_ABI=$1
fi

echo "ARM_TARGET_LANG is ${ARM_TARGET_LANG}"
echo "ARM_ABI is ${ARM_ABI}"
echo "NCNN_LITE_DIR is ${NCNN_LITE_DIR}"
echo "OPENCV_LITE_DIR is ${OPENCV_LITE_DIR}"
# build
if [ -d "$(pwd)/build" ]; then
  rm -rf build
fi
mkdir build
#make clean
cd build
cmake -DNCNN_LITE_DIR=${NCNN_LITE_DIR} -DARM_ABI=${ARM_ABI} -DARM_TARGET_LANG=${ARM_TARGET_LANG} -DOPENCV_LITE_DIR=${OPENCV_LITE_DIR} -DNDK_ROOT=${NDK_ROOT} -DPROJECT_NAME=${PROJECT_NAME} ..
make -j4
cd ..

echo "make successful!"

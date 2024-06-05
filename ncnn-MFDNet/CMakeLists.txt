cmake_minimum_required(VERSION 3.10)
set(CMAKE_SYSTEM_NAME linux)

if(ARM_TARGET_LANG STREQUAL "gcc")
    set(ANDROID_ARM_NEON True)
    set(CMAKE_NDK_ROOT_TOOLCHAIN_VERSION ${ARM_TARGET_LANG})
    set(ANDROID_TOOLCHAIN gcc)
    set(CMAKE_TOOLCHAIN_FILE ${NDK_ROOT}/build/cmake/android.toolchain.cmake)
    set(CMAKE_C_FLAGS gcc)
    set(CMAKE_CXX_FLAGS g++)
endif()

if(ARM_TARGET_LANG STREQUAL "clang")
    set(ANDROID_ARM_NEON True)
    set(CMAKE_NDK_ROOT_TOOLCHAIN_VERSION ${ARM_TARGET_LANG})
    set(ANDROID_TOOLCHAIN clang)
    if (${ARM_ABI} STREQUAL "arm64-v8a")
        set(CMAKE_C_COMPILER ${NDK_ROOT}/toolchains/llvm/prebuilt/${CMAKE_SYSTEM_NAME}-x86_64/bin/aarch64-linux-android30-clang)
        set(CMAKE_CXX_COMPILER ${NDK_ROOT}/toolchains/llvm/prebuilt/${CMAKE_SYSTEM_NAME}-x86_64/bin/aarch64-linux-android30-clang++)
    else()
        set(CMAKE_TOOLCHAIN_FILE ${NDK_ROOT}/build/cmake/android.toolchain.cmake)
    endif()
endif()

project(${PROJECT_NAME})
message(STATUS "TARGET ARM LANG: ${ARM_TARGET_LANG}")
message(STATUS "TARGET ARCH ABI: ${ARM_ABI}")
message(STATUS "NCNN LITE DIR: ${NCNN_LITE_DIR}")
message(STATUS "OPENCV LITE DIR: ${OPENCV_LITE_DIR}")
message(STATUS "C_COMPILER: ${CMAKE_C_COMPILER}")
message(STATUS "CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

set(Ncnn_LIBS ${NCNN_LITE_DIR}/${ARM_ABI}/ncnn-rtti-open/libglslang-default-resource-limits.a
              ${NCNN_LITE_DIR}/${ARM_ABI}/ncnn-rtti-open/libglslang.a
              ${NCNN_LITE_DIR}/${ARM_ABI}/ncnn-rtti-open/libncnn.a
              ${NCNN_LITE_DIR}/${ARM_ABI}/ncnn-rtti-open/libMachineIndependent.a
              ${NCNN_LITE_DIR}/${ARM_ABI}/ncnn-rtti-open/libGenericCodeGen.a
              ${NCNN_LITE_DIR}/${ARM_ABI}/ncnn-rtti-open/libOGLCompiler.a
              ${NCNN_LITE_DIR}/${ARM_ABI}/ncnn-rtti-open/libOSDependent.a
              ${NCNN_LITE_DIR}/${ARM_ABI}/ncnn-rtti-open/libSPIRV.a)
    
include_directories(${NCNN_LITE_DIR}/${ARM_ABI}/include/ncnn/)
message(STATUS "Ncnn_LIBS is ${Ncnn_LIBS}")

set(OPENCV_PATH "${OPENCV_LITE_DIR}/${ARM_ABI}")
message(STATUS "OPENCV_PATH: ${OPENCV_PATH}")
set(OPENCV_INCLUDE_DIRS "${OPENCV_PATH}/include/")
set(OPENCV_LIBS ${OPENCV_PATH}/libs/libopencv_imgcodecs.a
                ${OPENCV_PATH}/libs/libopencv_imgproc.a
                ${OPENCV_PATH}/libs/libopencv_core.a
                ${OPENCV_PATH}/3rdparty/libs/libtegra_hal.a
                ${OPENCV_PATH}/3rdparty/libs/liblibjpeg-turbo.a
                ${OPENCV_PATH}/3rdparty/libs/liblibwebp.a
                ${OPENCV_PATH}/3rdparty/libs/liblibpng.a
                ${OPENCV_PATH}/3rdparty/libs/liblibjasper.a
                ${OPENCV_PATH}/3rdparty/libs/liblibtiff.a
                ${OPENCV_PATH}/3rdparty/libs/libIlmImf.a
                ${OPENCV_PATH}/3rdparty/libs/libtbb.a
                ${OPENCV_PATH}/3rdparty/libs/libcpufeatures.a)
include_directories(${OPENCV_INCLUDE_DIRS})


set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -ffast-math -Ofast -Os -DNDEBUG -fno-exceptions -fomit-frame-pointer -fno-asynchronous-unwind-tables -fno-unwind-tables"
)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fexceptions -fvisibility=hidden -fvisibility-inlines-hidden -fdata-sections -ffunction-sections"
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp -static-openmp")

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cc)
target_link_libraries(${PROJECT_NAME} ${Ncnn_LIBS} ${OPENCV_LIBS} android -lz -llog)
target_link_options(${PROJECT_NAME} PRIVATE "-Wl,-rpath,/data/data/com.termux/files/home/jl/denoise-mfdnet")
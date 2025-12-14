cmake_minimum_required(VERSION 3.21) # 3.21 才能支持 TARGET_RUNTIME_DLLS

project(cv_code_2025 LANGUAGES C CXX)

# ==============================================================================
# 1. 全局配置 & 选项
# ==============================================================================
option(ENABLE_AVX2 "Enable AVX2 optimizations" ON)
option(ENABLE_CUDA "Enable CUDA support if available" ON)
option(BUILD_OPENCV "Build OpenCV from source if not found (Time consuming!)" ON)

# 必须要求 C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(MANUAL_OpenCV_DIR, "D:/cppPackages/opencv/build")
# set(MANUAL_OpenCV_BIN "D:/cppPackages/opencv/build/x64/vc17/bin")
include(FetchContent)

# Policy for FetchContent timestamps
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# --- 2.2 CUDA (Optional) ---
if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
            message(STATUS "Detected Clang. Configuring NVCC to use Clang host and libc++.")

            # [步骤 A] 设置 NVCC 的 Host Compiler
            # 这必须在 enable_language(CUDA) 之前设置
            set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")

            # 定义我们要使用的标志
            set(CLANG_STDLIB_FLAG "-stdlib=libc++")

            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${CLANG_STDLIB_FLAG}")
        endif()

        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        set(CUDA_FOUND TRUE)
        message(STATUS "CUDA detected. Enabling GPU acceleration.")
    else()
        message(WARNING "CUDA compiler not found. Falling back to CPU-only build.")
        set(CUDA_FOUND FALSE)
    endif()
endif()

set(OpenCV_ROOT "${VCPKG_INSTALLED_DIR}/x64-linux/share/opencv4")

# --- 2.3 OpenCV (Robust Find) ---
# 尝试 1: 标准查找 (利用 CMAKE_PREFIX_PATH 或 环境变量)
find_package(OpenCV QUIET)

# 尝试 2: 如果环境变量中设置了 OpenCV_DIR 但标准查找失败，尝试显式提示
if(NOT OpenCV_FOUND AND DEFINED ENV{OpenCV_DIR})
    message(STATUS "OpenCV not found in system path, trying ENV{OpenCV_DIR}...")
    set(OpenCV_DIR $ENV{OpenCV_DIR})
    find_package(OpenCV QUIET)

elseif(NOT OpenCV_FOUND AND DEFINED MANUAL_OpenCV_DIR)
    message(STATUS "OpenCV not found in system path, trying MANUAL_OpenCV_DIR...")
    set(OpenCV_DIR ${MANUAL_OpenCV_DIR})
    find_package(OpenCV QUIET)
endif()

# 尝试 3: 从源码构建 (如果开启了选项)
if(NOT OpenCV_FOUND)
    if(BUILD_OPENCV)
        message(STATUS "OpenCV not found. Fetching and building from source (This may take a while)...")
        FetchContent_Declare(
            opencv
            GIT_REPOSITORY https://github.com/opencv/opencv.git
            GIT_TAG 4.10.0
            GIT_SHALLOW TRUE
        )
        set(BUILD_LIST core highgui imgproc features2d calib3d videoio) # 仅构建核心模块以加速
        set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
        set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(opencv)
        set(OpenCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_features2d opencv_calib3d)
    else()
        message(FATAL_ERROR "OpenCV not found! Please set OpenCV_DIR environment variable or enable BUILD_OPENCV option.")
    endif()
else()
    message(STATUS "Found OpenCV: ${OpenCV_VERSION} at ${OpenCV_DIR}")
endif()

# ==============================================================================
# 3. Common Compilation Configuration (Interface Library)
# ==============================================================================
add_library(common_settings INTERFACE)

# 3.1 基础定义
target_compile_definitions(common_settings INTERFACE
    PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src"
    IMAGE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/images"
)

# 3.2 编译器选项 (UTF-8, AVX2)
target_compile_options(common_settings INTERFACE

    # --- 仅针对 C++ 源文件 (.cpp) ---
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<BOOL:${ENABLE_AVX2}>>:/arch:AVX2>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<BOOL:${ENABLE_AVX2}>>:-mavx2>

    # --- 仅针对 CUDA 源文件 (.cu) ---
    # 必须用 -Xcompiler 将 MSVC 的 flag 包裹起来传给 cl.exe
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<BOOL:${ENABLE_AVX2}>>:-Xcompiler=/arch:AVX2>
)

if(ENABLE_AVX2)
    target_compile_definitions(common_settings INTERFACE ENABLE_AVX2=1)
endif()

# 3.3 CUDA 相关配置
if(CUDA_FOUND)
    target_compile_definitions(common_settings INTERFACE HAS_CUDA=1)
    target_compile_options(common_settings INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-allow-unsupported-compiler>

        # $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:--compiler-options=/utf-8>
    )
endif()

# ==============================================================================
# 4. Utility Functions
# ==============================================================================

# 函数：自动处理 Windows DLL 复制
function(copy_dlls_post_build target_name)
    if(WIN32 and CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${target_name}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_RUNTIME_DLLS:${target_name}>"
            "$<TARGET_FILE_DIR:${target_name}>"
            COMMAND_EXPAND_LISTS
            COMMENT "Copying runtime DLLs for ${target_name}..."
        )
    endif()
endfunction()

# 函数：为目标添加 CUDA 支持
# 用法：target_add_cuda(<target_name> SOURCES <src1.cu> ... [ARCHS <80;86...>])
# 如果不指定 ARCHS，默认使用 "native" (自动检测本机显卡)
function(target_add_cuda target_name)
    # 0. 如果没有找到 CUDA，直接返回
    if(NOT CUDA_FOUND)
        return()
    endif()

    # 1. 解析参数
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SOURCES ARCHS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # 2. 添加源文件
    if(ARG_SOURCES)
        target_sources(${target_name} PRIVATE ${ARG_SOURCES})
    endif()

    # 3. 设置 CUDA 架构
    if(ARG_ARCHS)
        # 如果用户显式指定了架构 (例如发布版本: 75;80;86)，则使用用户指定的
        set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES ${ARG_ARCHS})
    else()
        # 【关键修改】如果用户没指定，使用 "native"
        # 这会自动检测当前机器的 GPU 架构 (例如 RTX 3060 -> sm_86)
        set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES native)
    endif()

    # CUDA 语言标准
    set_property(TARGET ${target_name} PROPERTY CUDA_STANDARD 20)
    set_property(TARGET ${target_name} PROPERTY CUDA_STANDARD_REQUIRED ON)

    # 4. 开启分离编译 (RDC)
    set_property(TARGET ${target_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

    # 5. 链接 CUDA 运行时
    target_link_libraries(${target_name} PRIVATE CUDA::cudart)

    # 6. 添加宏定义
    target_compile_definitions(${target_name} PRIVATE HAS_CUDA=1)
endfunction()

# ==============================================================================
# 5. Targets
# ==============================================================================

# --- Target 1: Simple Execute ---
add_executable(test_opencv "src/test_opencv.cpp")
target_link_libraries(test_opencv PRIVATE ${OpenCV_LIBS} common_settings)

# --- Target 2: Final Project (DFF Pipeline) ---
add_executable(final_project "src/final_project.cpp")
target_link_libraries(final_project PRIVATE ${OpenCV_LIBS} common_settings)

# Find yaml-cpp
find_package(yaml-cpp CONFIG)

if(yaml-cpp_FOUND)
    target_link_libraries(final_project PRIVATE yaml-cpp::yaml-cpp)
    message(STATUS "Found yaml-cpp: ${yaml-cpp_VERSION}")
else()
    message(WARNING "yaml-cpp not found. Please install it via vcpkg or system package manager.")
endif()

# copy_dlls_post_build(test_opencv)
add_executable(test_lib "src/compiler_check.cpp")

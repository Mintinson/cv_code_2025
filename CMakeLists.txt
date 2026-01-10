cmake_minimum_required(VERSION 3.21)

# 如果不想运行 vcpkg，可以通过取消下面这一行的注释来禁用 （一定要在 project 之前）
# unset(CMAKE_TOOLCHAIN_FILE CACHE)

project(cv_code_2025 LANGUAGES C CXX)

# ==============================================================================
# 1. 全局配置 & 选项
# ==============================================================================
option(ENABLE_AVX2 "Enable AVX2 optimizations" ON)
option(ENABLE_CUDA "Enable CUDA support if available" OFF)

set(MANUAL_OpenCV_DIR "/home/mintinson/opencv_gcc15/lib64/cmake/opencv4") # 手动指定 OpenCV 路径（可选）

# [需求 1] OpenCV 策略开关
option(PREFER_SYSTEM_OPENCV "Prioritize system/manual OpenCV over vcpkg" ON)

# [需求 2] yaml-cpp 策略开关
option(USE_INTERNAL_YAML_CPP "Use FetchContent for yaml-cpp (Source build) instead of vcpkg" OFF)

# C++ 标准设置
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# ==============================================================================
# 2. 依赖查找 (OpenCV & yaml-cpp)
# ==============================================================================

# --- 2.1 OpenCV Configuration ---
set(OPENCV_SEARCH_PATHS "")

# 如果用户开启了优先使用系统 OpenCV
if(PREFER_SYSTEM_OPENCV)
    # 1. 尝试环境变量
    if(DEFINED ENV{OpenCV_DIR})
        list(APPEND OPENCV_SEARCH_PATHS "$ENV{OpenCV_DIR}")
        message(STATUS "OpenCV: Checking ENV{OpenCV_DIR} -> $ENV{OpenCV_DIR}")
    endif()

    # 2. 尝试手动定义的变量 (可在 CMakePresets 或 命令行定义)
    if(DEFINED MANUAL_OpenCV_DIR)
        list(APPEND OPENCV_SEARCH_PATHS "${MANUAL_OpenCV_DIR}")
        message(STATUS "OpenCV: Checking MANUAL_OpenCV_DIR -> ${MANUAL_OpenCV_DIR}")
    endif()

    # 关键点：如果找到了路径，显式设置 OpenCV_DIR 给 CMake，这会覆盖 vcpkg 的默认行为
    if(OPENCV_SEARCH_PATHS)
        # 这里只是为了给 find_package 提供线索，具体的路径有效性由 find_package 验证
        set(OpenCV_DIR "${OPENCV_SEARCH_PATHS}" CACHE PATH "Path to OpenCV build" FORCE)
    endif()
endif()

# 查找 OpenCV (如果上面没设置 OpenCV_DIR，且 vcpkg 激活，这里会自动找 vcpkg 的)
find_package(OpenCV QUIET)

# 如果没找到，且允许 vcpkg (通常 vcpkg 已经通过 toolchain 介入了)，这里再次尝试报错或提示
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found! \n"
        "Tried System Paths: ${OPENCV_SEARCH_PATHS}\n"
        "Please set ENV{OpenCV_DIR} or check vcpkg installation.")
else()
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
    message(STATUS "   -> Location: ${OpenCV_DIR}")
endif()

# --- 2.2 yaml-cpp Configuration ---
if(USE_INTERNAL_YAML_CPP)
    message(STATUS "yaml-cpp: Using FetchContent (Internal Build)...")
    FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG master # 建议锁定具体的 commit hash 以保证稳定性
    )

    # 这一步会下载并 add_subdirectory
    FetchContent_MakeAvailable(yaml-cpp)

    # 关键：vcpkg 的 yaml-cpp 提供的是 yaml-cpp::yaml-cpp 目标
    # FetchContent 构建通常只提供 yaml-cpp。为了兼容性，我们需要创建一个别名。
    if(NOT TARGET yaml-cpp::yaml-cpp)
        add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
    endif()
else()
    message(STATUS "yaml-cpp: Using find_package (vcpkg/System)...")
    find_package(yaml-cpp CONFIG REQUIRED)
endif()

# ==============================================================================
# 3. CUDA Setup
# ==============================================================================
if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND DEFINED USE_LIBCXX AND USE_LIBCXX)
            message(STATUS "Detected Clang. Configuring NVCC to use Clang host and libc++.")
            set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
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

# ==============================================================================
# 4. Common Settings & Interface
# ==============================================================================
add_library(common_settings INTERFACE)
target_compile_definitions(common_settings INTERFACE
    PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src"
    IMAGE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/images"
)
target_compile_options(common_settings INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<BOOL:${ENABLE_AVX2}>>:/arch:AVX2>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<BOOL:${ENABLE_AVX2}>>:-mavx2>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Clang>>:-fexperimental-library>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<BOOL:${ENABLE_AVX2}>>:-Xcompiler=/arch:AVX2>
)

if(ENABLE_AVX2)
    target_compile_definitions(common_settings INTERFACE ENABLE_AVX2=1)
endif()

if(CUDA_FOUND)
    target_compile_definitions(common_settings INTERFACE HAS_CUDA=1)
    target_compile_options(common_settings INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-allow-unsupported-compiler>
    )
endif()

# ==============================================================================
# 5. Utility Functions: Smart DLL Copy
# ==============================================================================

set(COPY_SCRIPT_FILE "${CMAKE_BINARY_DIR}/cmake/CopyDllsIfMissing.cmake")
file(WRITE "${COPY_SCRIPT_FILE}" "
# 这是一个构建时运行的脚本
# 输入参数: DLL_LIST (分号分隔), TARGET_DIR (目标目录)

foreach(dll_path IN LISTS DLL_LIST)
    # 获取文件名 (例如 opencv_core.dll)
    get_filename_component(dll_name \"\${dll_path}\" NAME)
    
    # 1. 在系统 PATH 中查找该文件
    # find_program 默认会在 ENV{PATH} 中查找
    # 我们每次都要清理缓存变量，确保对每个文件都重新查找
    unset(DLL_IN_SYSTEM_PATH CACHE)
    find_program(DLL_IN_SYSTEM_PATH NAMES \"\${dll_name}\")
    
    if(DLL_IN_SYSTEM_PATH)
        message(STATUS \"[SKIP] \${dll_name} found in system path: \${DLL_IN_SYSTEM_PATH}\")
    else()
        # 2. 如果没找到，执行复制
        # 只有当目标文件不存在或比源文件旧时才复制 (file(COPY) 默认行为覆盖，但开销很小)
        message(STATUS \"[COPY] \${dll_name} NOT found in system path. Copying to output...\")
        file(COPY \"\${dll_path}\" DESTINATION \"\${TARGET_DIR}\")
    endif()
endforeach()
")

function(copy_dlls_smart target_name)
    if(NOT DEFINED CMAKE_TOOLCHAIN_FILE) # 仅当未使用 vcpkg 时启用
        # 使用生成表达式获取该目标依赖的所有运行时 DLL
        # 注意: COMMAND 中调用 cmake -P 执行我们上面生成的脚本
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -D "DLL_LIST=$<TARGET_RUNTIME_DLLS:${target_name}>"
            -D "TARGET_DIR=$<TARGET_FILE_DIR:${target_name}>"
            -P "${COPY_SCRIPT_FILE}"
            COMMAND_EXPAND_LISTS
            COMMENT "Checking system PATH for DLLs and copying if missing..."
        )
    endif()
endfunction()

function(target_add_cuda target_name)
    if(NOT CUDA_FOUND)
        return()
    endif()

    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SOURCES ARCHS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(ARG_SOURCES)
        target_sources(${target_name} PRIVATE ${ARG_SOURCES})
    endif()

    if(ARG_ARCHS)
        set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES ${ARG_ARCHS})
    else()
        set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES native)
    endif()

    set_property(TARGET ${target_name} PROPERTY CUDA_STANDARD 20)
    set_property(TARGET ${target_name} PROPERTY CUDA_STANDARD_REQUIRED ON)
    set_property(TARGET ${target_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${target_name} PRIVATE CUDA::cudart)
    target_compile_definitions(${target_name} PRIVATE USE_CUDA=1)
endfunction()

# ==============================================================================
# 6. Targets
# ==============================================================================
add_executable(test_lib "src/compiler_check.cpp")

# --- Target 1 ---
add_executable(test_opencv "src/test_opencv.cpp")
target_link_libraries(test_opencv PRIVATE ${OpenCV_LIBS} common_settings)

# 应用智能 DLL 复制
copy_dlls_smart(test_opencv)

# --- Target 2 ---
add_executable(final_project "src/final_project.cpp")
target_link_libraries(final_project PRIVATE ${OpenCV_LIBS} common_settings yaml-cpp::yaml-cpp)

# 应用智能 DLL 复制
copy_dlls_smart(final_project)
target_add_cuda(final_project
    SOURCES "src/cuda_operators.cu"
)
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)
set(VCPKG_CMAKE_SYSTEM_NAME Linux)

# 1. 强制使用 Clang/LLVM 相关的编译器标志
set(VCPKG_CXX_FLAGS "-stdlib=libc++ -fPIC")
set(VCPKG_C_FLAGS "-fPIC")
set(VCPKG_LINKER_FLAGS "-stdlib=libc++ -lc++abi")

# 2. 如果 vcpkg 内部调用 CMake 构建包，确保它知道我们要用 libc++
if(VCPKG_CMAKE_SYSTEM_NAME STREQUAL "Linux")
    string(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS " -DCMAKE_CXX_FLAGS=-stdlib=libc++ -DCMAKE_EXE_LINKER_FLAGS=-stdlib=libc++ -DCMAKE_SHARED_LINKER_FLAGS=-stdlib=libc++")
endif()
# 使用 cmake + cmake preset + vcpkg 来管理环境和依赖

CMake 配合 Vcpkg 提供了强大的依赖管理和跨平台构建能力。

测试过的环境和工具链：
* Windows + MSVC + STL ✅
* Windows + MinGW + libstdc++ ✅
* Linux + GCC + libstdc++✅
* Linux + Clang + libstdc++✅

待测试的环境
* Windows + Clang-cl + STL
* Linux + Clang + libc++

## 1. 安装必要工具

### 安装 `CMake`

从 [CMake 官网](https://cmake.org/download/) 下载安装。推荐安装 >= 3.32 或者 4.x 版本。

```bash
# 测试是否安装成功
cmake --version
```

```
cmake version 3.xx.x

CMake suite maintained and supported by Kitware (kitware.com/cmake).
```

### 安装 `Ninja`

`Ninja` 是一个极简、极快的构建执行器，用于根据已生成的构建规则，高效地增量编译 C/C++ 项目。

从系统包管理或者[Ninja 官网](https://ninja-build.org/) 中下载安装。

```bash
ninja --version
```

```
1.xx.x
```

## 2. 设置 Vcpkg

Vcpkg 是由 Microsoft 和 C++ 社区维护的免费开源 C/C++ 包管理器，可在 Windows、macOS 和 Linux 上运行。 它是核心的 C++ 工具，使用 C++ 和 CMake 脚本编写。

克隆 Vcpkg:
```bash
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
```

引导 Vcpkg:
```bash
# Windows
.\bootstrap-vcpkg.bat

# Linux/macOS
./bootstrap-vcpkg.sh
```

**设置环境变量** (推荐):
```bash
# Windows (PowerShell)
$env:VCPKG_ROOT = "C:\path\to\vcpkg"  # 替换为实际路径

# Linux/macOS
export VCPKG_ROOT=/path/to/vcpkg     # 添加到 ~/.bashrc 或 ~/.zshrc
```

有问题可参考 [vcpkg 文档 | Microsoft Learn](https://learn.microsoft.com/zh-cn/vcpkg/)

## 3. 使用 CMake Presets 构建:

项目提供了预配置的 CMake Presets，可以直接使用：

```bash
# 回到项目目录
cd cd cv_code_2025

# Windows - x64 Release
cmake --preset x64-release

# Windows - MinGW 
cmake --preset mingw-release

# Linux - Release
cmake --preset linux-release

# Linux - Clang + libstdc++
cmake --preset clang-release

# Linux - Clang + libc++
cmake --preset  clang-libc++-release
# 指定 Vcpkg 工具链（如果没有设置 VCPKG_ROOT）
cmake --preset x64-release -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
```

⚠️: 使用 clang + libc++ 构建 vcpkg 下载的包并不是显然的，需要专门编写自定义的 triplets，目前还没有测试过。谨慎处理。


如果配置正确，执行上述命令后，CMake会自动利用 vcpkg 从 github 上拉取源码并编译。由于 OpenCV 是一个很大且依赖很多的库。编译时间较长，占用CPU和内存资源也较大，请耐心等待。

### 若系统已安装了 OpenCV

考虑到一些Linux 发行版系统库自带 OpenCV（比如 `Ubuntu`），以及一些用户在Windows下已经下载了 OpenCV 库，我们可以跳过OpenCV 漫长的下载和编译。直接链接。

#### 如果 OpenCV 库已经在系统环境变量上

1. 删除 [vcpkg.json](../vcpkg.json) 中 `opencv4` 这一行即可。

注意：**在环境变量**中指的是类似(Windows)`opencv\build`( 用于 Cmake
 查找) 和 类似 `opencv\build\x64\vc16\bin` (用于程序动态链接) 都在 环境变量下。两者缺一不可。

可以通过以下命令行查看：

**Windows powershell**

```pwsh
Get-ChildItem Env: | Select-String "opencv"
```

**Linux**

```bash
env | grep -i opencv
```

#### 如果 OpenCV 库下载了但是不在环境变量

如果下载了 OpenCV 库，但是不想 *污染* 环境变量，可以：
1. 删除 vcpkg.json 中 opencv4 这一行。
2. 设置 [CmakeLists.txt](../CMakeLists.txt) 中 的 `MANUAL_OpenCV_DIR` 变量指定 `opencv/build` 目录
3. 设置 [CmakeLists.txt](../CMakeLists.txt) 中 的 `MANUAL_OpenCV_BIN` 变量指定 `bin` 目录

然后按照上述要求构建即可。

## 4. 编译目标

```bash
# 编译全部目标
cmake --build 

# 编译指定目标
cmake --build --target final_project
```

生成的目标以及 `vcpkg` 下载的第三方库会在 `${sourceDir}/out/build/${presetName}` 下。

## 5. 运行目标

```bash
# 变量换成你的 presetName
cd out/build/${presetName}

# 不带任何参数
./final_project

# 带参数
./final_project --config your_config.yaml
```

## 6. 其他问题

待补充
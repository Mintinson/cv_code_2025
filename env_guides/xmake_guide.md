# 使用 xmake 来管理环境和依赖

[Xmake](https://xmake.io/zh/) 是一个轻量级的跨平台构建工具，配置简单，依赖管理自动化。

测试过的环境和工具链：
* Windows + MSVC ✅
* Windows + MinGW ❌ （存在链接失败的问题）
* Linux + GCC ✅

待测试的环境
* Windows + Clang-cl 
* Linux + Clang
* Linux + Clang + libc++

若系统满足上述条件，可以接着使用xmake，其更加方便，配置文件更加直观快捷。

## 1. 安装 xmake

请参考 [Xmake 官网 | Xmake](https://xmake.io/zh/guide/quick-start.html) 安装最新的 xmake。

## 2. 配置项目

进入项目目录：
```bash
cd cv_code_2025
```

**开始配置**:
```bash
xmake f -m release
```

若之前没有安装 `opencv` 和 `yaml-cpp`，该命令会自动引导你下载并安装包。

**自定义配置**:
```bash
# 指定构建模式
xmake f -m debug          # 调试模式
xmake f -m release        # 发布模式（推荐）

# 更换编译器
xmake f --toolchain=auto # 默认工具链
xmake f --toolchain=gcc # 使用 gcc
```

## 3. 构建项目

```bash
# 构建所有目标
xmake

# 或者只构建主程序
xmake build final_project
```

默认生成的二进制目录在 `build/{system}` 下，其中 `system` 根据 linux还是 windows 有所不同。

在 [xmake.lua](../xmake.lua) 中查看可构建的目标。

## 4. 运行程序

```bash
xmake run final_project
```

命令行参数也可以直接跟着目标名称输入：

```bash
xmake run final_project --config your_config.yaml
```

或者直接到生成的二级制目录下运行：

```bash
./build/windows/x64/release/final_project.exe       # Windows
./build/linux/x86_64/release/final_project          # Linux
```

## 5. 其他问题

* 如遇网络问题，可以查看 [网络优化 | Xmake](https://xmake.io/zh/guide/package-management/network-optimization.html)
* 关于 xmake.lua 如何配置以及更多xmake 问题，可以查看官网 [简介 | Xmake](https://xmake.io/zh/guide/introduction.html)
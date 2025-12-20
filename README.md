# 深度从聚焦 (Depth from Focus) 项目

这是一个针对2025年计算机视觉课后大作业，项目三————深度从聚焦（DFF）流水线的实现。该项目基于 C++23，并支持 CUDA 加速。

## 功能特性

- 🚀 **高性能**: 支持 CPU 多线程并行处理和可选的 CUDA GPU 加速
- 🎯 **多种算法**: 支持 SML（修正拉普拉斯和）和 RDF（环形差分滤波器）聚焦度量
- 🔧 **灵活配置**: 支持 YAML 配置文件和命令行参数
- 📐 **图像对齐**: 支持 ORB 特征匹配和 ECC（增强相关系数）对齐
- 🎨 **深度细化**: 使用引导滤波器或联合双边滤波器进行边缘保持平滑
- 📊 **丰富输出**: 生成深度图、置信度图、全聚焦图像和物理深度图

## 环境要求

### 必需环境

- **C++23 编译器**:
  - Windows: MSVC 2022 (19.37+) 或更新版本
  - Linux: GCC 14+ 或 Clang 20+

- **依赖库**:
  - OpenCV 4.x (测试版本: 4.11.0)
  - yaml-cpp (测试版本: 0.8.0)

### 可选环境

- **CUDA 支持** (可选，用于 GPU 加速):
  - CUDA Toolkit 12.0+ (支持C++20语法)
  - CUDA 目前在 Windows 下只支持 MSVC 编译器；而在 Linux 下支持 GNU 和 Clang 编译器。

## 环境搭建

本项目提供两种构建方式，您可以根据需要选择其一。

### 方法一：使用 Xmake（推荐）

见 [xmake_guide](env_guides/xmake_guide.md)。


### 方法二：使用 CMake + Vcpkg

见 [cmake_guide](env_guides/cmake_guide.md)

## 使用说明

### 准备图像数据

1. 将待处理的图像序列放入 `images/project_imgs/` 目录
2. 图像应按焦距顺序命名，且序号应为 001,002 而不是 1, 2（程序会自动排序）
3. 支持的图像格式：`.jpg`, `.png`

**目录结构示例**:
```
cv_code_2025/
├── images/
│   └── project_imgs/
│       ├── image_001.bmp
│       ├── image_002.bmp
│       ├── image_003.bmp
│       └── ...
```

### 配置参数

#### 方式一：使用配置文件（推荐）

编辑 `src/configs/default_config.yaml` 文件，或创建自己的配置文件，注意新配置文件一定要在`src/configs/`目录下才能自动识别到：

```yaml
# 基础设置
useColor: false              # 使用彩色图像而非灰度图
useCuda: false               # 启用 CUDA 加速
useParallelExec: true        # 启用 CPU 多线程并行

# 图像对齐
useEcc: false                # 使用 ECC 对齐（否则使用 ORB）
orbNumFeatures: 1000         # ORB 特征点数量
eccResizeFactor: 0.5         # ECC 缩放因子（加速）
eccMaxCount: 50              # ECC 最大迭代次数
eccEpsilon: 0.0001           # ECC 收敛阈值
ransacThreshold: 5.0         # RANSAC 阈值

# 聚焦度量
useRDF: false                # 使用 RDF（否则使用 SML）
smlWindowSize: 5             # SML 窗口大小
rdfOuterR: 3                 # RDF 外环半径
rdfInnerR: 1                 # RDF 内环半径

# 深度细化
useGuidedFilter: true        # 使用引导滤波器（否则使用双边滤波器）
guidedFilterRadius: 8        # 引导滤波器半径
guidedFilterEps: 0.01        # 引导滤波器正则化参数
bilateralFilterD: 9          # 双边滤波器直径
bilateralSigmaColor: 10.0    # 双边滤波器颜色标准差
bilateralSigmaSpace: 5.0     # 双边滤波器空间标准差

# 全聚焦图像生成
useFusionRefine: true        # 使用加权融合（否则直接索引）
fusionGaussKSize: 7          # 融合时的高斯模糊核大小
refinedGaussKSize: 5         # 深度图预模糊核大小

# 其他选项
useInpaint: true             # 对低置信度区域进行修补
saveData: true               # 保存数据到文本文件

# 物理深度参数
physicalDepthStart: 1.0      # 起始距离（毫米）
physicalDepthStep: 2.0       # 步进距离（毫米）

# 输出
outputSubDir: unified        # 结果输出子目录
```

#### 方式二：使用命令行参数

命令行参数会覆盖配置文件中的设置。

**查看所有可用参数**:
```bash
./final_project --help
```

**常用命令行参数示例**:

```bash
# 使用自定义配置文件
./final_project --config my_config.yaml

# 启用 CUDA 加速
./final_project --useCuda true

# 启用彩色图像处理
./final_project --useColor true

# 使用 ECC 对齐
./final_project --useEcc true

# 使用 RDF 聚焦度量
./final_project --useRDF true

# 组合多个参数
./final_project \
  --config custom.yaml \
  --useCuda true \
  --useParallelExec true \
  --outputSubDir my_results
```

## 输出结果

程序运行完成后，结果将保存在 `results/project/<outputSubDir>/` 目录中。

### 输出文件说明

```
results/
└── project/
    └── unified/                          # 输出子目录（可通过参数配置）
        ├── config_used.yaml              # 本次运行使用的完整配置
        ├── depth_index_map.png           # 深度索引图（灰度）
        ├── physical_depth_map.png        # 物理深度图（伪彩色）
        ├── confidence_map.png            # 置信度图
        ├── all_in_focus.png              # 全聚焦图像
        ├── refined_depth_index_map.txt   # 细化后的深度索引（文本格式）
        ├── physical_depth_map.txt        # 物理深度值（文本格式）
        └── confidence_map.txt            # 置信度值（文本格式）
```

### 结果解释

1. **depth_index_map.png**: 深度索引图，表示每个像素最清晰对应的图像帧索引（灰度图）
2. **physical_depth_map.png**: 物理深度图，将深度索引转换为实际物理距离，使用伪彩色可视化
3. **confidence_map.png**: 置信度图，表示深度估计的可靠性（越亮越可靠）
4. **all_in_focus.png**: 全聚焦图像，整个场景都清晰的合成图像
5. **config_used.yaml**: 记录本次运行使用的所有参数配置
6. **\*.txt**: 文本格式的数值数据，可用于进一步分析或可视化

## 目录结构

```
cv_code_2025/
├── src/                          # 源代码目录
│   ├── final_project.cpp         # 主程序
│   ├── cuda_operators.cu         # CUDA 加速实现（可选）
│   ├── cuda_operators.hpp        # CUDA 接口头文件
│   ├── my_cmd_parser.hpp         # 命令行解析器
│   ├── my_logger.hpp             # 日志系统
│   └── configs/                  # 配置文件目录
│       └── default_config.yaml   # 默认配置
├── images/                       # 图像数据目录
│   └── project_imgs/             # 待处理图像（用户放置）
├── results/                      # 输出结果目录
│   └── project/                  # 项目结果
│       └── unified/              # 默认输出子目录
├── env+guides/                   # 环境构建指导
│   ├── cmake_guide.md/           # 使用 cmake 的环境构建指导                    
│   └── xmake_guide.md/           # 使用 xmake 的环境构建指导                    
├── build/                        # 构建输出（xmake）
├── out/                          # 构建输出（cmake）
├── xmake.lua                     # Xmake 构建配置
├── CMakeLists.txt                # CMake 构建配置
├── CMakePresets.json             # CMake 预设配置
├── vcpkg.json                    # Vcpkg 依赖配置
└── README.md                     # 本文档
```

## 算法流程

程序执行以下步骤处理图像序列：

1. **图像加载**: 从指定目录加载图像序列
2. **图像对齐**: 使用 ORB 特征匹配或 ECC 算法对齐图像
3. **聚焦度量计算**: 使用 SML 或 RDF 计算每张图像的聚焦度
4. **深度估计**: 通过抛物线插值估算亚像素级深度
5. **深度细化**: 使用引导滤波器或双边滤波器平滑深度图
6. **物理深度转换**: 将深度索引转换为实际物理距离
7. **结果保存**: 保存所有结果图像和数据

## 性能优化建议

### CPU 优化
- 启用 `--useParallelExec true` 使用多线程加速
- 调整 `--eccResizeFactor` 参数加快 ECC 对齐速度（降低精度）
- 减少 `--orbNumFeatures` 降低特征匹配计算量

### GPU 优化
- 如果有 NVIDIA GPU，编译时启用 CUDA 支持
- 运行时使用 `--useCuda true` 启用 GPU 加速
- GPU 加速主要影响聚焦度量计算步骤（SML/RDF）

### 质量与速度权衡
- **快速模式**: `useEcc=false`, `smlWindowSize=3`, `useParallelExec=true`
- **高质量模式**: `useEcc=true`, `smlWindowSize=7`, `useGuidedFilter=true`


## 开发说明

### 添加新的配置参数

1. 在 `final_project.cpp` 中的 `DFF_CONFIG_FIELDS` 宏中添加新字段：
   ```cpp
   X(type, name, default_value, "description")
   ```

2. 新参数会自动支持：
   - 命令行参数解析
   - YAML 配置文件读写
   - 参数覆盖机制

### 扩展功能

项目采用模块化设计，主要类包括：
- `DepthFromFocusPipeline`: 主流水线类
- `GuidedFilter`: 引导滤波器实现
- `JointBilateralFilter`: 联合双边滤波器实现
- `MyCmdParser`: 命令行解析器
- `Logger`: 日志系统

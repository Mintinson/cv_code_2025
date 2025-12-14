#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <execution>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <span>
#include <vector>

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 16

void checkCuda(cudaError_t result, const char* func)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error in " << func << ": " << cudaGetErrorString(result) << std::endl;
        throw std::runtime_error("CUDA Error");
    }
}

#define checkCudaErrors(val) cuda_check((val), #val, __FILE__, __LINE__)

template <typename T>
void cuda_check(T err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

// 辅助函数：处理边界 (Clamp/Replicate border)
// OpenCV 默认是 REFLECT，为了由纯 CUDA 实现简洁性，这里使用 Replicate (Clamp to edge)
// 如果严格需要 REFLECT，需要修改坐标映射逻辑 (e.g. abs(x) if x < 0 etc.)
__device__ int clamp_idx(int p, int max_val) { return max(0, min(p, max_val - 1)); }

__device__ __forceinline__ int reflect_idx(int p, int len)
{
    if (p < 0) p = -p - 1;
    if (p >= len) p = 2 * len - 1 - p;
    // 防止极其越界的情况
    return max(0, min(p, len - 1));
}

// -------------------------------------------------------------------------
// Kernel 1: 融合 (Uint8->Float) + (Normalize) + (MLAP X&Y) + (Abs Sum)
// -------------------------------------------------------------------------
template <int CHANNELS>
__global__ void fused_mlap_kernel(const unsigned char* __restrict__ src,
                                  float* __restrict__ dst,
                                  int width,
                                  int height,
                                  size_t src_step,
                                  size_t dst_step)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const float scale = 1.0f / 255.0f;
    float sum_lap     = 0.0f;

    // 预计算坐标，减少重复计算
    int x_l = reflect_idx(x - 1, width);
    int x_r = reflect_idx(x + 1, width);
    int y_u = reflect_idx(y - 1, height);
    int y_d = reflect_idx(y + 1, height);

// 遍历通道 (编译期循环展开)
#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        // 计算各点在 Global Memory 中的 Byte Offset
        // Row ptr
        const unsigned char* row_c = src + y * src_step;
        const unsigned char* row_u = src + y_u * src_step;
        const unsigned char* row_d = src + y_d * src_step;

        float val_c = row_c[x * CHANNELS + c] * scale;
        float val_l = row_c[x_l * CHANNELS + c] * scale;
        float val_r = row_c[x_r * CHANNELS + c] * scale;
        float val_u = row_u[x * CHANNELS + c] * scale;
        float val_d = row_d[x * CHANNELS + c] * scale;

        // MLAP Logic
        float lapX = fabsf(2.0f * val_c - val_l - val_r);
        float lapY = fabsf(2.0f * val_c - val_u - val_d);

        sum_lap += (lapX + lapY);
    }

    // 写入结果 (dst 始终是单通道 float)
    // 注意：dst 是 float 指针，索引不需要考虑字节步长(假设调用者传入的是紧凑或正确转换的指针)
    // 但通常 GPU memory pitch 是按字节算的，这里为了通用性，建议 dst 也传 pitch
    // 下面的实现假设 dst 是 pitched linear memory，需要手动计算 byte offset
    // 但为了配合 box filter，我们假设 dst 传入时已经处理好或使用 pitch
    // 此处调用方传入的是 pitch (size_t dst_step)，所以需要转成 float*

    // 修正：kernel 参数 dst_step 是 byte pitch
    float* row_out = (float*)((char*)dst + y * dst_step); // dst_step is in bytes
    row_out[x]     = sum_lap;
}

// -------------------------------------------------------------------------
// Kernel 2: Box Filter - Row Pass (水平方向求和)
// -------------------------------------------------------------------------
__global__ void box_row_kernel(const float* __restrict__ src,
                               float* __restrict__ dst,
                               int width,
                               int height,
                               size_t step_bytes, // float data 的 step 也是字节数
                               int ksize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int radius = ksize / 2;
    float sum  = 0.0f;

    // 简单的滑动窗口求和 (非优化版，优化版可使用前缀和或Shared Memory)
    size_t stride_float  = step_bytes / sizeof(float);
    const float* row_ptr = src + y * stride_float;

    for (int k = -radius; k <= radius; ++k) {
        int sample_x = reflect_idx(x + k, width);
        sum += row_ptr[sample_x];
    }

    dst[y * stride_float + x] = sum;
}

// -------------------------------------------------------------------------
// Kernel 3: Box Filter - Col Pass (垂直方向求和)
// -------------------------------------------------------------------------
__global__ void box_col_kernel(const float* __restrict__ src,
                               float* __restrict__ dst,
                               int width,
                               int height,
                               size_t step_bytes,
                               int ksize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int radius = ksize / 2;
    float sum  = 0.0f;

    size_t stride_float = step_bytes / sizeof(float);

    for (int k = -radius; k <= radius; ++k) {
        int sample_y = reflect_idx(y + k, height);
        sum += src[sample_y * stride_float + x];
    }

    dst[y * stride_float + x] = sum;
}

template <int CHANNELS>
void process_single_sml_cuda_impl(const cv::Mat& h_src,
                                  cv::Mat& h_dst,
                                  int kWindowSize,
                                  unsigned char* d_src,
                                  size_t src_pitch,
                                  float* d_temp1,
                                  size_t temp1_pitch,
                                  float* d_temp2,
                                  size_t temp2_pitch,
                                  cudaStream_t stream)
{
    int width  = h_src.cols;
    int height = h_src.rows;

    // 1. Upload
    checkCudaErrors(cudaMemcpy2DAsync(d_src,
                                      src_pitch,
                                      h_src.data,
                                      h_src.step,
                                      width * CHANNELS * sizeof(uint8_t),
                                      height,
                                      cudaMemcpyHostToDevice,
                                      stream));

    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // 2. Fused MLAP
    // temp1_pitch 是字节数，传递给 Kernel
    fused_mlap_kernel<CHANNELS>
        <<<grid, block, 0, stream>>>(d_src,
                                     d_temp1,
                                     width,
                                     height,
                                     src_pitch,
                                     temp1_pitch); // 注意：移除了 dst_pitch 参数，复用了内部计算

    // 3. Box Filter (Row)
    box_row_kernel<<<grid, block, 0, stream>>>(
        d_temp1, d_temp2, width, height, temp1_pitch, kWindowSize);

    // 4. Box Filter (Col)
    box_col_kernel<<<grid, block, 0, stream>>>(
        d_temp2, d_temp1, width, height, temp1_pitch, kWindowSize);

    // 5. Download
    checkCudaErrors(cudaMemcpy2DAsync(h_dst.data,
                                      h_dst.step,
                                      d_temp1,
                                      temp1_pitch,
                                      width * sizeof(float),
                                      height,
                                      cudaMemcpyDeviceToHost,
                                      stream));
}

std::vector<cv::Mat> sml_cuda(const std::span<const cv::Mat> alignedStack, int kSMLWindowSize)
{
    if (alignedStack.empty()) return {};

    int width       = alignedStack[0].cols;
    int height      = alignedStack[0].rows;
    int type        = alignedStack[0].type();
    int channels    = alignedStack[0].channels();
    size_t n_images = alignedStack.size();

    if (channels != 1 && channels != 3) {
        throw std::runtime_error("sml_cuda: Only supports 1 or 3 channel images.");
    }

    std::vector<cv::Mat> focusVolume(n_images);

    // 资源分配
    unsigned char* d_src = nullptr;
    float* d_temp1       = nullptr;
    float* d_temp2       = nullptr;
    size_t src_pitch, float_pitch;

    // 根据通道数分配足够的 pitch width
    checkCudaErrors(
        cudaMallocPitch(&d_src, &src_pitch, width * channels * sizeof(uint8_t), height));
    checkCudaErrors(cudaMallocPitch(&d_temp1, &float_pitch, width * sizeof(float), height));
    checkCudaErrors(cudaMallocPitch(&d_temp2, &float_pitch, width * sizeof(float), height));

    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    for (size_t i = 0; i < n_images; ++i) {
        focusVolume[i].create(height, width, CV_32F);

        if (channels == 1) {
            process_single_sml_cuda_impl<1>(alignedStack[i],
                                            focusVolume[i],
                                            kSMLWindowSize,
                                            d_src,
                                            src_pitch,
                                            d_temp1,
                                            float_pitch,
                                            d_temp2,
                                            float_pitch,
                                            stream);
        }
        else {
            process_single_sml_cuda_impl<3>(alignedStack[i],
                                            focusVolume[i],
                                            kSMLWindowSize,
                                            d_src,
                                            src_pitch,
                                            d_temp1,
                                            float_pitch,
                                            d_temp2,
                                            float_pitch,
                                            stream);
        }
    }

    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaStreamDestroy(stream));
    checkCudaErrors(cudaFree(d_src));
    checkCudaErrors(cudaFree(d_temp1));
    checkCudaErrors(cudaFree(d_temp2));

    return focusVolume;
}

constexpr std::size_t kMaxKernelSize = 25 * 25;
__constant__ float kKernel[kMaxKernelSize];

__global__ void rdf_convolution_tex_kernel(
    cudaTextureObject_t texObj, float* __restrict__ dst, int width, int height, int k_radius)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum {};
    int kIdx {};
    int kWidth = (2 * k_radius) + 1;

    // The convolution
    // using texture memory tex2D reads, which automatically handles boundary reflection
    // and type conversion (uchar->float).
    // the coordinate offset +0.5f is to align the texture center.
#pragma unroll
    for (int dy = -k_radius; dy <= k_radius; ++dy) {
        for (int dx = -k_radius; dx <= k_radius; ++dx) {
            float val = tex2D<float>(texObj, x + dx + 0.5f, y + dy + 0.5f);
            sum += val * kKernel[kIdx++];
        }
    }

    dst[y * width + x] = fabsf(sum);
}

// Added: Global Memory Kernel (for 3 channels)
// Why not use Texture?
// 1. cudaCreateChannelDesc does not support uchar3.
// 2. Converting uchar3 to uchar4 (padding) requires extra device memory and copy time.
// 3. On modern GPUs, for simple convolutions, global memory reads + L1 cache are similarly
// efficient.
__global__ void rdf_convolution_global_3c_kernel(const unsigned char* __restrict__ src,
                                                 size_t src_pitch,
                                                 float* __restrict__ dst,
                                                 int width,
                                                 int height,
                                                 int k_radius)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum_r       = 0.0f;
    float sum_g       = 0.0f;
    float sum_b       = 0.0f;
    int kIdx          = 0;
    const float scale = 1.0f / 255.0f;

    for (int dy = -k_radius; dy <= k_radius; ++dy) {

        int cur_y                    = clamp_idx(y + dy, height);
        const unsigned char* row_ptr = src + cur_y * src_pitch;

        for (int dx = -k_radius; dx <= k_radius; ++dx) {
            int cur_x = clamp_idx(x + dx, width);

            // 读取 BGR
            int px_idx  = cur_x * 3;
            float val_b = row_ptr[px_idx + 0] * scale;
            float val_g = row_ptr[px_idx + 1] * scale;
            float val_r = row_ptr[px_idx + 2] * scale;

            float k_val = kKernel[kIdx++];

            sum_b += val_b * k_val;
            sum_g += val_g * k_val;
            sum_r += val_r * k_val;
        }
    }

    // Add: |Sum_R| + |Sum_G| + |Sum_B|
    dst[y * width + x] = fabsf(sum_r) + fabsf(sum_g) + fabsf(sum_b);
}

std::vector<cv::Mat> rdf_cuda(std::span<const cv::Mat> alignedStack, const cv::Mat& rdfKernel)
{
    if (alignedStack.empty()) return {};

    int width               = alignedStack[0].cols;
    int height              = alignedStack[0].rows;
    int channels            = alignedStack[0].channels();
    size_t num_alignedStack = alignedStack.size();

    std::vector<cv::Mat> focusVolume(num_alignedStack);

    // 1. Prepare RDF Kernel
    // -------------------------------------------------
    // int k_radius = outer_r;
    int kRadius = rdfKernel.rows / 2;

    // copy kernel to constant memory
    size_t k_bytes = rdfKernel.total() * sizeof(float);
    if (k_bytes > sizeof(kKernel)) {
        std::cerr << "Error: Kernel size too large for constant memory!" << std::endl;
        return {};
    }
    checkCudaErrors(cudaMemcpyToSymbol(kKernel, rdfKernel.ptr<float>(), k_bytes));

    float* d_dst;
    checkCudaErrors(cudaMalloc(&d_dst, width * height * sizeof(float)));
    dim3 block(32, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // 2. Pre CUDA resources (Single Channel: CUDA Array for Texture)
    // -------------------------------------------------
    if (channels == 1) {

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
        cudaArray* cuArray;
        checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, width, height));

        // 准备 Texture Object 参数
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeMirror; // correspond to cv::BORDER_REFLECT
        texDesc.addressMode[1] = cudaAddressModeMirror;
        texDesc.filterMode
            = cudaFilterModePoint; // no linear interpolation needed, use nearest neighbor
        texDesc.readMode = cudaReadModeNormalizedFloat; // key optimization: automatically convert
                                                        // uchar[0-255] to float[0.0-1.0]
        texDesc.normalizedCoords = 0; // use non-normalized coordinates (0..W, 0..H)

        cudaTextureObject_t texObj = 0;

        // size_t dst_pitch; // use pitch memory alignment for better performance
        // checkCudaErrors(cudaMallocPitch(&d_dst, &dst_pitch, width * sizeof(float), height));

        // prepare pitched memory for output (for better performance, we allocate once and reuse in
        // loop)

        for (size_t i = 0; i < num_alignedStack; ++i) {
            const cv::Mat& src = alignedStack[i];
            checkCudaErrors(cudaMemcpy2DToArray(cuArray,
                                                0,
                                                0,
                                                src.data,
                                                src.step,
                                                width * sizeof(unsigned char),
                                                height,
                                                cudaMemcpyHostToDevice));

            checkCudaErrors(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

            // Launch RDF convolution kernel
            rdf_convolution_tex_kernel<<<grid, block>>>(texObj, d_dst, width, height, kRadius);

            // destroy texture object
            checkCudaErrors(cudaDestroyTextureObject(texObj));

            // [Device -> Host] copy result back
            focusVolume[i].create(height, width, CV_32F);
            checkCudaErrors(cudaMemcpy(focusVolume[i].ptr<float>(),
                                       d_dst,
                                       width * height * sizeof(float),
                                       cudaMemcpyDeviceToHost));
        }

        // 4. Clean up resources
        checkCudaErrors(cudaFreeArray(cuArray));
    }
    // 3 Channels (Global Memory + Pitch)
    else {
        unsigned char* d_src;
        size_t src_pitch;
        checkCudaErrors(
            cudaMallocPitch(&d_src, &src_pitch, width * 3 * sizeof(unsigned char), height));

        for (size_t i = 0; i < num_alignedStack; ++i) {
            const cv::Mat& src = alignedStack[i];

            // Upload
            checkCudaErrors(cudaMemcpy2D(d_src,
                                         src_pitch,
                                         src.data,
                                         src.step,
                                         width * 3 * sizeof(unsigned char),
                                         height,
                                         cudaMemcpyHostToDevice));

            // Compute
            rdf_convolution_global_3c_kernel<<<grid, block>>>(
                d_src, src_pitch, d_dst, width, height, kRadius);

            // Download
            focusVolume[i].create(height, width, CV_32F);
            checkCudaErrors(cudaMemcpy(focusVolume[i].ptr<float>(),
                                       d_dst,
                                       width * height * sizeof(float),
                                       cudaMemcpyDeviceToHost));
        }
        checkCudaErrors(cudaFree(d_src));
    }
    checkCudaErrors(cudaFree(d_dst));

    return focusVolume;
}

// bilinear transform
__device__ float bilinear_pixel(unsigned char* src, int w, int h, int step, float x, float y)
{
    if (x < 0 || x > w - 1 || y < 0 || y > h - 1) return 0.0f;

    int x_l = floorf(x);
    int y_l = floorf(y);
    int x_h = min(x_l + 1, w - 1);
    int y_h = min(y_l + 1, h - 1);

    float dx  = x - x_l;
    float dy  = y - y_l;
    float w00 = (1 - dx) * (1 - dy);
    float w10 = dx * (1 - dy);
    float w01 = (1 - dx) * dy;
    float w11 = dx * dy;

    // Note: step is the row byte size of cv::Mat
    unsigned char p00 = src[y_l * step + x_l];
    unsigned char p10 = src[y_l * step + x_h];
    unsigned char p01 = src[y_h * step + x_l];
    unsigned char p11 = src[y_h * step + x_h];

    return p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11;
}

// Core Warp Kernel
// H_inv: Inverse transformation matrix (Target -> Source mapping)
__global__ void warp_texture_kernel(cudaTextureObject_t texObj,
                                    unsigned char* dst,
                                    int width,
                                    int height,
                                    int step,
                                    float h00,
                                    float h01,
                                    float h02,
                                    float h10,
                                    float h11,
                                    float h12,
                                    float h20,
                                    float h21,
                                    float h22)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // 1. 计算透视变换坐标 (Target -> Source)
    // 这里的 H 应该是 H_inverse
    float z = h20 * x + h21 * y + h22;
    // 避免除零，尽管在良性变换中极少出现
    float w = (fabs(z) > 1e-6f) ? 1.0f / z : 0.0f;

    float src_x = (h00 * x + h01 * y + h02) * w;
    float src_y = (h10 * x + h11 * y + h12) * w;

    // 2. 硬件纹理采样 (自动双线性插值)
    // tex2D<float> 返回归一化的 float 值 (0.0 - 255.0)，因为我们在创建纹理时指定了
    // ReadModeElementType
    float val = tex2D<float>(texObj, src_x, src_y);

    // 3. 写入显存
    dst[y * step + x] = (unsigned char)(val + 0.5f);
}

// 封装类：管理显存生命周期，避免重复 malloc/free
class CudaWarpProcessor
{
public:
    int width_, height_;
    void* d_src_array_          = nullptr; // CUDA Array 用于纹理绑定
    unsigned char* d_dst_       = nullptr;
    cudaTextureObject_t texObj_ = 0;
    cudaArray* cuArray_         = nullptr;

    CudaWarpProcessor(int w, int h)
        : width_(w)
        , height_(h)
    {
        // 1. 分配输出显存 (Linear Memory)
        checkCudaErrors(cudaMalloc(&d_dst_, w * h * sizeof(unsigned char)));

        // 2. 分配输入显存 (CUDA Array，专为纹理优化)
        cudaChannelFormatDesc channelDesc
            = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
        checkCudaErrors(cudaMallocArray(&cuArray_, &channelDesc, w, h));

        // 3. 创建资源描述符
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray_;

        // 4. 创建纹理描述符
        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeBorder; // 越界部分填充零 (黑色)
        texDesc.addressMode[1] = cudaAddressModeBorder;
        texDesc.filterMode     = cudaFilterModeLinear;    // 硬件双线性插值
        texDesc.readMode       = cudaReadModeElementType; // 读原始数据类型，这里其实建议读 float
                                                          // 归一化，但为了简单用 ElementType
        texDesc.normalizedCoords = 0;                     // 使用像素坐标 (0 ~ width) 而不是 (0 ~ 1)

        checkCudaErrors(cudaCreateTextureObject(&texObj_, &resDesc, &texDesc, NULL));
    }

    ~CudaWarpProcessor()
    {
        if (texObj_) checkCudaErrors(cudaDestroyTextureObject(texObj_));
        if (cuArray_) checkCudaErrors(cudaFreeArray(cuArray_));
        if (d_dst_) checkCudaErrors(cudaFree(d_dst_));
    }

    // 执行单张图片的 Warp
    void process(const unsigned char* h_src, unsigned char* h_dst, const float* h_inv_data)
    {
        // 1. 将 Host 数据拷贝到 CUDA Array (纹理内存)
        // 5120x5120 的拷贝是瓶颈之一，如果有可能，输入数据最好是 Pinned Memory
        checkCudaErrors(cudaMemcpyToArray(cuArray_,
                                          0,
                                          0,
                                          h_src,
                                          width_ * height_ * sizeof(unsigned char),
                                          cudaMemcpyHostToDevice));

        // 2. 配置 Grid/Block
        dim3 block(32, 32);
        dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);

        // 3. 启动 Kernel
        // 传递 H_inverse 的 9 个参数
        warp_texture_kernel<<<grid, block>>>(texObj_,
                                             d_dst_,
                                             width_,
                                             height_,
                                             width_, // step = width (tightly packed)
                                             h_inv_data[0],
                                             h_inv_data[1],
                                             h_inv_data[2],
                                             h_inv_data[3],
                                             h_inv_data[4],
                                             h_inv_data[5],
                                             h_inv_data[6],
                                             h_inv_data[7],
                                             h_inv_data[8]);

        // 4. 等待完成并拷贝回 Host
        checkCudaErrors(cudaMemcpy(
            h_dst, d_dst_, width_ * height_ * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    }
};

struct ScopedCudaProcessor
{
    void* ptr = nullptr;
    ScopedCudaProcessor(int w, int h)
    {
        // ptr = create_warp_processor(w, h);
        ptr = new CudaWarpProcessor(w, h);
    }
    ~ScopedCudaProcessor()
    {
        if (ptr) delete static_cast<CudaWarpProcessor*>(ptr);
        ptr = nullptr;
    }
};

// class EccAligner
// {
// public:
void ecc_align_cuda(std::span<const cv::Mat> inputImages,
                    size_t refIndex,
                    std::vector<cv::Mat>& outputImages)
{
    if (inputImages.empty()) return;

    const size_t numImages = inputImages.size();
    const cv::Size imgSize = inputImages[0].size();

    // 预分配输出
    outputImages.resize(numImages);

    // --- 阶段 0: 准备 Reference Image ---
    cv::Mat refGray;
    if (inputImages[refIndex].channels() == 3) {
        cv::cvtColor(inputImages[refIndex], refGray, cv::COLOR_BGR2GRAY);
    }
    else {
        refGray = inputImages[refIndex].clone();
    }

    // 计算缩放比例：目标长边 1024
    // 对于 5120x5120，scale = 0.2
    float targetDim = 1024.0f;
    float scale     = targetDim / std::max(imgSize.width, imgSize.height);
    if (scale > 1.0f) scale = 1.0f;

    cv::Mat refSmall;
    cv::resize(refGray, refSmall, cv::Size(), scale, scale);

    // 存储所有计算出的变换矩阵 (Forward: Src -> Ref)
    std::vector<cv::Mat> homographies(numImages);

    // --- 阶段 1: CPU 并行计算 ECC (OpenMP) ---
    // 这是计算密集型任务，OpenMP 可以满载 CPU
    std::cout << "[Step 1] Calculating ECC matrices on CPU..." << std::endl;

#pragma omp parallel for
    for (int i = 0; i < (int)numImages; ++i) {
        if (i == (int)refIndex) {
            homographies[i] = cv::Mat::eye(3, 3, CV_32F);
            outputImages[i] = refGray; // Reference 直接拷贝
            continue;
        }

        // 预处理当前帧
        cv::Mat currGray;
        if (inputImages[i].channels() == 3) {
            cv::cvtColor(inputImages[i], currGray, cv::COLOR_BGR2GRAY);
        }
        else {
            currGray = inputImages[i]; // 引用
        }

        cv::Mat currSmall;
        cv::resize(currGray, currSmall, cv::Size(), scale, scale);

        // ECC 初始化
        cv::Mat H_small = cv::Mat::eye(3, 3, CV_32F);
        try {
            // 终止条件：50次迭代够了，不需要5000次，精度1e-4在1024px下已经很准了
            cv::findTransformECC(
                currSmall,
                refSmall,
                H_small,
                cv::MOTION_HOMOGRAPHY,
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 1e-4));
        }
        catch (...) {
            // 如果 ECC 失败，保持单位矩阵
            H_small = cv::Mat::eye(3, 3, CV_32F);
        }

        // 还原到原图尺度: H_full = S_inv * H_small * S
        // S = diag(scale, scale, 1)
        // 代数推导简化版:
        // h00, h01, h02/s
        // h10, h11, h12/s
        // h20*s, h21*s, h22
        cv::Mat H_full = H_small.clone();
        H_full.at<float>(0, 2) /= scale;
        H_full.at<float>(1, 2) /= scale;
        H_full.at<float>(2, 0) *= scale;
        H_full.at<float>(2, 1) *= scale;

        homographies[i] = H_full;
    }

    // --- 阶段 2: GPU 串行 Warp ---
    // 显存资源是宝贵的，串行复用显存比并行分配更安全
    std::cout << "[Step 2] Warping images on GPU..." << std::endl;

    ScopedCudaProcessor cudaProc(imgSize.width, imgSize.height);

    for (size_t i = 0; i < numImages; ++i) {
        if (i == refIndex) continue;

        // 准备逆矩阵 (Inverse Mapping for Warping)
        // OpenCV warpPerspective 默认也是用逆矩阵原理，但如果没传
        // WARP_INVERSE_MAP，它内部会求逆 我们自定义的 Kernel 逻辑是: pixel_src = H_inv *
        // pixel_dst 所以我们需要将 ECC 算出的 (Src -> Ref) 矩阵直接传进去？ ECC 算出的是:
        // Ref_coords = H * Curr_coords (即 Src 到 Dst) Warp 逻辑通常遍历 Dst 坐标，找 Src
        // 坐标，所以我们需要 H_inv = H.inv() 如果 ECC 算出的是把 Current 对齐到 Ref，那变换是
        // Current -> Ref。 我们遍历 Ref 网格，需要找对应的 Current 坐标，所以需要 Ref ->
        // Current 的变换 (H.inv())

        cv::Mat H     = homographies[i];
        cv::Mat H_inv = H.inv();

        // 确保矩阵数据连续且为 float
        std::vector<float> h_data(9);
        for (int k = 0; k < 9; ++k) h_data[k] = H_inv.at<float>(k / 3, k % 3);

        // 分配输出 Mat
        outputImages[i].create(imgSize, CV_8UC1);

        // 获取输入指针 (灰度图)
        cv::Mat inputGray;
        if (inputImages[i].channels() == 3)
            cv::cvtColor(inputImages[i], inputGray, cv::COLOR_BGR2GRAY);
        else
            inputGray = inputImages[i];

        // 调用 GPU
        // execute_warp(cudaProc.ptr, inputGray.data, outputImages[i].data, h_data.data());
        static_cast<CudaWarpProcessor*>(cudaProc.ptr)
            ->process(inputGray.data, outputImages[i].data, h_data.data());

        // Logging (Optional)
        if (i % 10 == 0) std::cout << "Warped " << i << "/" << numImages << std::endl;
    }
}
// };
/**
 * @file final_project_unified.cpp
 * @brief Unified Depth from Focus (DFF) Implementation
 *
 * This file integrates multiple DFF pipeline implementations into a single,
 * well-structured, object-oriented C++23 program.
 *
 * Features:
 * - Robust image alignment using ORB features and RANSAC
 * - Focus measure computation (SML) with CPU/CUDA acceleration
 * - Subpixel depth estimation with confidence map
 * - Depth map refinement using guided filter or bilateral filter
 * - Weight-based fusion for all-in-focus image generation
 * - Physical depth conversion
 *
 * @author wwf
 * @date 2025-12-03
 */

#include "opencv2/core.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <algorithm>
#include <cmath>
#include <execution>
#include <filesystem>
#include <functional>

#include <fstream>
#include <future>
#include <iostream>
#include <ranges>
#ifdef __cpp_lib_mdspan
#include <mdspan>
#endif
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// OpenCV
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#ifdef HAVE_OPENCV_XIMGPROC
#include <opencv2/ximgproc.hpp>
#endif

// #include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

// CUDA support (conditional)
#ifdef USE_CUDA
#include "cuda_operators.hpp"
#endif

#include "my_cmd_parser.hpp"
#include "my_logger.hpp"

namespace fs    = std::filesystem;
namespace views = std::ranges::views;
namespace rng   = std::ranges;


// ============================================================================
// Configuration Parameters
// ============================================================================

/**
 * @brief Configuration field list (single source of truth)
 *
 * Format: X(type, name, default_value, description)
 */
#define DFF_CONFIG_FIELDS                                                                          \
    X(bool, useColor, false, "Use rgb image instead of grayscale")                                 \
    X(bool, useCuda, false, "Use CUDA acceleration")                                               \
    X(bool, useEcc, false, "Use ECC alignment")                                                    \
    X(bool, useRDF, false, "Use RDF to compute focus volume")                                      \
    X(bool, useGuidedFilter, false, "Use guided filter")                                            \
    X(bool, useFusionRefine, false, "Use weight-based fusion")                                      \
    X(bool, useParallelExec, false, "Use parallel execution")                                       \
    X(bool, useInpaint, false, "Use inpainting")                                                    \
    X(bool, saveData, false, "Save result data to txt")                                             \
    X(int, orbNumFeatures, 1000, "ORB features count")                                             \
    X(double, eccResizeFactor, 0.5, "ECC resize factor")                                           \
    X(int, eccMaxCount, 50, "ECC max count")                                                       \
    X(double, eccEpsilon, 1e-4, "ECC epsilon")                                                     \
    X(int, smlWindowSize, 5, "SML window size (Only used when useRDF is false)")                   \
    X(int, rdfOuterR, 3, "RDF outer radius")                                                       \
    X(int, rdfInnerR, 1, "RDF inner radius")                                                       \
    X(double, ransacThreshold, 5.0, "RANSAC threshold")                                            \
    X(int, refinedGaussKSize, 5, "Blur kernel size in refined depth map")                          \
    X(int, fusionGaussKSize, 7, "Blur kernel size in fusion all in focus image")                   \
    X(int, guidedFilterRadius, 8, "Guided filter radius")                                          \
    X(double, guidedFilterEps, 0.01, "Guided filter eps")                                          \
    X(int, bilateralFilterD, 9, "Bilateral diameter")                                              \
    X(double, bilateralSigmaColor, 10.0, "Bilateral color sigma")                                  \
    X(double, bilateralSigmaSpace, 5.0, "Bilateral space sigma")                                   \
    X(float, physicalDepthStart, 1.0f, "Start distance (mm)")                                      \
    X(float, physicalDepthStep, 2.0f, "Step distance (mm)")                                        \
    X(std::string, outputSubDir, "unified", "Output directory")

/**
 * @brief DFF Pipeline Configuration
 */
struct DFFConfig
{
    // Generate member variables
#define X(type, name, default_val, doc) type name = default_val; ///< doc
    DFF_CONFIG_FIELDS
#undef X

    static void addCommandLineArguments(MyCmdParser& parser)
    {
#define X(type, name, default_val, doc)                                                            \
    parser.addArgument<type>(#name, doc, default_val);
        DFF_CONFIG_FIELDS
#undef X
    }

    /**
     * @brief Deserialize from YAML
     */
    static DFFConfig fromYaml(const YAML::Node& node)
    {
        DFFConfig config;
#define X(type, name, default_val, doc)                                                            \
    if (node[#name]) { config.name = node[#name].as<type>(); }
        DFF_CONFIG_FIELDS
#undef X
        return config;
    }

    static void yamlToParser(const YAML::Node& node, MyCmdParser& parser)
    {
#define X(type, name, default_val, doc)                                                            \
if (node[#name]) { parser.addArgument<type>(#name, doc, node[#name].as<type>()); }
        DFF_CONFIG_FIELDS
#undef X
    }

    /**
     * @brief Serialize to YAML
     */
    [[nodiscard]] std::string toYaml() const
    {
        YAML::Emitter out;
        out << YAML::BeginMap;
#define X(type, name, default_val, doc)                                                            \
    out << YAML::Key << #name << YAML::Value << name << YAML::Comment(doc);
        DFF_CONFIG_FIELDS
#undef X
        out << YAML::EndMap;
        return std::string(out.c_str());
    }

    void overrideFromCmdParser(const MyCmdParser& parser)
    {
#define X(type, name, default_val, doc)                                                            \
    if (auto val = parser.getValue<type>(#name); val.has_value()) { name = val.value(); }
        DFF_CONFIG_FIELDS
#undef X
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Helper to specify OpenCV output parameters (improves readability)
 */
template <typename T>
    requires(!std::is_const_v<T>)
[[nodiscard]] constexpr T& CVOutput(T& t) // NOLINT
{
    return t;
}
template <typename Op>
auto parallel_dispatch(bool useParallel, Op&& op)
{
    if (useParallel) { return op(std::execution::par); }
    return op(std::execution::seq);
}

// ============================================================================
// Filter Implementations (CPU fallback when OpenCV ximgproc unavailable)
// ============================================================================
/**
 * @brief Custom Guided Filter implementation
 */
class GuidedFilter
{
public:
    static cv::Mat apply(const cv::Mat& guide, const cv::Mat& src, int radius, double eps)
    {
#ifdef HAVE_OPENCV_XIMGPROC
        cv::Mat result;
        cv::ximgproc::guidedFilter(guide, src, CVOutput(result), radius, eps, -1);
        return result;
#else
        return applyManual(guide, src, radius, eps);
#endif
    }

private:
    static cv::Mat applyManual(const cv::Mat& I, const cv::Mat& p, int r, double eps)
    {
        // Convert to float for precision
        cv::Mat imageFloat;
        cv::Mat pf;
        I.convertTo(imageFloat, CV_32F);
        p.convertTo(pf, CV_32F);

        cv::Size winSize((2 * r) + 1, (2 * r) + 1);

        // Compute means
        cv::Mat meanI;
        cv::Mat meanP;
        cv::Mat meanII;
        cv::Mat meanIp;
        cv::boxFilter(imageFloat, meanI, CV_32F, winSize);
        cv::boxFilter(pf, meanP, CV_32F, winSize);
        cv::boxFilter(imageFloat.mul(imageFloat), meanII, CV_32F, winSize);
        cv::boxFilter(imageFloat.mul(pf), meanIp, CV_32F, winSize);

        // Compute variance and covariance
        cv::Mat varI   = meanII - meanI.mul(meanI);
        cv::Mat convIp = meanIp - meanI.mul(meanP);

        // Compute linear coefficients
        cv::Mat a;
        cv::Mat b;
        cv::divide(convIp, varI + eps, a);
        b = meanP - a.mul(meanI);

        // Average coefficients
        cv::Mat meanA;
        cv::Mat meanB;
        cv::boxFilter(a, meanA, CV_32F, winSize);
        cv::boxFilter(b, meanB, CV_32F, winSize);

        // Generate output
        cv::Mat qF = meanA.mul(imageFloat) + meanB;

        cv::Mat q;
        if (p.depth() == CV_8U) { qF.convertTo(q, CV_8U); }
        else {
            q = qF;
        }
        return q;
    }
};

/**
 * @brief Custom Joint Bilateral Filter implementation
 */
class JointBilateralFilter
{
public:
    static cv::Mat
    apply(const cv::Mat& guide, const cv::Mat& src, int d, double sigmaColor, double sigmaSpace)
    {
#ifdef HAVE_OPENCV_XIMGPROC
        cv::Mat result;
        cv::Mat floatGuide;
        if (result.type() == CV_8U) { guide.convertTo(floatGuide, CV_32F); }
        else
            floatGuide = guide;
        cv::ximgproc::jointBilateralFilter(
            floatGuide, src, CVOutput(result), d, sigmaColor, sigmaSpace);
        return result;
#else
        return applyManual(guide, src, d, sigmaColor, sigmaSpace);
#endif
    }

private:
    static cv::Mat applyManual(
        const cv::Mat& guide, const cv::Mat& src, int d, double sigmaColor, double sigmaSpace)
    {
        int r = d / 2;
        if (r <= 0) r = 1;
        d = 2 * r + 1;

        // Precompute color weights LUT
        double gaussColorCoeff = -0.5 / (sigmaColor * sigmaColor);
        auto colorWeights
            = views::iota(0, 256)
              | views::transform([&](int i) -> float { return std::expf(i * i * gaussColorCoeff); })
              | rng::to<std::vector<float>>();

        // Precompute spatial weights
        double gaussSpaceCoeff = -0.5 / (sigmaSpace * sigmaSpace);
        cv::Mat spatialKernel(d, d, CV_32F);
        for (int dy = -r; dy <= r; ++dy) {
            for (int dx = -r; dx <= r; ++dx) {
                float dist                              = std::sqrtf(dx * dx + dy * dy);
                spatialKernel.at<float>(dy + r, dx + r) = std::expf(dist * dist * gaussSpaceCoeff);
            }
        }

        bool isSrc32F = (src.depth() == CV_32F);
        cv::Mat dst   = cv::Mat::zeros(src.size(), src.type());
        int width     = src.cols;
        int height    = src.rows;

        // Parallel processing
        cv::parallel_for_(cv::Range(0, height), [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                const auto* guideRow  = guide.ptr<uchar>(y);
                const auto* srcRow8u  = isSrc32F ? nullptr : src.ptr<uchar>(y);
                const auto* srcRow32f = isSrc32F ? src.ptr<float>(y) : nullptr;
                auto* dstRow8u        = isSrc32F ? nullptr : dst.ptr<uchar>(y);
                auto* dstRow32f       = isSrc32F ? dst.ptr<float>(y) : nullptr;

                for (int x = 0; x < width; ++x) {
                    float centerGuide = guideRow[x];
                    float sumWeight   = 0.0f;
                    float sumValue    = 0.0f;

                    for (int dy = -r; dy <= r; ++dy) {
                        int ny                  = std::clamp(y + dy, 0, height - 1);
                        const auto* neighGuide  = guide.ptr<uchar>(ny);
                        const auto* neighSrc8u  = isSrc32F ? nullptr : src.ptr<uchar>(ny);
                        const auto* neighSrc32f = isSrc32F ? src.ptr<float>(ny) : nullptr;

                        for (int dx = -r; dx <= r; ++dx) {
                            int nx = std::clamp(x + dx, 0, width - 1);

                            float neighGuideVal = neighGuide[nx];
                            int colorDiff = std::abs(static_cast<int>(centerGuide - neighGuideVal));

                            float wColor = colorWeights[colorDiff];
                            float wSpace = spatialKernel.at<float>(dy + r, dx + r);
                            float weight = wColor * wSpace;

                            float pixelValue = isSrc32F ? neighSrc32f[nx] : neighSrc8u[nx];
                            sumWeight += weight;
                            sumValue += weight * pixelValue;
                        }
                    }

                    if (isSrc32F) { dstRow32f[x] = sumValue / sumWeight; }
                    else {
                        dstRow8u[x] = cv::saturate_cast<uchar>(sumValue / sumWeight);
                    }
                }
            }
        });

        return dst;
    }
};

// ============================================================================
// Main DFF Pipeline Class
// ============================================================================

/**
 * @brief Depth from Focus (DFF) Pipeline
 *
 * This class encapsulates the complete DFF workflow:
 * 1. Image Alignment
 * 2. Focus Measure Computation (SML)
 * 3. Depth Estimation with Confidence
 * 4. Depth Refinement
 * 5. Physical Depth Conversion
 */
class DepthFromFocusPipeline
{
public:
    /**
     * @brief Constructor
     * @param config Pipeline configuration
     */
    explicit DepthFromFocusPipeline(DFFConfig config = DFFConfig {})
        : config_(std::move(config))
    {
    }

    /**
     * @brief Run the complete DFF pipeline
     * @param inputImages Raw input image stack (BGR format)
     * @return True if successful
     */
    bool process(std::span<const cv::Mat> inputImages)
    {
        if (inputImages.empty()) {
            Log.error("Input image stack is empty!");
            return false;
        }

        numImages_ = inputImages.size();
        imageSize_ = inputImages[0].size();

        // Generate physical distances
        generatePhysicalDistances();

        // Step 1: Align images
        if (!alignImages(inputImages)) { return false; }

        // Step 2: Compute focus volume
        if (!computeFocusVolume()) { return false; }

        // Step 3: Estimate depth and confidence
        if (!estimateDepthAndConfidence()) { return false; }

        // Step 4: Refine depth map
        if (!refineDepthMap()) { return false; }

        // Step 5: Convert to physical depth
        if (!convertToPhysicalDepth()) { return false; }

        return true;
    }

    /**
     * @brief Save results to disk
     * @param outputDir Output directory path
     * @return True if successful
     */
    [[nodiscard]] bool saveResults(const fs::path& outputDir) const
    {
        if (!fs::exists(outputDir)) { fs::create_directories(outputDir); }

        bool success = true;

        std::vector<std::future<bool>> saveFutures;

        if (config_.saveData) {
            // save refined depth index
            saveFutures.emplace_back(std::async(std::launch::async, [this, outputDir]() {
                return saveMatToTxt(refinedDepthMap_, outputDir / "refined_depth_index_map.txt");
            }));

            // save physical depth
            saveFutures.emplace_back(std::async(std::launch::async, [this, outputDir]() {
                return saveMatToTxt(physicalDepthMap_, outputDir / "physical_depth_map.txt");
            }));

            // save confidence map
            saveFutures.emplace_back(std::async(std::launch::async, [this, outputDir]() {
                return saveMatToTxt(confidenceMap_, outputDir / "confidence_map.txt");
            }));

            Log.info("Started {} async file save tasks", saveFutures.size());
        }

        // Depth index map (colormap)
        {
            cv::Mat depthVis;
            refinedDepthMap_.convertTo(depthVis, CV_8U, 255.0 / numImages_);
            // cv::normalize(refinedDepthMap_, depthVis, 0, 255, cv::NORM_MINMAX, CV_8U);
            // cv::applyColorMap(depthVis, depthVis, cv::COLORMAP_JET);
            success &= cv::imwrite((outputDir / "depth_index_map.png").string(), depthVis);
        }

        // Physical depth map (colormap)
        {
            cv::Mat physicalVis;
            cv::normalize(physicalDepthMap_, physicalVis, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::applyColorMap(physicalVis, physicalVis, cv::COLORMAP_JET);
            success &= cv::imwrite((outputDir / "physical_depth_map.png").string(), physicalVis);
        }

        // Confidence map
        {
            cv::Mat confVis;
            cv::log(confidenceMap_ + 1.0, confVis);
            cv::normalize(confVis, confVis, 0, 255, cv::NORM_MINMAX, CV_8U);
            // cv::normalize(confVis, confVis, 0, 255, cv::NORM_MINMAX, CV_8U);
            success &= cv::imwrite((outputDir / "confidence_map.png").string(), confVis);
        }

        // All-in-focus image
        success &= cv::imwrite((outputDir / "all_in_focus.png").string(), allInFocusImage_);

        if (config_.saveData) {
            for (auto& fut : saveFutures) {
                try {
                    success &= fut.get();
                }
                catch (const std::exception& e) {
                    Log.error("Async save task failed: {}", e.what());
                    success = false;
                }
            }
            Log.info("All async save tasks completed");
        }

        if (success) { Log.info("Results saved to: {}", outputDir.string()); }
        else {
            Log.error("Failed to save some results");
        }

        return success;
    }

    // Getters for results
    [[nodiscard]] const cv::Mat& getRefinedDepthMap() const { return refinedDepthMap_; }
    [[nodiscard]] const cv::Mat& getPhysicalDepthMap() const { return physicalDepthMap_; }
    [[nodiscard]] const cv::Mat& getConfidenceMap() const { return confidenceMap_; }
    [[nodiscard]] const cv::Mat& getAllInFocusImage() const { return allInFocusImage_; }

private:
    // === Configuration ===
    DFFConfig config_;

    // === Pipeline Data ===
    size_t numImages_ = 0;
    cv::Size imageSize_;
    std::vector<float> physicalDistances_;

    std::vector<cv::Mat> alignedImages_; ///< Aligned grayscale images
    std::vector<cv::Mat> focusVolume_;   ///< Focus measure maps (SML)
    cv::Mat depthIndexMap_;              ///< Raw depth index map
    cv::Mat confidenceMap_;              ///< Confidence map
    cv::Mat refinedDepthMap_;            ///< Refined depth index map
    cv::Mat allInFocusImage_;            ///< All-in-focus image
    cv::Mat physicalDepthMap_;           ///< Physical depth map (mm)

    /**
     * @brief Generate physical distance array
     */
    void generatePhysicalDistances()
    {
        physicalDistances_
            = views::iota(0) | views::take(numImages_) | views::transform([this](int i) {
                  return config_.physicalDepthStart + (i * config_.physicalDepthStep);
              })
              | rng::to<std::vector<float>>();
    }

    /**
     * @brief Step 1: Align image stack using ORB features and RANSAC
     */
    bool alignImages(std::span<const cv::Mat> inputImages)
    {
        Log.info("[STEP 1] Aligning {} images using {}",
                 numImages_,
                 config_.useEcc ? "ECC" : "ORB features");

        alignedImages_.clear();
        alignedImages_.resize(numImages_);

        // Choose reference image (middle frame)
        std::size_t refIndex = numImages_ / 2;

        // Convert reference to grayscale
        cv::Mat refGray;
        if (inputImages[refIndex].channels() == 3) {
            cv::cvtColor(inputImages[refIndex], refGray, cv::COLOR_BGR2GRAY);
        }
        else {
            refGray = inputImages[refIndex].clone();
        }
        alignedImages_[refIndex] = inputImages[refIndex].clone();

        if (!config_.useEcc) {
            // Align by features
            alignByFeatures(inputImages, refIndex);
        }
        else {
            // Alternatively, align by ECC
            // if (config_.useCuda) { ecc_align_cuda(inputImages, refIndex, alignedImages_); }
            // else {

            alignByECC(inputImages, refIndex);
            // }
        }

        Log.info("Image alignment completed");
        return true;
    }
    void alignByFeatures(std::span<const cv::Mat> inputImages, size_t refIndex)
    {
        cv::Mat refGray;
        if (inputImages[refIndex].channels() == 3) {
            cv::cvtColor(inputImages[refIndex], refGray, cv::COLOR_BGR2GRAY);
        }
        else {
            refGray = inputImages[refIndex].clone();
        }
        alignedImages_[refIndex] = inputImages[refIndex].clone();

        // Initialize ORB detector
        auto orb = cv::ORB::create(config_.orbNumFeatures);
        std::vector<cv::KeyPoint> refKeypoints;
        cv::Mat refDescriptors;
        orb->detectAndCompute(
            refGray, cv::noArray(), CVOutput(refKeypoints), CVOutput(refDescriptors));

        auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);

        // Align each image to reference
        for (size_t i = 0; i < numImages_; ++i) {
            if (i == refIndex) continue;

            // Convert to grayscale
            cv::Mat currGray;
            if (inputImages[i].channels() == 3) {
                cv::cvtColor(inputImages[i], currGray, cv::COLOR_BGR2GRAY);
            }
            else {
                currGray = inputImages[i].clone();
            }

            // Detect and match features
            std::vector<cv::KeyPoint> currKeypoints;
            cv::Mat currDescriptors;
            orb->detectAndCompute(
                currGray, cv::noArray(), CVOutput(currKeypoints), CVOutput(currDescriptors));

            std::vector<std::vector<cv::DMatch>> knnMatches;
            matcher->knnMatch(currDescriptors, refDescriptors, CVOutput(knnMatches), 2);

            // Lowe's ratio test
            auto goodMatches = knnMatches | views::filter([](const auto& m) {
                                   return m.size() == 2 && m[0].distance < 0.75f * m[1].distance;
                               })
                               | views::transform([](const auto& m) { return m[0]; })
                               | rng::to<std::vector>();

            if (goodMatches.size() < 10) {
                alignedImages_[i] = inputImages[i];
                continue;
            }

            // Extract point correspondences
            std::vector<cv::Point2f> currPoints;
            std::vector<cv::Point2f> refPoints;
            for (const auto& match : goodMatches) {
                currPoints.push_back(currKeypoints[match.queryIdx].pt);
                refPoints.push_back(refKeypoints[match.trainIdx].pt);
            }

            // Compute homography with RANSAC
            try {
                cv::Mat H = cv::findHomography(
                    currPoints, refPoints, cv::RANSAC, config_.ransacThreshold);
                if (H.empty()) {
                    // Log.warn("Homography failed for image {}, using original", i);
                    alignedImages_[i] = inputImages[i].clone();
                }
                else {
                    cv::warpPerspective(inputImages[i], alignedImages_[i], H, imageSize_);
                }
            }
            catch (const cv::Exception& e) {
                Log.error("Alignment failed for image {}: {}", i, e.what());
                alignedImages_[i] = inputImages[i].clone();
            }
        }
    }

    void alignByECC(std::span<const cv::Mat> inputImages, size_t refIndex)
    {
        auto resizeFactor = static_cast<float>(config_.eccResizeFactor);
        cv::Mat refGray;
        if (inputImages[refIndex].channels() == 3) {
            cv::cvtColor(inputImages[refIndex], refGray, cv::COLOR_BGR2GRAY);
        }
        else {
            refGray = inputImages[refIndex].clone();
        }
        alignedImages_[refIndex] = inputImages[refIndex];

        cv::Mat refSmall;
        // Resize for speed
        cv::resize(refGray, refSmall, cv::Size(), resizeFactor, resizeFactor);

        cv::Mat H             = cv::Mat::eye(3, 3, CV_32F);
        cv::Mat S             = cv::Mat::eye(3, 3, CV_32F);
        S.at<float>(0, 0)     = 1.0f / resizeFactor;
        S.at<float>(1, 1)     = 1.0f / resizeFactor;
        cv::Mat S_inv         = cv::Mat::eye(3, 3, CV_32F);
        S_inv.at<float>(0, 0) = resizeFactor;
        S_inv.at<float>(1, 1) = resizeFactor;

        for (size_t i = 0; i < numImages_; ++i) {
            if (i == refIndex) continue;

            cv::Mat currGray;
            if (inputImages[i].channels() == 3) {
                cv::cvtColor(inputImages[i], currGray, cv::COLOR_BGR2GRAY);
            }
            else {
                currGray = inputImages[i].clone();
            }
            cv::Mat currSmall;
            cv::resize(currGray, currSmall, cv::Size(), resizeFactor, resizeFactor);
            try {
                Log.debug("find transform ecc from image {}.", i);
                cv::findTransformECC(
                    currSmall,
                    refSmall,
                    H,
                    cv::MOTION_HOMOGRAPHY,

                    cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                     config_.eccMaxCount,
                                     config_.eccEpsilon));
                Log.debug("successfully find the transform");

                cv::Mat H_full = S * H * S_inv;

                Log.debug("warp perspective for image {}.", i);
                cv::warpPerspective(inputImages[i],
                                    alignedImages_[i],
                                    H_full,
                                    imageSize_,
                                    cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);
                Log.debug("successfully warp the image.");

                // H = H_full;
            }
            catch (const cv::Exception& e) {
                Log.error("ECC alignment failed for image {}: {}", i, e.what());
                alignedImages_[i] = inputImages[i];
            }
        }
    }

    /**
     * @brief Step 2: Compute focus volume using SML (CPU or CUDA)
     */
    bool computeFocusVolume()
    {
        Log.info("[STEP 2] Computing focus volume");

        if (config_.useRDF) {
            if (config_.useCuda) {
#ifdef USE_CUDA
                auto rdfKernel = generate_RDF_kernel(config_.rdfInnerR, config_.rdfOuterR);
                focusVolume_   = rdf_cuda(alignedImages_, rdfKernel);
                Log.info("RDF: Focus volume computed using CUDA");
                return true;
#else
                Log.warn("CUDA support not available for RDF, falling back to CPU implementation");
#endif
            }
            return rdfCpu();
        }
        else {
            if (config_.useCuda) {
#ifdef USE_CUDA
                focusVolume_ = sml_cuda(alignedImages_, config_.smlWindowSize);
                Log.info("SML: Focus volume computed using CUDA");
                return true;
#else
                Log.warn("CUDA support not available, falling back to CPU implementation");
#endif
            }
            return sml_cpu();
        }
    }

    bool sml_cpu()
    {
        // CPU implementation
        focusVolume_.resize(numImages_);
        int floatType   = alignedImages_[0].channels() == 3 ? CV_32FC3 : CV_32F;
        cv::Mat kernelX = (cv::Mat_<float>(1, 3) << -1.0f, 2.0f, -1.0f);
        cv::Mat kernelY = (cv::Mat_<float>(3, 1) << -1.0f, 2.0f, -1.0f);
        cv::Size smlWindow(config_.smlWindowSize, config_.smlWindowSize);

        auto MLAPFilter // NOLINT
            = [&](const cv::Mat& inputImage) -> cv::Mat {
            // Modified Laplacian
            cv::Mat lapX;
            cv::Mat lapY;
            cv::filter2D(inputImage,
                         CVOutput(lapX),
                         floatType,
                         kernelX,
                         cv::Point(-1, -1),
                         0,
                         cv::BORDER_REFLECT);
            cv::filter2D(inputImage,
                         CVOutput(lapY),
                         floatType,
                         kernelY,
                         cv::Point(-1, -1),
                         0,
                         cv::BORDER_REFLECT);

            return cv::abs(lapX) + cv::abs(lapY);
        };
        std::vector<cv::Mat> channels;
        for (size_t i = 0; i < numImages_; ++i) {
            // Convert to float
            cv::Mat img32f;
            alignedImages_[i].convertTo(CVOutput(img32f), CV_32F, 1.0 / 255.0);

            cv::Mat filtered = MLAPFilter(img32f);

            // Box filter for SML
            cv::boxFilter(filtered,
                          filtered,
                          floatType,
                          smlWindow,
                          cv::Point(-1, -1),
                          false,
                          cv::BORDER_REFLECT);
            if (alignedImages_[i].channels() == 3) {
                cv::split(filtered, CVOutput(channels));
                focusVolume_[i] = channels[0] + channels[1] + channels[2];
            }
            else {
                focusVolume_[i] = filtered;
            }
        }

        Log.info("SML: Focus volume computed using CPU");
        return true;
    }

    bool rdfCpu()
    {
        focusVolume_.resize(numImages_);
        auto rdfKernel = generate_RDF_kernel(config_.rdfInnerR, config_.rdfOuterR);

        for (std::size_t i = 0; i < numImages_; ++i) {
            // Convert to float
            cv::Mat img32f;
            alignedImages_[i].convertTo(CVOutput(img32f), CV_32F, 1.0 / 255.0);
            std::vector<cv::Mat> channels;
            cv::split(img32f, CVOutput(channels));
            focusVolume_[i] = cv::Mat::zeros(imageSize_, CV_32F);
            for (const auto& ch : channels) {
                cv::Mat filteredCh;
                cv::filter2D(ch,
                             CVOutput(filteredCh),
                             CV_32F,
                             rdfKernel,
                             cv::Point(-1, -1),
                             0,
                             cv::BORDER_REFLECT);
                focusVolume_[i] += cv::abs(filteredCh);
            }
        }

        Log.info("RDF: Focus volume computed using CPU");
        return true;
    }

    /**
     * @brief Step 3: Estimate subpixel depth and confidence
     */
    bool estimateDepthAndConfidence()
    {
        Log.info("[STEP 3] Estimating depth and confidence");

        depthIndexMap_ = cv::Mat(imageSize_, CV_32F, cv::Scalar(0));
        confidenceMap_ = cv::Mat(imageSize_, CV_32F, cv::Scalar(0));

        std::vector<const float*> focusPtrs(numImages_);
        for (size_t i = 0; i < numImages_; ++i) { focusPtrs[i] = focusVolume_[i].ptr<float>(0); }
        const float* const* focusPtrsRaw = focusPtrs.data();

        auto* depthPtr = depthIndexMap_.ptr<float>(0);
        auto* confPtr  = confidenceMap_.ptr<float>(0);
        int numPixels  = imageSize_.width * imageSize_.height;

        auto zipView = views::zip(std::span(depthPtr, numPixels), std::span(confPtr, numPixels));
        // auto for_each = parallel_exec_if(config, std::for_each)
        parallel_dispatch(config_.useParallelExec, [&](auto&& policy) {
            std::for_each(std::forward<decltype(policy)>(policy),
                          zipView.begin(),
                          zipView.end(),
                          [depthPtr, focusPtrsRaw, this](auto&& pair) {
                              auto& [depth, conf] = pair;
                              size_t pixelIdx     = &depth - depthPtr;

                              // Find peak in focus stack using quadratic interpolation
                              float maxVal = -1.0f;
                              int maxIdx   = 0;

                              for (int k = 0; k < numImages_; ++k) {
                                  float val = focusPtrsRaw[k][pixelIdx];
                                  if (val > maxVal) {
                                      maxVal = val;
                                      maxIdx = k;
                                  }
                              }

                              // Subpixel refinement using parabola fitting
                              if (maxIdx > 0 && maxIdx < numImages_ - 1) {
                                  float prevVal = focusPtrsRaw[maxIdx - 1][pixelIdx];
                                  float currVal = maxVal;
                                  float nextVal = focusPtrsRaw[maxIdx + 1][pixelIdx];

                                  float denom = 2.0f * (prevVal - 2.0f * currVal + nextVal);
                                  if (std::abs(denom) > 1e-6f) {
                                      float offset = (prevVal - nextVal) / denom;
                                      offset       = std::clamp(offset, -1.0f, 1.0f);
                                      depth        = maxIdx + offset;
                                      maxVal
                                          = currVal
                                            - (0.25f * (prevVal - nextVal) * offset); // refined max
                                  }
                                  else {
                                      depth = static_cast<float>(maxIdx);
                                  }
                              }
                              else {
                                  depth = static_cast<float>(maxIdx);
                              }

                              conf = maxVal;
                          });
        });

        Log.info("Depth estimation completed");
        return true;
    }

    /**
     * @brief Step 4: Refine depth map using guided/bilateral filter
     */
    bool refineDepthMap()
    {
        Log.info("[STEP 4] Refining depth map using {} filter",
                 config_.useGuidedFilter ? "guided" : "bilateral");

        // Generate all-in-focus image
        if (config_.useFusionRefine) {
            Log.info("Generating all-in-focus image using weight-based fusion");
            if (alignedImages_[0].channels() == 3)
                allInFocusImage_ = generateAllInFocusWithFusion<cv::Vec3b>();
            else
                allInFocusImage_ = generateAllInFocusWithFusion<uchar>();
        }
        else {
            Log.info("Generating all-in-focus image using direct indexing");
            if (alignedImages_[0].channels() == 3)
                allInFocusImage_ = generateAllInFocusDirect<cv::Vec3b>();
            else
                allInFocusImage_ = generateAllInFocusDirect<uchar>();
        }
        Log.info("All-in-focus image generated");

        // Blur depth map slightly before filtering
        cv::Mat blurredDepth;
        cv::Size ksize(config_.refinedGaussKSize, config_.refinedGaussKSize);
        if (config_.refinedGaussKSize > 0) {
            cv::GaussianBlur(depthIndexMap_, blurredDepth, ksize, 0);
        }
        else {
            blurredDepth = depthIndexMap_.clone();
        }

        // Apply guided or bilateral filter
        if (config_.useGuidedFilter) {
            refinedDepthMap_ = GuidedFilter::apply(allInFocusImage_,
                                                   blurredDepth,
                                                   config_.guidedFilterRadius,
                                                   config_.guidedFilterEps);
        }
        else {
            refinedDepthMap_ = JointBilateralFilter::apply(allInFocusImage_,
                                                           blurredDepth,
                                                           config_.bilateralFilterD,
                                                           config_.bilateralSigmaColor,
                                                           config_.bilateralSigmaSpace);
        }

        // Optional: Inpaint low-confidence regions
        if (config_.useInpaint) inpaintLowConfidenceRegions();

        Log.info("Depth refinement completed");
        return true;
    }

    /**
     * @brief Generate all-in-focus image using direct indexing
     */
    template <typename ElementType>
    [[nodiscard]] cv::Mat generateAllInFocusDirect() const
    {
        cv::Mat result(imageSize_, alignedImages_[0].type());
        // auto* resultPtr = result.ptr<uchar>(0);
        auto* resultPtr = result.ptr<ElementType>();
        // auto alignedImageIter = alignedImages_.begin();
        const auto* depthPtr = depthIndexMap_.ptr<float>(0);
        int numPixels        = imageSize_.width * imageSize_.height;

        parallel_dispatch(config_.useParallelExec, [&]<typename Policy>(Policy&& policy) {
            std::transform(std::forward<Policy>(policy),
                           depthPtr,
                           depthPtr + numPixels,
                           resultPtr,
                           [&](const float& depth) -> ElementType {
                               int idx            = std::clamp(static_cast<int>(std::round(depth)),
                                                    0,
                                                    static_cast<int>(numImages_) - 1);
                               size_t pixelOffset = std::addressof(depth) - depthPtr;
                               //    return alignedImages_[idx].ptr<uchar>(0)[pixelOffset];
                               return alignedImages_[idx].ptr<ElementType>(0)[pixelOffset];
                           });
        });

        return result;
    }

    /**
     * @brief Generate all-in-focus image using weight-based fusion
     */
    template <typename ElementType>
    [[nodiscard]] cv::Mat generateAllInFocusWithFusion() const
    {
        int floatType = CV_32F;
        int intType   = CV_8U;
        if constexpr (std::is_same_v<ElementType, cv::Vec3b>) {
            floatType = CV_32FC3;
            intType   = CV_8UC3;
        }
        cv::Mat result    = cv::Mat::zeros(imageSize_, floatType);
        cv::Mat weightSum = cv::Mat::zeros(imageSize_, CV_32F);

        auto* resultPtr = result.ptr<ElementType>(0);
        // auto* resultPtrVec3b = result.ptr<cv::Vec3b>(0);
        auto* weightSumPtr = weightSum.ptr<float>(0);
        int numPixels      = imageSize_.width * imageSize_.height;

        auto indices = views::iota(0, numPixels);
        cv::Mat blurredFocus;
        cv::Size ksize(config_.fusionGaussKSize, config_.fusionGaussKSize);
        for (size_t i = 0; i < numImages_; ++i) {
            cv::GaussianBlur(focusVolume_[i], CVOutput(blurredFocus), ksize, 0);
            const auto* focusPtr = blurredFocus.ptr<float>(0);
            const auto* imgPtr   = alignedImages_[i].ptr<ElementType>(0);

            parallel_dispatch(config_.useParallelExec, [&]<typename Policy>(Policy&& policy) {
                std::for_each(
                    std::forward<Policy>(policy), indices.begin(), indices.end(), [=](int idx) {
                        float weight = focusPtr[idx];
                        auto value   = imgPtr[idx];
                        resultPtr[idx] += weight * value;
                        weightSumPtr[idx] += weight;
                    });
            });
        }

        // Normalize
        parallel_dispatch(config_.useParallelExec, [&]<typename Policy>(Policy&& policy) {
            std::for_each(
                std::forward<Policy>(policy), indices.begin(), indices.end(), [=](int idx) {
                    if (weightSumPtr[idx] > 1e-6f) { resultPtr[idx] /= weightSumPtr[idx]; }
                    else {
                        resultPtr[idx] = 0.0f;
                    }
                });
        });

        cv::Mat allInFocusInt;
        result.convertTo(allInFocusInt, intType);
        return allInFocusInt;
    }

    /**
     * @brief Inpaint low-confidence regions in depth map
     */
    void inpaintLowConfidenceRegions()
    {
        cv::Scalar meanConf;
        cv::Scalar stdDevConf;
        cv::meanStdDev(confidenceMap_, CVOutput(meanConf), CVOutput(stdDevConf));

        double confThresh = std::max(0.0, meanConf[0] - (0.5 * stdDevConf[0]));

        cv::Mat confMask;
        cv::threshold(confidenceMap_, confMask, confThresh, 255, cv::THRESH_BINARY);
        confMask.convertTo(confMask, CV_8U);

        cv::Mat unreliableMask = (confMask == 0);

        cv::inpaint(
            refinedDepthMap_, unreliableMask, CVOutput(refinedDepthMap_), 5, cv::INPAINT_TELEA);
    }

    /**
     * @brief Step 5: Convert depth index to physical depth
     */
    bool convertToPhysicalDepth()
    {
        Log.info("[STEP 5] Converting to physical depth");

        physicalDepthMap_ = cv::Mat(imageSize_, CV_32F);

        auto* depthPtr    = refinedDepthMap_.ptr<float>(0);
        auto* physicalPtr = physicalDepthMap_.ptr<float>(0);
        int numPixels     = imageSize_.width * imageSize_.height;

        parallel_dispatch(config_.useParallelExec, [&]<typename Policy>(Policy&& policy) {
            std::transform(std::forward<Policy>(policy),
                           depthPtr,
                           depthPtr + numPixels,
                           physicalPtr,
                           [&](float depthIndex) -> float {
                               if (depthIndex < 0.0f) { return physicalDistances_[0]; }
                               if (depthIndex >= static_cast<float>(numImages_) - 1) {
                                   return physicalDistances_.back();
                               }

                               int idx0 = static_cast<int>(std::floor(depthIndex));
                               int idx1 = idx0 + 1;
                               float t  = depthIndex - static_cast<float>(idx0);
                               return ((1.0F - t) * physicalDistances_[idx0])
                                      + (t * physicalDistances_[idx1]);
                           });
        });

        Log.info("Physical depth conversion completed");
        return true;
    }

    /**
     * @brief Save a cv::Mat to a text file
     */
    [[nodiscard]] static bool saveMatToTxt(const cv::Mat& mat, const fs::path& filepath)
    {
        std::ofstream file(filepath);
        if (!file.is_open()) {
            Log.error("Failed to open file for writing: {}", filepath.string());
            return false;
        }
#ifdef __cpp_lib_mdspan
        std::mdspan matView(mat.ptr<float>(0), mat.rows, mat.cols);

        for (size_t i = 0; i < matView.extent(0); ++i) {
            for (size_t j = 0; j < matView.extent(1); ++j) {
                file << matView[i, j];
                if (j < matView.extent(1) - 1) { file << " "; }
            }
            file << "\n";
        }
#else
        const auto* matPtr = mat.ptr<float>(0);
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                file << matPtr[i * mat.cols + j];
                if (j < mat.cols - 1) { file << " "; }
            }
            file << "\n";
        }
#endif // __cpp_lib_mdspan

        return true;
    }

    static cv::Mat generate_RDF_kernel(int inner, int outer)
    {
        int innerRadius = inner;
        int outerRadius = outer;
        int kernelSize  = (2 * outerRadius) + 1;
        int center      = outerRadius;

        cv::Mat kernel  = cv::Mat::zeros(kernelSize, kernelSize, CV_32F);
        auto* kernelPtr = kernel.ptr<float>(0);
        float sumValue  = 0.0f;

        for (int y = 0; y < kernelSize; ++y) {
            float yDist = 1.0f * (y - center) * (y - center);

            for (int x = 0; x < kernelSize; ++x) {
                float xDist = 1.0f * (x - center) * (x - center);
                float dist  = std::sqrt(xDist + yDist);
                if (dist <= outerRadius && dist > innerRadius) {
                    kernelPtr[y * kernelSize + x] = -1.0f;
                    sumValue += -1.0f;
                }
            }
        }

        kernelPtr[center * kernelSize + center] = -sumValue;
        return kernel;
    }
};

// ============================================================================
// Main Function
// ============================================================================
// TODO: parser 在 命令行和输入配置文件之间的优先级问题，依然很混乱
int main(int argc, char** argv)
{
    // === Configuration ===
    DFFConfig config;
    
    // Command-line overrides
    std::string configFile;
    std::string imageDirectory;
    std::string outputDirectory;
    
    // Create command-line parser
    MyCmdParser parser("final_project", 
        "Unified Depth from Focus (DFF) Pipeline - High-quality depth estimation from focus stacks");
    
    config.addCommandLineArguments(parser);

    // Parse arguments
    if (!parser.parse(argc, argv)) {
        return 0; // Help was shown or parse error
    }
    
    // Load config file if specified
    if (!configFile.empty()) {
        try {
            fs::path configPath = configFile;
            if (!configPath.is_absolute()) {
                configPath = fs::path(SOURCE_DIR) / "configs" / configFile;
            }
            
            YAML::Node node = YAML::LoadFile(configPath.string());
            DFFConfig::yamlToParser(node, parser);
            Log.info("Loaded configuration from: {}", configPath.string());
        }
        catch (const YAML::Exception& e) {
            Log.error("YAML parsing error in configuration file: {}", e.what());
            return -1;
        }
        catch (const std::exception& e) {
            Log.error("Error reading configuration file: {}", e.what());
            return -1;
        }
    }
    
    // Determine image directory
    fs::path imageDir;
    if (!imageDirectory.empty()) {
        imageDir = imageDirectory;
    } else {
        imageDir = fs::path(IMAGE_DIR) / "project_imgs";
    }
    
    // Determine output directory
    fs::path resultDir;
    if (!outputDirectory.empty()) {
        resultDir = outputDirectory;
    } else {
        resultDir = fs::path(PROJECT_ROOT) / "results" / "project" / config.outputSubDir;
    }

    config.overrideFromCmdParser(parser);

    // === Paths ===
    if (!fs::exists(imageDir)) {
        Log.error("Image directory not found: {}", imageDir.string());
        return -1;
    }
    // make sure result directory exists
    if (!fs::exists(resultDir)) { fs::create_directories(resultDir); }

    std::string configYaml = config.toYaml();
    Log.info("Configuration Details:\n{}", configYaml);
    std::ofstream configOut(resultDir / "config_used.yaml");
    configOut << configYaml;
    configOut.close();

    Log.info("=== Unified Depth from Focus Pipeline ===");

    // === Load Images ===
    Log.setLevel(LogLevel::DEBUG);
    Log.info("Loading images from: {}", imageDir.string());

    auto imagePaths
        = fs::directory_iterator(imageDir) | views::filter([](const auto& entry) {
              return entry.is_regular_file()
                     && (entry.path().extension() == ".bmp" || entry.path().extension() == ".jpg"
                         || entry.path().extension() == ".png");
          })
          | views::transform([](const auto& entry) { return entry.path().string(); })
          | rng::to<std::vector>();

    rng::sort(imagePaths);
    // imagePaths.resize(100);

    if (imagePaths.empty()) {
        Log.error("No images found in directory");
        return -1;
    }

    Log.info("Found {} {} images", imagePaths.size(), config.useColor ? "color" : "grayscale");
    int colorType = config.useColor ? cv::IMREAD_COLOR_BGR : cv::IMREAD_GRAYSCALE;

    std::vector<cv::Mat> images(imagePaths.size());
    parallel_dispatch(config.useParallelExec, [&](auto&& policy) {
        std::transform(
            std::forward<decltype(policy)>(policy),
            imagePaths.begin(),
            imagePaths.end(),
            images.begin(),
            // [](const std::string& path) { return cv::imread(path, cv::IMREAD_GRAYSCALE); });
            [colorType](const std::string& path) { return cv::imread(path, colorType); });
    });

    // === Process ===
    DepthFromFocusPipeline pipeline(config);

    if (!pipeline.process(images)) {
        Log.error("Pipeline processing failed!");
        return -1;
    }

    // === Save Results ===
    if (!pipeline.saveResults(resultDir)) {
        Log.error("Failed to save results!");
        return -1;
    }

    return 0;
}

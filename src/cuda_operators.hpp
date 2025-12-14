/**
 * @file cuda_operators.hpp
 * @brief CUDA-accelerated operations for Depth from Focus pipeline
 * @author wwf
 * @date 2025-12-14
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <span>
#include <vector>

// CUDA Runtime

/**
 * @brief Compute Sum of Modified Laplacian (SML) focus measure using CUDA
 *
 * This function accelerates the SML computation using GPU parallelization.
 * SML is a focus measure that applies modified Laplacian filtering followed
 * by local summation to detect image sharpness.
 *
 * @param alignedStack Input image stack (aligned grayscale or color images)
 * @param kSMLWindowSize Window size for local summation (typically 5-11)
 * @return std::vector<cv::Mat> Focus measure maps for each input image
 *
 * @note All input images must have the same size and type
 * @note CUDA must be available at compile time (USE_CUDA defined)
 */
auto sml_cuda(std::span<const cv::Mat> alignedStack, int kSMLWindowSize) -> std::vector<cv::Mat>;

/**
 * @brief Compute Ring Difference Filter (RDF) focus measure using CUDA
 *
 * RDF is an alternative focus measure that uses a ring-shaped kernel to
 * detect edges and texture. This CUDA implementation provides accelerated
 * computation for large image stacks.
 *
 * @param alignedStack Input image stack (aligned grayscale or color images)
 * @param rdfKernel Pre-computed RDF kernel (ring-shaped filter)
 * @return std::vector<cv::Mat> Focus measure maps for each input image
 *
 * @note The RDF kernel should be generated using generateRDFKernel()
 * @note CUDA must be available at compile time (USE_CUDA defined)
 */
std::vector<cv::Mat> rdf_cuda(std::span<const cv::Mat> alignedStack, const cv::Mat& rdfKernel);

/**
 * @brief Align image stack using Enhanced Correlation Coefficient (ECC) on CUDA
 *
 * This function aligns a stack of images to a reference image using the ECC
 * maximization algorithm, accelerated with CUDA for improved performance.
 *
 * @param inputImages Input image stack to be aligned
 * @param refIndex Index of the reference image in the stack
 * @param outputImages [out] Aligned image stack (must be pre-allocated)
 *
 * @note The reference image at refIndex will be copied unchanged to outputImages
 * @note All images must have the same size
 * @note CUDA must be available at compile time (USE_CUDA defined)
 */
void ecc_align_cuda(std::span<const cv::Mat> inputImages,
                    size_t refIndex,
                    std::vector<cv::Mat>& outputImages);
#include <filesystem>
#include <iostream>
#include <opencv2/core/mat.hpp>
// #include <opencv2/opencv.hpp>
#include "opencv2/opencv.hpp"

int main(int argc, char** argv)
{
    std::cout << "C++23 OpenCV Test Program\n";
    auto imgPath = std::filesystem::path(IMAGE_DIR) / "test_opencv.jpg";
    // // std::cout << "hello world!" << std::endl;
    auto img = cv::imread(imgPath.string());
    // cv::Mat img = cv::Mat::zeros(200, 200, CV_8UC3);

    if (img.empty()) {
        std::cerr << "Failed to load image at: " << imgPath << '\n';
        return -1;
    }
    std::cout << "Image loaded successfully from: " << imgPath << '\n';
    std::cout << "Image size: " << img.cols << " x " << img.rows << '\n';

    // cv::imshow("Test OpenCV Image", img);
    // cv::waitKey(0);
    return 0;
}

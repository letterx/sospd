#include <iostream>
#include <string>
#include <cassert>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cout << "Usage: disparity-diff ground-truth stereo-result\n";
        exit(-1);
    }

    cv::Mat gt = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat stereo_result = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

    assert(gt.type() == CV_8UC1);
    assert(stereo_result.type() == CV_8UC1);
    assert(gt.rows == stereo_result.rows);
    assert(gt.cols == stereo_result.cols);

    // Counts for identical pixels, and within plus/minus 1 and 4
    int same = 0;
    int pm_1 = 0;
    int pm_4 = 0;
    int sum_squared_diff = 0;
    for (int i = 0; i < gt.rows; ++i) {
        for (int j = 0; j < gt.cols; ++j) {
            int d1 = gt.at<unsigned char>(i, j);
            int d2 = stereo_result.at<unsigned char>(i, j);
            int diff = d2 - d1;
            diff = abs(diff);
            if (diff == 0) same++;
            if (diff <= 1) pm_1++;
            if (diff <= 4) pm_4++;
            sum_squared_diff += diff*diff;
        }
    }
    auto numPixels = gt.rows * gt.cols;
    std::cout << "Total pixels:     " << numPixels << "\n";
    std::cout << "Same label:       " << same << "\n";
    std::cout << "Percent same:     " << double(same)/double(numPixels) << "\n";
    std::cout << "Plus/minus 1:     " << pm_1 << "\n";
    std::cout << "Percent in p/m 4: " << double(pm_1)/double(numPixels) << "\n";
    std::cout << "Plus/minus 4:     " << pm_4 << "\n";
    std::cout << "Percent in p/m 4: " << double(pm_4)/double(numPixels) << "\n";
    std::cout << "Sum squared diff: " << sum_squared_diff << "\n";



    return 0;
}

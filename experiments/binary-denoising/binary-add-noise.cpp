#include <iostream>
#include <string>
#include <stdlib.h>
#include <assert.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << "Usage: add-noise infile outfile percent\n";
        exit(-1);
    }

    cv::Mat im;
    cv::Mat gray;
    im = cv::imread(argv[1]);
    assert(im.channels() == 3);
    cvtColor(im, gray, CV_RGB2GRAY);
    float percent = atof(argv[3]);
    cv::Mat noise;
    noise.create(im.rows, im.cols, CV_32FC1);
    cv::randu(noise, 0, 100);
    cv::Point p;
    for (p.y = 0; p.y < im.rows; ++p.y) {
        for (p.x = 0; p.x < im.cols; ++p.x) {
            if (noise.at<float>(p) < percent)
                gray.at<unsigned char>(p) = 255 - gray.at<unsigned char>(p);
        }
    }

    cv::imwrite(argv[2], gray);

    return 0;
}

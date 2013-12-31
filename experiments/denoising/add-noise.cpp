#include <iostream>
#include <string>
#include <stdlib.h>
#include <assert.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << "Usage: add-noise infile outfile sigma\n";
        exit(-1);
    }

    cv::Mat im;
    cv::Mat out;
    cv::Mat gray;
    im = cv::imread(argv[1]);
    assert(im.channels() == 3);
    cvtColor(im, gray, CV_RGB2GRAY);
    double sigma = atof(argv[3]);
    cv::Mat noise;
    noise.create(im.rows, im.cols, CV_32FC1);
    cv::randn(noise, 0, sigma);

    gray.convertTo(out, CV_32FC1);
    out = out + noise;
    cv::imwrite(argv[2], out);



    return 0;
}

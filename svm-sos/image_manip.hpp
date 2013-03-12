#ifndef _IMAGE_MANIP_HPP_
#define _IMAGE_MANIP_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void ValidateExample(const cv::Mat& im, const cv::Mat& tri, const cv::Mat& gt) {
    ASSERT(im.data != NULL);
    ASSERT(tri.data != NULL);
    ASSERT(gt.data != NULL);
}


#endif

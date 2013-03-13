#ifndef _IMAGE_MANIP_HPP_
#define _IMAGE_MANIP_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/serialization/split_free.hpp>

inline void ValidateExample(const cv::Mat& im, const cv::Mat& tri, const cv::Mat& gt) {
    ASSERT(im.data != NULL);
    ASSERT(im.depth() == CV_8U);
    ASSERT(im.channels() == 3);

    ASSERT(tri.data != NULL);
    ASSERT(tri.depth() == CV_8U);
    ASSERT(im.channels() == 3);

    ASSERT(gt.data != NULL);
    ASSERT(gt.depth() == CV_8U);
    ASSERT(im.channels() == 3);
}

inline void ConvertToMask(const cv::Mat& trimap, cv::Mat& out) {
    out.create(trimap.rows, trimap.cols, CV_8UC1);
    cv::MatIterator_<unsigned char> out_it, out_end;
    cv::MatConstIterator_<cv::Vec3b> in_it, in_end;
    in_it = trimap.begin<cv::Vec3b>();
    in_end = trimap.end<cv::Vec3b>();
    out_it = out.begin<unsigned char>();
    out_end = out.end<unsigned char>();
    for (; in_it != in_end; ++in_it, ++out_it) {
        if (*in_it == cv::Vec3b(0xdb, 0x0, 0x0))
            *out_it = cv::GC_BGD;
        else if (*in_it == cv::Vec3b(0xff, 0xff, 0xcf))
            *out_it = cv::GC_FGD;
        else
            *out_it = cv::GC_PR_BGD;
    }
}

inline void ConvertGreyToMask(const cv::Mat& gt, cv::Mat& out) {
    out.create(gt.rows, gt.cols, CV_8UC1);
    cv::MatIterator_<uchar> out_it, out_end;
    cv::MatConstIterator_<uchar> in_it, in_end;
    in_it = gt.begin<uchar>();
    in_end = gt.end<uchar>();
    out_it = out.begin<uchar>();
    out_end = out.end<uchar>();
    for (; in_it != in_end; ++in_it, ++out_it) {
        if (*in_it == 0x0)
            *out_it = cv::GC_BGD;
        else if (*in_it == 0xff)
            *out_it = cv::GC_FGD;
        else
            *out_it = cv::GC_PR_BGD;
    }

}

namespace boost {
namespace serialization {
template <typename Archive>
void save(Archive& ar, const cv::Mat& im, const unsigned int version) {
    ar << im;
}

template <typename Archive>
void load(Archive& ar, cv::Mat& im, const unsigned int version) {
    ar >> im;
}

template <typename Archive>
void serialize(Archive& ar, cv::Mat& im, const unsigned int version) {
    split_free(ar, im, version);
}

}
}


#endif

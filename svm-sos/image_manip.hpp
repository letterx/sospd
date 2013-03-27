#ifndef _IMAGE_MANIP_HPP_
#define _IMAGE_MANIP_HPP_

#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/serialization/split_free.hpp>

template <typename Fn>
inline void ImageIterate(cv::Mat& im, Fn f) {
    cv::MatIterator_<unsigned char> iter, end;
    for (iter = im.begin<unsigned char>(), end = im.end<unsigned char>();
            iter != end; ++iter) {
        f(*iter);
    }
}

template <typename Fn>
inline void ImageCIterate(const cv::Mat& im, Fn f) {
    cv::MatConstIterator_<unsigned char> iter, end;
    for (iter = im.begin<unsigned char>(), end = im.end<unsigned char>();
            iter != end; ++iter) {
        f(*iter);
    }
}

template <typename Fn>
inline void ImageIterate(cv::Mat& im1, cv::Mat& im2, Fn f) {
    cv::MatIterator_<unsigned char> it1, end1;
    cv::MatIterator_<unsigned char> it2, end2;
    it1 = im1.begin<unsigned char>();
    end1 = im1.end<unsigned char>();
    it2 = im2.begin<unsigned char>();
    end2 = im2.end<unsigned char>();
    for (; it1 != end1; ++it1, ++it2) {
        f(*it1, *it2);
    }
    ASSERT(it2 == end2);
}

template <typename Fn>
inline void ImageCIterate(const cv::Mat& im1, const cv::Mat& im2, Fn f) {
    cv::MatConstIterator_<unsigned char> it1, end1;
    cv::MatConstIterator_<unsigned char> it2, end2;
    it1 = im1.begin<unsigned char>();
    end1 = im1.end<unsigned char>();
    it2 = im2.begin<unsigned char>();
    end2 = im2.end<unsigned char>();
    for (; it1 != end1; ++it1, ++it2) {
        f(*it1, *it2);
    }
    ASSERT(it2 == end2);
}

template <typename Fn>
inline void ImageCIterate3_1(const cv::Mat& im1, const cv::Mat& im2, Fn f) {
    cv::MatConstIterator_<cv::Vec3b> it1, end1;
    cv::MatConstIterator_<unsigned char> it2, end2;
    it1 = im1.begin<cv::Vec3b>();
    end1 = im1.end<cv::Vec3b>();
    it2 = im2.begin<unsigned char>();
    end2 = im2.end<unsigned char>();
    for (; it1 != end1; ++it1, ++it2) {
        f(*it1, *it2);
    }
    ASSERT(it2 == end2);
}

template <typename Fn>
inline void ImageIterate(const cv::Mat& im, const cv::Point& offset, Fn f) {
    ASSERT(offset.x >= 0 && offset.y >= 0);
    cv::Point p;
    for (p.y = 0; p.y + offset.y < im.rows; ++p.y) {
        for (p.x = 0; p.x + offset.x < im.cols; ++p.x) {
            f(im.at<unsigned char>(p), im.at<unsigned char>(p+offset));
        }
    }
}

template <typename Arg1, typename Arg2>
inline void ImageIterate(const cv::Mat& im, cv::Mat& out, const cv::Point& offset, 
        const std::function<void(const Arg1&, const Arg1&, Arg2&, Arg2&)>& f) {
    ASSERT(offset.x >= 0 && offset.y >= 0);
    cv::Point p;
    for (p.y = 0; p.y + offset.y < im.rows; ++p.y) {
        for (p.x = 0; p.x + offset.x < im.cols; ++p.x) {
            f(im.at<Arg1>(p), im.at<Arg1>(p+offset), out.at<Arg2>(p), out.at<Arg2>(p+offset));
        }
    }
}

template <typename Arg>
inline void ImageIteratePatch(const cv::Mat& im, const cv::Point& offset,
        const std::function<void(const std::vector<Arg>&)> fn) {
    ASSERT(offset.x >= 0 && offset.y >= 0);
    cv::Point base;
    cv::Point p;
    std::vector<Arg> args;
    args.reserve(offset.x*offset.y);
    for (base.y = 0; base.y + offset.y < im.rows; ++base.y) {
        for (base.x = 0; base.x + offset.x < im.cols; ++base.x) {
            args.clear();
            for (p = base; p.y <= base.y + offset.y; ++p.y) {
                for (p.x = base.x; p.x <= base.x + offset.x; ++p.x) {
                    args.push_back(im.at<Arg>(p));
                }
            }
            fn(args);
        }
    }
}

template <typename Fn>
inline void ImageIterpPatch(const cv::Mat& im, const cv::Point& offset, Fn f) {
    ASSERT(offset.x >= 0 && offset.y >= 0);
    cv::Point base;
    cv::Point p;
    std::vector<cv::Point> args;
    args.reserve(offset.x*offset.y);
    for (base.y = 0; base.y + offset.y < im.rows; ++base.y) {
        for (base.x = 0; base.x + offset.x < im.cols; ++base.x) {
            args.clear();
            for (p = base; p.y <= base.y + offset.y; ++p.y) {
                for (p.x = base.x; p.x <= base.x + offset.x; ++p.x) {
                    args.push_back(p);
                }
            }
            f(args);
        }
    }
}

template <typename Arg>
inline void ImageIteriPatch(const cv::Mat& im, const cv::Point& offset, 
        const std::function<void(const std::vector<Arg>&)>& f) {
    ASSERT(offset.x >= 0 && offset.y >= 0);
    cv::Point base;
    cv::Point p;
    std::vector<Arg> args;
    args.reserve(offset.x*offset.y);
    for (base.y = 0; base.y + offset.y < im.rows; ++base.y) {
        for (base.x = 0; base.x + offset.x < im.cols; ++base.x) {
            args.clear();
            for (p = base; p.y <= base.y + offset.y; ++p.y) {
                for (p.x = base.x; p.x <= base.x + offset.x; ++p.x) {
                    args.push_back(p.y*im.cols + p.x);
                }
            }
            f(args);
        }
    }
}

template <typename Arg1, typename Arg2>
inline void ImageIterate(const cv::Mat& im, const cv::Mat& out, const cv::Point& offset, 
        const std::function<void(const Arg1&, const Arg1&, const Arg2&, const Arg2&)>& f) {
    ASSERT(offset.x >= 0 && offset.y >= 0);
    cv::Point p;
    for (p.y = 0; p.y + offset.y < im.rows; ++p.y) {
        for (p.x = 0; p.x + offset.x < im.cols; ++p.x) {
            f(im.at<Arg1>(p), im.at<Arg1>(p+offset), out.at<Arg2>(p), out.at<Arg2>(p+offset));
        }
    }
}

template <typename Fn>
inline void ImageIteri(const cv::Mat& im, const cv::Point& offset, Fn f) {
    ASSERT(offset.x >= 0 && offset.y >= 0);
    cv::Point p;
    for (p.y = 0; p.y + offset.y < im.rows; ++p.y) {
        for (p.x = 0; p.x + offset.x < im.cols; ++p.x) {
            f(p.y*im.cols + p.x, (p.y + offset.y)*im.cols + p.x + offset.x);
        }
    }
}

template <typename Fn>
inline void ImageIterp(const cv::Mat& im, const cv::Point& offset, Fn f) {
    ASSERT(offset.x >= 0 && offset.y >= 0);
    cv::Point p;
    for (p.y = 0; p.y + offset.y < im.rows; ++p.y) {
        for (p.x = 0; p.x + offset.x < im.cols; ++p.x) {
            f(p, p+offset);
        }
    }
}



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

    ASSERT(im.rows == tri.rows);
    ASSERT(im.rows == gt.rows);
    ASSERT(im.cols == tri.cols);
    ASSERT(im.cols == gt.cols);
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
        if (*in_it == cv::Vec3b(0x00, 0x0, 0xdb))
            *out_it = cv::GC_BGD;
        else if (*in_it == cv::Vec3b(0xcf, 0xff, 0xff))
            *out_it = cv::GC_FGD;
        else if (*in_it == cv::Vec3b(0x0, 0x0, 0x0))
            *out_it = cv::GC_PR_BGD;
        else {
            std::cout << "Unknown color: " << *in_it << "\n";
            exit(1);
        }

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

inline cv::Mat MaskToColor(const cv::Mat& mask, const cv::Mat& image) {
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, CV_BGR2GRAY);
    ASSERT(image.depth() == CV_8U);
    ASSERT(gray_image.depth() == CV_8U);
    cv::Mat out_image(gray_image.rows, gray_image.cols, CV_64FC3);
    cv::Point p;
    for (p.y = 0; p.y < gray_image.rows; ++p.y) {
        for (p.x = 0; p.x < gray_image.cols; ++p.x) {
            double intensity = ((double)gray_image.at<unsigned char>(p))/255.0;
            unsigned char label = mask.at<unsigned char>(p);
            cv::Vec3d color;
            if (label == cv::GC_BGD || label == cv::GC_PR_BGD) {
                color = cv::Vec3d(1.0, 0.0, 0.0)*intensity;
            } else if (label == cv::GC_FGD || label == cv::GC_PR_FGD) {
                color = cv::Vec3d(0.0, 0.0, 1.0)*intensity;
            } else {
                ASSERT(false);
            }
            out_image.at<cv::Vec3d>(p) = color;
        }
    }
    return out_image;
}

inline void ShowImage(const cv::Mat& im) {
    cv::namedWindow("Display window", CV_WINDOW_AUTOSIZE);
    if (im.depth() == CV_8U)
        cv::imshow("Display window", im*(255/3));
    else
        cv::imshow("Display window", im);
    cv::waitKey(0);
}

namespace boost {
namespace serialization {
template <typename Archive>
void save(Archive& ar, const cv::Mat& im, const unsigned int version) {
    ar & im.rows;
    ar & im.cols;
    int channels = im.channels();
    ar & channels;
    unsigned char* data = im.data;
    for (int i = 0; i < im.rows*im.cols*im.channels(); ++i) {
        ar & data[i];
    }
}

template <typename Archive>
void load(Archive& ar, cv::Mat& im, const unsigned int version) {
    int rows, cols, channels;
    ar & rows;
    ar & cols;
    ar & channels;
    im.create(rows, cols*channels, CV_8UC1);
    unsigned char* data = im.data;
    for (int i = 0; i < rows*cols*channels; ++i) {
        ar & data[i];
    }
}

template <typename Archive>
void serialize(Archive& ar, cv::Mat& im, const unsigned int version) {
    split_free(ar, im, version);
}

}
}


#endif

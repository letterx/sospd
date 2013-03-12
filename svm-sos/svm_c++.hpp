#ifndef _SVM_CXX_HPP_
#define _SVM_CXX_HPP_

extern "C" {
#include "svm_light/svm_common.h"
#include "svm_struct/svm_struct_common.h"
}
#include <vector>
#include "submodular-flow.hpp"

class PatternData {
    public:
        PatternData(const cv::Mat& im, const cv::Mat& tri)
            : m_image(im), m_trimap(tri) { }
        cv::Mat m_image;
        cv::Mat m_trimap;
};


class LabelData {
    public:
        LabelData(const std::vector<int>& gt)
            : m_gt(gt) { }
        std::vector<int> m_gt;

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, const unsigned int version) {
            ar & m_gt;
        }
};


typedef SubmodularFlow CRF;

template <typename PatternData, typename LabelData, typename CRF>
class FeatureGroup {
    public:
        virtual const size_t NumFeatures() const = 0;
        virtual std::vector<FVAL> Psi(const PatternData& p, const LabelData& l) const = 0;
        virtual void AddToCRF(CRF& c, const PatternData& p) const = 0;
};

typedef FeatureGroup<PatternData, LabelData, CRF> FG;

class ModelData {
    public:
        long NumFeatures() const {
            long n = 0;
            for (auto fgp : m_features)
                n += fgp->NumFeatures();
            return n;
        }
        std::vector<std::shared_ptr<FG>> m_features;
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, const unsigned int version) {
            // FIXME: Write out features???
        }
};



#endif


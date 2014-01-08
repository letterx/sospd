#ifndef _BINARY_UNARY_FEATURE_HPP_
#define _BINARY_UNARY_FEATURE_HPP_

#include "feature.hpp"
#include "binary_seg_app.hpp"
#include <map>
#include <fstream>
#include <boost/serialization/export.hpp>

class BinaryUnaryFeature : public AppTraits<BinarySegApp>::FG {
    public:
    double m_scale;
    BinaryUnaryFeature() : m_scale(1.0) { }
    explicit BinaryUnaryFeature(double scale) : m_scale(10.0*scale) { }

    virtual size_t NumFeatures() const { return 1; }
    virtual std::vector<FVAL> Psi(const BS_PatternData& p, const BS_LabelData& l) const {
        std::vector<FVAL> psi = {0.0};
        cv::Point pt;
        for (pt.y = 0; pt.y < p.m_image.rows; ++pt.y) {
            for (pt.x = 0; pt.x < p.m_image.cols; ++pt.x) {
                int label = l.m_gt.at<unsigned char>(pt);
                int value = p.m_image.at<unsigned char>(pt);
                if (label == 255)
                    psi[0] += m_scale*(255-value);
                else if (label == 0)
                    psi[0] += m_scale*value;
                else
                    ASSERT(false);
            }
        }
        for (auto& v : psi)
            v = -v;
        return psi;
    }
    virtual void AddToCRF(CRF& crf, const BS_PatternData& p, double* w) const {
        cv::Point pt;
        for (pt.y = 0; pt.y < p.m_image.rows; ++pt.y) {
            for (pt.x = 0; pt.x < p.m_image.cols; ++pt.x) {
                CRF::NodeId id = pt.y * p.m_image.cols + pt.x;
                int value = p.m_image.at<unsigned char>(pt);
                double E0 = w[0]*m_scale*value;
                double E1 = w[0]*m_scale*(255-value);
                crf.AddUnaryTerm(id, doubleToREAL(E0), doubleToREAL(E1));
            }
        }
    }

    private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        //std::cout << "Serializing BinaryUnaryFeature\n";
        ar & boost::serialization::base_object<FeatureGroup>(*this);
        ar & m_scale;
    }

};

BOOST_CLASS_EXPORT_GUID(BinaryUnaryFeature, "BinaryUnaryFeature")

#endif

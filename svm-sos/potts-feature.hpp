#ifndef _POTTS_FEATURE_HPP_
#define _POTTS_FEATURE_HPP_

#include "semantic_seg_app.hpp"
#include "feature.hpp"
#include <opencv2/core/core.hpp>

class PottsFeature : public SemanticSegApp::FG {
    public: 
    double m_scale;
    int m_num_labels;

    PottsFeature() : m_scale(1.0) { }
    explicit PottsFeature(double scale, int num_labels) : m_scale(scale), m_num_labels(num_labels) { }

    virtual size_t NumFeatures() const override { return 2; }
    virtual std::vector<FVAL> Psi(const Sem_PatternData& p, const Sem_LabelData& l) const override {
        std::vector<FVAL> psi(NumFeatures(), 0);
        cv::Point base, pt;
        const cv::Point patch_size(1.0, 1.0);
        for (base.y = 0; base.y + patch_size.y < p.m_image.rows; ++base.y) {
            for (base.x = 0; base.x + patch_size.x < p.m_image.cols; ++base.x) {
                int first_label = l.m_gt.at<int>(base);
                bool all_equal = true;
                for (pt.y = base.y; pt.y <= base.y + patch_size.y; ++pt.y) {
                    for (pt.x = base.x; pt.x <= base.x + patch_size.x; ++pt.x) {
                        int label = l.m_gt.at<int>(pt);
                        if (label != first_label) all_equal = false;
                    }
                }
                if (all_equal) psi[0] += m_scale;
                else psi[1] += m_scale;
            }
        }
        for (auto& v : psi)
            v = -v;
        return psi;
    }
    virtual void AddToCRF(MultiLabelCRF& crf, const Sem_PatternData& p, double* w) const override {
        cv::Point base, pt;
        const cv::Point patch_size(1.0, 1.0);
        for (base.y = 0; base.y + patch_size.y < p.m_image.rows; ++base.y) {
            for (base.x = 0; base.x + patch_size.x < p.m_image.cols; ++base.x) {
                std::vector<MultiLabelCRF::NodeId> vars;
                for (pt.y = base.y; pt.y <= base.y + patch_size.y; ++pt.y) {
                    for (pt.x = base.x; pt.x <= base.x + patch_size.x; ++pt.x) {
                        vars.push_back(pt.y*p.m_image.cols + pt.x);
                    }
                }
                MultiLabelCRF::CliquePtr cp(new PottsClique(vars, doubleToREAL(w[0]*m_scale), doubleToREAL(w[1]*m_scale)));
                crf.AddClique(cp);
            }
        }
    }
    private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) { 
        //std::cout << "Serializing PottsFeature\n";
        ar & boost::serialization::base_object<FeatureGroup>(*this);
        ar & m_scale;
        ar & m_num_labels;
    }
};

BOOST_CLASS_EXPORT_GUID(PottsFeature, "PottsFeature")

#endif

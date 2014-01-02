#ifndef _ALE_FEATURE_HPP_
#define _ALE_FEATURE_HPP_

#include "semantic_seg_app.hpp"
#include "feature.hpp"
#include <opencv2/core/core.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/map.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

class ALE_Feature : public SemanticSegApp::FG {
    public: 
    double m_scale;
    int m_num_labels;

    ALE_Feature() : m_scale(1.0) { }
    explicit ALE_Feature(double scale, int num_labels) : m_scale(scale), m_num_labels(num_labels) { }

    virtual size_t NumFeatures() const override { return 1; }
    virtual std::vector<FVAL> Psi(const Sem_PatternData& p, const Sem_LabelData& l) const override {
        const std::vector<cv::Mat>& ale_feature = m_ale_feature[p.Name()];
        for (int i = 0; i < m_num_labels; ++i) {
            ASSERT(ale_feature[i].rows == p.m_image.rows);
            ASSERT(ale_feature[i].cols == p.m_image.cols);
        }
        std::vector<FVAL> psi(NumFeatures(), 0);
        cv::Point pt;
        for (pt.y = 0; pt.y < p.m_image.rows; ++pt.y) {
            for (pt.x = 0; pt.x < p.m_image.cols; ++pt.x) {
                int label = l.m_gt.at<int>(pt);
                double cost = ale_feature[label].at<float>(pt);
                psi[0] += cost*m_scale;
            }
        }
        for (auto& v : psi)
            v = -v;
        return psi;
    }
    virtual void AddToCRF(MultiLabelCRF& crf, const Sem_PatternData& p, double* w) const override {
        const std::vector<cv::Mat>& ale_feature = m_ale_feature[p.Name()];
        for (int i = 0; i < m_num_labels; ++i) {
            ASSERT(ale_feature[i].rows == p.m_image.rows);
            ASSERT(ale_feature[i].cols == p.m_image.cols);
        }
        cv::Point pt;
        for (pt.y = 0; pt.y < p.m_image.rows; ++pt.y) {
            for (pt.x = 0; pt.x < p.m_image.cols; ++pt.x) {
                std::vector<REAL> cost_table(m_num_labels);
                for (int label = 0; label < m_num_labels; ++label)
                    cost_table[label] = doubleToREAL(w[0]*m_scale*ale_feature[label].at<float>(pt));
                MultiLabelCRF::NodeId id = pt.y*p.m_image.cols + pt.x;
                crf.AddUnaryTerm(id, cost_table);
            }
        }
    }
    virtual void LoadEvaluation(const std::string& output_dir) {
        std::string infile = output_dir + "/ale-feature.dat";
        std::ifstream is(infile, std::ios_base::binary);
        boost::archive::binary_iarchive ar(is);
        ar & m_ale_feature;
    }
    private:
    static constexpr size_t filters = 5;

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) { 
        //std::cout << "Serializing ALE_Feature\n";
        ar & boost::serialization::base_object<FeatureGroup>(*this);
        ar & m_scale;
        ar & m_num_labels;
    }
    mutable std::map<std::string, std::vector<cv::Mat>> m_ale_feature;
    std::mt19937 gen;
};

BOOST_CLASS_EXPORT_GUID(ALE_Feature, "ALE_Feature")

#endif

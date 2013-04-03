#ifndef _CLUSTER_COLOR_FEATURE_HPP_
#define _CLUSTER_COLOR_FEATURE_HPP_

#include "semantic_seg_app.hpp"
#include "feature.hpp"
#include <opencv2/core/core.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/map.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

class ClusterColorFeature : public SemanticSegApp::FG {
    public: 
    typedef SemanticSegApp::FG::Constr Constr;

    static constexpr size_t num_clusters = 50;
    double m_scale;
    int m_num_labels;

    ClusterColorFeature() : m_scale(1.0) { }
    explicit ClusterColorFeature(double scale, int num_labels) : m_scale(scale), m_num_labels(num_labels) { }

    virtual size_t NumFeatures() const override { return num_clusters*m_num_labels; }
    virtual std::vector<FVAL> Psi(const Sem_PatternData& p, const Sem_LabelData& l) const override {
        const cv::Mat& color_feature = m_color_feature[p.Name()];
        std::vector<FVAL> psi(NumFeatures(), 0);
        cv::Point pt;
        for (pt.y = 0; pt.y < p.m_image.rows; ++pt.y) {
            for (pt.x = 0; pt.x < p.m_image.cols; ++pt.x) {
                int feature = color_feature.at<int>(pt);
                int label = l.m_gt.at<int>(pt);
                psi[feature*m_num_labels + label] += m_scale;
            }
        }
        for (auto& v : psi)
            v = -v;
        return psi;
    }
    virtual void AddToCRF(MultiLabelCRF& crf, const Sem_PatternData& p, double* w) const override {
        const cv::Mat& color_feature = m_color_feature[p.Name()];
        std::vector<std::vector<REAL>> costTables(num_clusters, std::vector<REAL>(m_num_labels));
        for (size_t i = 0; i < num_clusters; ++i) {
            for (int j = 0; j < m_num_labels; ++j) {
                costTables[i][j] = doubleToREAL(m_scale*w[i*m_num_labels+j]);
            }
        }
        cv::Point pt;
        for (pt.y = 0; pt.y < p.m_image.rows; ++pt.y) {
            for (pt.x = 0; pt.x < p.m_image.cols; ++pt.x) {
                int feature = color_feature.at<int>(pt);
                MultiLabelCRF::NodeId id = pt.y*p.m_image.cols + pt.x;
                crf.AddUnaryTerm(id, costTables[feature]);
            }
        }
    }
    virtual void Train(const std::vector<Sem_PatternData*>& patterns, const std::vector<Sem_LabelData*>& labels) {
        cv::Mat samples;
        std::cout << "Training color features -- "; 
        std::cout.flush();
        samples.create(0, 3, CV_32FC1);
        for (const Sem_PatternData* xp : patterns) {
            const cv::Mat& im = xp->m_image;
            cv::Mat response;
            GetResponse(im, response);
            samples.push_back(response);
        }
        std::cout << samples.rows << " samples...";
        std::cout.flush();
        cv::Mat best_labels;
        cv::kmeans(samples, num_clusters, best_labels, cv::TermCriteria(CV_TERMCRIT_EPS, 10, 0.2), 1, cv::KMEANS_RANDOM_CENTERS, m_centers);
        std::cout << "Done!\n";
    }
    virtual void Evaluate(const std::vector<Sem_PatternData*>& patterns) override {
        std::cout << "Evaluating Contrast Submodular Features...";
        std::cout.flush();
        for (const Sem_PatternData* xp : patterns) {
            const cv::Mat& im = xp->m_image;
            cv::Mat response;
            cv::Mat& features = m_color_feature[xp->Name()];
            GetResponse(im, response);
            features.create(im.rows, im.cols, CV_32SC1);
            Classify(response, features, m_centers);
        }
        std::cout << "Done!\n";
    }
    virtual void SaveEvaluation(const std::string& output_dir) const {
        std::string outfile = output_dir + "/contrast-submodular-feature.dat";
        std::ofstream os(outfile, std::ios_base::binary);
        boost::archive::binary_oarchive ar(os);
        ar & m_color_feature;
    }
    virtual void LoadEvaluation(const std::string& output_dir) {
        std::string infile = output_dir + "/contrast-submodular-feature.dat";
        std::ifstream is(infile, std::ios_base::binary);
        boost::archive::binary_iarchive ar(is);
        ar & m_color_feature;
    }
    private:
    static constexpr size_t filters = 5;
    void GetResponse(const cv::Mat& im, cv::Mat& response) {
        im.convertTo(response, CV_32F);
        response *= 1./255;
        response = response.reshape(1, im.rows*im.cols);
    }
    void Classify(const cv::Mat& response, cv::Mat& out, const cv::Mat& centers) {
        ASSERT((size_t)response.cols == 3);
        ASSERT((size_t)centers.cols == 3);
        size_t i = 0;
        cv::Point p;
        for (p.y = 0; p.y < out.rows; ++p.y) {
            for (p.x = 0; p.x < out.cols; ++p.x) {
                cv::Mat row = response.row(i);
                double min_dist = std::numeric_limits<double>::max();
                int min_center = 0;
                for (int j = 0; j < centers.rows; ++j) {
                    cv::Mat center = centers.row(j);
                    double dist = cv::norm(row, center);
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_center = j;
                    }
                }
                ASSERT(min_dist < std::numeric_limits<double>::max());
                out.at<int>(p) = min_center;
                i++;
            }
        }
    }
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) { 
        //std::cout << "Serializing ClusterColorFeature\n";
        ar & boost::serialization::base_object<FeatureGroup>(*this);
        ar & m_scale;
    }
    mutable std::map<std::string, cv::Mat> m_color_feature;
    std::mt19937 gen;
    cv::Mat m_centers;
};

BOOST_CLASS_EXPORT_GUID(ClusterColorFeature, "ClusterColorFeature")

#endif

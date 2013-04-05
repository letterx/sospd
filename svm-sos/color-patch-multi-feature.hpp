#ifndef _COLOR_PATCH_MULTI_FEATURE_HPP_
#define _COLOR_PATCH_MULTI_FEATURE_HPP_

#include "interactive_seg_app.hpp"
#include "feature.hpp"
#include <opencv2/core/core.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <unordered_set>

class ColorPatchMultiFeature : public SemanticSegApp::FG {
    public: 
    static constexpr int num_clusters = 50;
    static constexpr int num_filters = 27;
    static constexpr size_t samples_per_image = 5000;
    double m_scale;
    int m_num_labels;

    ColorPatchMultiFeature() : m_scale(1.0) { }
    explicit ColorPatchMultiFeature(double scale, int num_labels) : m_scale(scale), m_num_labels(num_labels) { }

    virtual size_t NumFeatures() const override { return num_clusters*m_num_labels; }
    virtual std::vector<FVAL> Psi(const Sem_PatternData& p, const Sem_LabelData& l) const override {
        cv::Mat patch_feature = m_patch_feature[p.Name()];
        std::vector<FVAL> psi(NumFeatures(), 0);
        cv::Point pt;
        for (pt.y = 0; pt.y < p.m_image.rows; ++pt.y) {
            for (pt.x = 0; pt.x < p.m_image.cols; ++pt.x) {
                int feature = patch_feature.at<int>(pt);
                ASSERT(feature < num_clusters);
                int label = l.m_gt.at<int>(pt);
                psi[feature*m_num_labels + label] += m_scale;
            }
        }
        for (auto& v : psi)
            v = -v;
        return psi;
    }
    virtual void AddToCRF(MultiLabelCRF& crf, const Sem_PatternData& p, double* w) const override {
        std::vector<std::vector<REAL>> costTables(num_clusters, std::vector<REAL>(m_num_labels, 0));
        for (int c = 0; c < num_clusters; ++c) {
            for (int l = 0; l < m_num_labels; ++l) {
                costTables[c][l] = doubleToREAL(m_scale*w[c*m_num_labels+l]);
            }
        }
        cv::Mat patch_feature = m_patch_feature[p.Name()];
        cv::Point pt;
        for (pt.y = 0; pt.y < p.m_image.rows; ++pt.y) {
            for (pt.x = 0; pt.x < p.m_image.cols; ++pt.x) {
                int feature = patch_feature.at<int>(pt);
                ASSERT(feature < num_clusters);
                MultiLabelCRF::NodeId id = pt.y*p.m_image.cols + pt.x;
                crf.AddUnaryTerm(id, costTables[feature]);
            }
        }
    }
    virtual void Train(const std::vector<Sem_PatternData*>& patterns, const std::vector<Sem_LabelData*>& labels) {
        cv::Mat samples;
        std::cout << "Training color patches -- "; 
        std::cout.flush();
        samples.create(0, num_filters, CV_32FC1);
        for (const Sem_PatternData* xp : patterns) {
            cv::Mat im = xp->m_image;
            cv::Mat response;
            GetResponse(im, response);
            cv::Mat subsampled;
            Subsample(response, subsampled, samples_per_image);
            samples.push_back(subsampled);
        }
        std::cout << samples.rows << " samples...";
        std::cout.flush();
        cv::Mat best_labels;
        cv::kmeans(samples, num_clusters, best_labels, cv::TermCriteria(CV_TERMCRIT_EPS, 10, 0.01), 3, cv::KMEANS_RANDOM_CENTERS, m_centers);
        //std::cout << "Centers:\n";
        //std::cout << m_centers;
        std::cout << "Done!\n";
    }
    virtual void Evaluate(const std::vector<Sem_PatternData*>& patterns) override {
        std::cout << "Evaluating Color Patch Features...";
        std::cout.flush();
        for (const Sem_PatternData* xp : patterns) {
            cv::Mat im = xp->m_image;
            cv::Mat response;
            cv::Mat& features = m_patch_feature[xp->Name()];
            GetResponse(im, response);
            features.create(im.rows, im.cols, CV_32SC1);
            Classify(response, features, m_centers);
            ASSERT(features.rows == xp->m_image.rows);
            ASSERT(features.cols == xp->m_image.cols);
            ASSERT(features.type() == CV_32SC1);
        }
        std::cout << "Done!\n";
    }
    void Subsample(cv::Mat in, cv::Mat& out, size_t num_samples) {
        std::uniform_int_distribution<size_t> dist(0, in.rows-1);
        out.create(num_samples, in.cols, in.type());
        for (size_t i = 0; i < num_samples; ++i) {
            size_t r = dist(gen);
            in.row(r).copyTo(out.row(i));
        }
    }
    virtual void SaveEvaluation(const std::string& output_dir) const {
        std::string outfile = output_dir + "/color-patch-feature.dat";
        std::ofstream os(outfile, std::ios_base::binary);
        boost::archive::binary_oarchive ar(os);
        ar & m_patch_feature;
    }
    virtual void LoadEvaluation(const std::string& output_dir) {
        std::string infile = output_dir + "/color-patch-feature.dat";
        std::ifstream is(infile, std::ios_base::binary);
        boost::archive::binary_iarchive ar(is);
        ar & m_patch_feature;
    }
    private:
    void FilterResponse(cv::Mat im, cv::Mat& out) {
        ASSERT(im.type() == CV_8UC3);
        const size_t samples = im.rows*im.cols;
        out.create(samples, num_filters, CV_32FC1);
        cv::Mat im_border;
        copyMakeBorder(im, im_border, 1, 1, 1, 1, cv::BORDER_REPLICATE);

        cv::Point p_im, p_out;
        p_out.y = 0;
        for (p_im.y = 0; p_im.y < im.rows; ++p_im.y) {
            for (p_im.x = 0; p_im.x < im.cols; ++p_im.x) {
                p_out.x = 0;
                cv::Point offset;
                for (offset.y = -1; offset.y <= 1; ++offset.y) {
                    for (offset.x = -1; offset.x <= 1; ++offset.x) {
                        const cv::Vec3b& color = im_border.at<cv::Vec3b>(p_im+offset);
                        for (int i = 0; i < 3; ++i, ++p_out.x)
                            out.at<float>(p_out) = (float)(color[i]);
                    }
                }
                ASSERT(p_out.x == num_filters);
                p_out.y++;
            }
        }
        ASSERT(p_out.y == out.rows);
    }
    void GetResponse(cv::Mat im, cv::Mat& response) {
            FilterResponse(im, response);
            ASSERT(response.rows == im.cols*im.rows);
            ASSERT(response.cols == num_filters);
    }
    void Classify(cv::Mat response, cv::Mat& out, cv::Mat centers) {
        ASSERT(response.cols == num_filters);
        ASSERT(centers.cols == num_filters);
        ASSERT(centers.rows == num_clusters);
        std::vector<size_t> counts(num_clusters, 0);
        size_t i = 0;
        cv::Point p;
        for (p.y = 0; p.y < out.rows; ++p.y) {
            for (p.x = 0; p.x < out.cols; ++p.x) {
                ASSERT(i < (size_t)response.rows);
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
                counts[min_center]++;
                ASSERT(min_dist < std::numeric_limits<double>::max());
                out.at<int>(p) = min_center;
                i++;
            }
        }
        std::cout << "Cluster counts: ";
        for (auto c : counts)
            std::cout << c << " ";
        std::cout << "\n";
    }
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) { 
        //std::cout << "Serializing ColorPatchMultiFeature\n";
        ar & boost::serialization::base_object<FeatureGroup>(*this);
        ar & m_scale;
    }
    mutable std::map<std::string, cv::Mat> m_patch_feature;
    std::mt19937 gen;
    cv::Mat m_centers;
};

BOOST_CLASS_EXPORT_GUID(ColorPatchMultiFeature, "ColorPatchMultiFeature")

#endif

#ifndef _COLOR_PATCH_FEATURE_HPP_
#define _COLOR_PATCH_FEATURE_HPP_

#include "interactive_seg_app.hpp"
#include "feature.hpp"
#include <opencv2/core/core.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <unordered_set>

class ColorPatchFeature : public InteractiveSegApp::FG {
    public: 
    static constexpr int num_clusters = 50;
    static constexpr int per_cluster = 4;
    static constexpr int num_filters = 27;
    double m_scale;

    ColorPatchFeature() : m_scale(1.0) { }
    explicit ColorPatchFeature(double scale) : m_scale(scale) { }

    virtual size_t NumFeatures() const override { return num_clusters*2*per_cluster; }
    virtual std::vector<FVAL> Psi(const IS_PatternData& p, const IS_LabelData& l) const override {
        cv::Mat patch_feature = m_patch_feature[p.Name()];
        std::vector<FVAL> psi(NumFeatures(), 0);
        cv::Point pt;
        for (pt.y = 0; pt.y < p.m_image.rows; ++pt.y) {
            for (pt.x = 0; pt.x < p.m_image.cols; ++pt.x) {
                int feature = patch_feature.at<int>(pt);
                ASSERT(feature < num_clusters*per_cluster);
                unsigned char label = l.m_gt.at<unsigned char>(pt);
                psi[feature] += m_scale*LabelDiff(label, cv::GC_FGD);
                psi[feature+num_clusters*per_cluster] += m_scale*LabelDiff(label, cv::GC_BGD);
            }
        }
        for (auto& v : psi)
            v = -v;
        return psi;
    }
    virtual void AddToCRF(CRF& crf, const IS_PatternData& p, double* w) const override {
        cv::Mat patch_feature = m_patch_feature[p.Name()];
        cv::Point pt;
        for (pt.y = 0; pt.y < p.m_image.rows; ++pt.y) {
            for (pt.x = 0; pt.x < p.m_image.cols; ++pt.x) {
                int feature = patch_feature.at<int>(pt);
                ASSERT(feature < num_clusters*per_cluster);
                CRF::NodeId id = pt.y*p.m_image.cols + pt.x;
                REAL E0 = doubleToREAL(m_scale*w[feature]);
                REAL E1 = doubleToREAL(m_scale*w[feature+num_clusters*per_cluster]);
                crf.AddUnaryTerm(id, E0, E1);
            }
        }
    }
    virtual void Train(const std::vector<IS_PatternData*>& patterns, const std::vector<IS_LabelData*>& labels) {
        cv::Mat samples;
        std::cout << "Training color patches -- "; 
        std::cout.flush();
        samples.create(0, num_filters, CV_32FC1);
        for (const IS_PatternData* xp : patterns) {
            cv::Mat im = xp->m_image;
            cv::Mat response;
            GetResponse(im, response);
            samples.push_back(response);
        }
        std::cout << samples.rows << " samples...";
        std::cout.flush();
        cv::Mat best_labels;
        cv::kmeans(samples, num_clusters, best_labels, cv::TermCriteria(CV_TERMCRIT_EPS, 10, 0.01), 3, cv::KMEANS_RANDOM_CENTERS, m_centers);
        //std::cout << "Centers:\n";
        //std::cout << m_centers;
        std::cout << "Done!\n";
    }
    virtual void Evaluate(const std::vector<IS_PatternData*>& patterns) override {
        std::cout << "Evaluating Color Patch Features...";
        std::cout.flush();
        for (const IS_PatternData* xp : patterns) {
            cv::Mat im = xp->m_image;
            cv::Mat response;
            cv::Mat& features = m_patch_feature[xp->Name()];
            GetResponse(im, response);
            features.create(im.rows, im.cols, CV_32SC1);
            Classify(response, features, m_centers);
            ASSERT(features.rows == xp->m_image.rows);
            ASSERT(features.cols == xp->m_image.cols);
            ASSERT(features.type() == CV_32SC1);

            // Get 4 features per cluster, one for each combination of whether 
            // feature occurs in fgd and bgd labeled pixels in trimap
            std::unordered_set<int> fgd_features;
            std::unordered_set<int> bgd_features;
            cv::Point p;
            const cv::Mat& tri = xp->m_tri;
            for (p.y = 0; p.y < tri.rows; ++p.y) {
                for (p.x = 0; p.x < tri.cols; ++p.x) {
                    unsigned char label = tri.at<unsigned char>(p);
                    int feature = features.at<int>(p);
                    if (label == cv::GC_FGD) fgd_features.insert(feature);
                    if (label == cv::GC_BGD) bgd_features.insert(feature);
                }
            }
            std::vector<int> in_fgd_bgd(num_clusters, 0);
            for (int i = 0; i < num_clusters; ++i) {
                auto f_iter = fgd_features.find(i);
                auto b_iter = bgd_features.find(i);
                if (f_iter == fgd_features.end() && b_iter == bgd_features.end())
                    in_fgd_bgd[i] = 0;
                else if (f_iter == fgd_features.end() && b_iter != bgd_features.end())
                    in_fgd_bgd[i] = 1;
                else if (f_iter != fgd_features.end() && b_iter == bgd_features.end())
                    in_fgd_bgd[i] = 2;
                else
                    in_fgd_bgd[i] = 3;
            }

            for (p.y = 0; p.y < features.rows; ++p.y) {
                for (p.x = 0; p.x < features.cols; ++p.x) {
                    int feature = features.at<int>(p);
                    features.at<int>(p) = feature + in_fgd_bgd[feature] * num_clusters;
                    ASSERT(features.at<int>(p) < num_clusters*per_cluster);
                }
            }
        }
        std::cout << "Done!\n";
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
        //std::cout << "Serializing ColorPatchFeature\n";
        ar & boost::serialization::base_object<FeatureGroup>(*this);
        ar & m_scale;
    }
    mutable std::map<std::string, cv::Mat> m_patch_feature;
    std::mt19937 gen;
    cv::Mat m_centers;
};

BOOST_CLASS_EXPORT_GUID(ColorPatchFeature, "ColorPatchFeature")

#endif

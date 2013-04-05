#ifndef _CONTRAST_SUBMODULAR_MULTI_HPP_
#define _CONTRAST_SUBMODULAR_MULTI_HPP_

#include "semantic_seg_app.hpp"
#include "feature.hpp"
#include <opencv2/core/core.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

class ContrastSubmodularMultiFeature : public SemanticSegApp::FG {
    public: 
    typedef SemanticSegApp::FG::Constr Constr;
    typedef uint32_t Assgn;

    static constexpr Assgn clique_size = 4;
    static constexpr int num_clusters = 50;
    static constexpr size_t samples_per_image = 5000;
    double m_scale;
    int m_num_labels;
    int per_cluster;
    int per_label;

    ContrastSubmodularMultiFeature() : m_scale(1.0) { }
    explicit ContrastSubmodularMultiFeature(double scale, int num_labels) : m_scale(scale), m_num_labels(num_labels), per_cluster((1 << clique_size)*num_labels), per_label(1 << clique_size) { }

    virtual size_t NumFeatures() const override { return per_cluster*num_clusters; }
    virtual std::vector<FVAL> Psi(const Sem_PatternData& p, const Sem_LabelData& y) const override {
        cv::Mat patch_feature = m_patch_feature[p.Name()];
        std::vector<FVAL> psi(NumFeatures(), 0);
        cv::Point base, pt;
        const cv::Point patch_size(1.0, 1.0);
        for (base.y = 0; base.y + patch_size.y < p.m_image.rows; ++base.y) {
            for (base.x = 0; base.x + patch_size.x < p.m_image.cols; ++base.x) {
                int feature = patch_feature.at<int>(base);
                for (int l = 0; l < m_num_labels; ++l) {
                    Assgn a = 0;
                    int i = 0;
                    for (pt.y = base.y; pt.y <= base.y + patch_size.y; ++pt.y) {
                        for (pt.x = base.x; pt.x <= base.x + patch_size.x; ++pt.x) {
                            int label = y.m_gt.at<int>(pt);
                            if (label == l)
                                a |= 1 << i;
                            i++;
                        }
                    }
                    psi[a + l*per_label + feature*per_cluster] += m_scale;
                }
            }
        }
        for (auto& v : psi)
            v = -v;
        return psi;
    }
    virtual void AddToCRF(MultiLabelCRF& crf, const Sem_PatternData& p, double* w) const override {
        cv::Mat patch_feature = m_patch_feature[p.Name()];
        typedef SeparableClique::EnergyTable EnergyTable;
        std::vector<EnergyTable> costTables;
        for (int i = 0; i < num_clusters; ++i) {
            costTables.push_back(EnergyTable());
            for (int l = 0; l < m_num_labels; ++l) {
                costTables[i].push_back(std::vector<REAL>(per_cluster, 0));
                for (int j = 0; j < per_cluster; ++j) {
                    costTables[i][l][j] = doubleToREAL(m_scale*w[i*per_cluster + l*per_label + j]);
                }
            }
        }
        cv::Point base, pt;
        const cv::Point patch_size(1.0, 1.0);
        for (base.y = 0; base.y + patch_size.y < p.m_image.rows; ++base.y) {
            for (base.x = 0; base.x + patch_size.x < p.m_image.cols; ++base.x) {
                std::vector<MultiLabelCRF::NodeId> vars;
                int feature = patch_feature.at<int>(base);
                for (pt.y = base.y; pt.y <= base.y + patch_size.y; ++pt.y) {
                    for (pt.x = base.x; pt.x <= base.x + patch_size.x; ++pt.x) {
                        vars.push_back(pt.y*p.m_image.cols + pt.x);
                    }
                }
                MultiLabelCRF::CliquePtr cp(new SeparableClique(vars, costTables[feature]));
                crf.AddClique(cp);
            }
        }
    }
    virtual Constr CollectConstrs(size_t feature_base, double constraint_scale) const override {
        typedef std::vector<std::pair<size_t, double>> LHS;
        typedef double RHS;
        Constr ret;
        Assgn all_ones = (1 << clique_size) - 1;
        for (int cluster = 0; cluster < num_clusters; ++cluster) {
            for (int l = 0; l < m_num_labels; ++l) {
                size_t base = feature_base + cluster*per_cluster + l*per_label;
                for (Assgn s = 0; s < all_ones; ++s) {
                    for (size_t i = 0; i < clique_size; ++i) {
                        Assgn si = s | (1 << i);
                        if (si != s) {
                            for (size_t j = i+1; j < clique_size; ++j) {
                                Assgn t = s | (1 << j);
                                if (t != s && j != i) {
                                    Assgn ti = t | (1 << i);
                                    // Decreasing marginal costs, so we require
                                    // f(ti) - f(t) <= f(si) - f(s)
                                    // i.e. f(si) - f(s) - f(ti) + f(t) >= 0
                                    LHS lhs = {{base+si, constraint_scale}, {base+t, constraint_scale}};
                                    lhs.push_back(std::make_pair(base+s, -constraint_scale));
                                    lhs.push_back(std::make_pair(base+ti, -constraint_scale));
                                    RHS rhs = 0;
                                    ret.push_back(std::make_pair(lhs, rhs));
                                }
                            }
                        }
                    }
                }
            }
        }
        return ret;
    }
    virtual double Violation(size_t feature_base, double* w) const override {
        size_t num_constraints = 0;
        double total_violation = 0;
        Assgn all_ones = (1 << clique_size) - 1;
        for (int cluster = 0; cluster < num_clusters; ++cluster) {
            for (int l = 0; l < m_num_labels; ++l) {
                size_t base = feature_base + cluster*per_cluster + l*per_label;
                for (Assgn s = 0; s < all_ones; ++s) {
                    for (size_t i = 0; i < clique_size; ++i) {
                        Assgn si = s | (1 << i);
                        if (si != s) {
                            for (size_t j = i+1; j < clique_size; ++j) {
                                Assgn t = s | (1 << j);
                                if (t != s && j != i) {
                                    Assgn ti = t | (1 << i);
                                    num_constraints++;
                                    // Decreasing marginal costs, so we require
                                    // f(ti) - f(t) <= f(si) - f(s)
                                    // i.e. f(si) - f(s) - f(ti) + f(t) >= 0
                                    double violation = -w[base+si] - w[base+t];
                                    violation += w[base+s];
                                    violation += w[base+ti];
                                    if (violation > 0) total_violation += violation;
                                }
                            }
                        }
                    }
                }
            }
        }
        //std::cout << "Num constraints: " << num_constraints <<"\n";
        //std::cout << "Min violation: " << min_violation << "\n";
        return total_violation;
    }
    virtual void Train(const std::vector<Sem_PatternData*>& patterns, const std::vector<Sem_LabelData*>& labels) {
        cv::Mat samples;
        std::cout << "Training submodular filters -- "; 
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
        std::cout << "Done!\n";
    }
    virtual void Evaluate(const std::vector<Sem_PatternData*>& patterns) override {
        std::cout << "Evaluating Contrast Submodular Features...";
        std::cout.flush();
        for (const Sem_PatternData* xp : patterns) {
            cv::Mat im = xp->m_image;
            cv::Mat response;
            cv::Mat& features = m_patch_feature[xp->Name()];
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
        ar & m_patch_feature;
    }
    virtual void LoadEvaluation(const std::string& output_dir) {
        std::string infile = output_dir + "/contrast-submodular-feature.dat";
        std::ifstream is(infile, std::ios_base::binary);
        boost::archive::binary_iarchive ar(is);
        ar & m_patch_feature;
    }
    private:
    static constexpr int num_filters = 12;
    void FilterResponse(cv::Mat im, cv::Mat& out) {
        ASSERT(im.channels() == 3);
        ASSERT(im.type() == CV_32FC3);
        const size_t samples = im.rows*im.cols;
        out.create(samples, num_filters, CV_32FC1);

        std::vector<cv::Mat> all_filtered;
        cv::Mat filtered;
        filtered.create(im.rows, im.cols, CV_32FC3);

        cv::Mat x_deriv = (cv::Mat_<float>(2,2) << 1.0, -1.0, 1.0, -1.0);
        cv::filter2D(im, filtered, CV_32F, x_deriv, cv::Point(0,0));
        all_filtered.push_back(filtered);
        
        cv::Mat y_deriv = (cv::Mat_<float>(2,2) << 1.0, 1.0, -1.0, -1.0);
        cv::filter2D(im, filtered, CV_32F, y_deriv, cv::Point(0,0));
        all_filtered.push_back(filtered);
        
        cv::Mat xy_deriv = (cv::Mat_<float>(2,2) << 1.0, 0.0, 0.0, -1.0);
        cv::filter2D(im, filtered, CV_32F, xy_deriv, cv::Point(0,0));
        all_filtered.push_back(filtered);

        cv::Mat yx_deriv = (cv::Mat_<float>(2,2) << 0.0, 1.0, -1.0, 0.0);
        cv::filter2D(im, filtered, CV_32F, yx_deriv, cv::Point(0,0));
        all_filtered.push_back(filtered);

        cv::Point pf, po;
        po.y = 0;
        for (pf.y = 0; pf.y < im.rows; ++pf.y) {
            for (pf.x = 0; pf.x < im.cols; ++pf.x) {
                po.x = 0;
                for (int i = 0; i < num_filters/3; ++i) {
                    cv::Vec3f v = all_filtered[i].at<cv::Vec3f>(pf);
                    for (int j = 0; j < 3; ++j, ++po.x) {
                        out.at<float>(po) = v[j];
                    }
                }
                ASSERT(po.x == num_filters);
                po.y++;
            }
        }
        ASSERT(po.y == out.rows);
    }
    void GetResponse(cv::Mat im, cv::Mat& response) {
            cv::Mat tmp;
            im.convertTo(tmp, CV_32F);
            FilterResponse(tmp, response);
            ASSERT(response.rows == im.cols*im.rows);
            ASSERT(response.cols == num_filters);
    }
    void Subsample(cv::Mat in, cv::Mat& out, size_t num_samples) {
        std::uniform_int_distribution<size_t> dist(0, in.rows-1);
        out.create(num_samples, in.cols, in.type());
        for (size_t i = 0; i < num_samples; ++i) {
            size_t r = dist(gen);
            in.row(r).copyTo(out.row(i));
        }
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
        //std::cout << "Serializing ContrastSubmodularMultiFeature\n";
        ar & boost::serialization::base_object<FeatureGroup>(*this);
        ar & m_scale;
    }
    mutable std::map<std::string, cv::Mat> m_patch_feature;
    std::mt19937 gen;
    cv::Mat m_centers;
};

BOOST_CLASS_EXPORT_GUID(ContrastSubmodularMultiFeature, "ContrastSubmodularMultiFeature")

#endif

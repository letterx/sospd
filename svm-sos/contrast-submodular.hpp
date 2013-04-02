#ifndef _SUBMODULAR_FEATURE_HPP_
#define _SUBMODULAR_FEATURE_HPP_

#include "interactive_seg_app.hpp"
#include "feature.hpp"
#include "kmeans.hpp"
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

class ContrastSubmodularFeature : public InteractiveSegApp::FG {
    public: 
    typedef InteractiveSegApp::FG::Constr Constr;
    typedef std::function<void(const std::vector<unsigned char>&)> PatchFn;
    typedef uint32_t Assgn;

    static constexpr Assgn clique_size = 4;
    static constexpr int per_cluster = (1 << clique_size) - 2;
    static constexpr int num_clusters = 50;
    double m_scale;

    ContrastSubmodularFeature() : m_scale(1.0) { }
    explicit ContrastSubmodularFeature(double scale) : m_scale(scale) { }

    virtual size_t NumFeatures() const override { return per_cluster*num_clusters; }
    virtual std::vector<FVAL> Psi(const IS_PatternData& p, const IS_LabelData& l) const override {
        const cv::Mat& patch_feature = m_patch_feature[p.Name()];
        const Assgn all_zeros = 0;
        const Assgn all_ones = (1 << clique_size) - 1;
        std::vector<FVAL> psi(NumFeatures(), 0);
        cv::Point base, pt;
        const cv::Point patch_size(1.0, 1.0);
        for (base.y = 0; base.y + patch_size.y < p.m_image.rows; ++base.y) {
            for (base.x = 0; base.x + patch_size.x < p.m_image.cols; ++base.x) {
                Assgn a = 0;
                int feature = patch_feature.at<int>(base);
                int i = 0;
                for (pt.y = base.y; pt.y <= base.y + patch_size.y; ++pt.y) {
                    for (pt.x = base.x; pt.x <= base.x + patch_size.x; ++pt.x) {
                        unsigned char label = l.m_gt.at<unsigned char>(pt);
                        if (label == cv::GC_FGD || label == cv::GC_PR_FGD)
                            a |= 1 << i;
                        i++;
                    }
                }
                if (a != all_zeros && a != all_ones)
                    psi[a-1 + feature*per_cluster] += m_scale;
            }
        }
        for (auto& v : psi)
            v = -v;
        return psi;
    }
    virtual void AddToCRF(CRF& crf, const IS_PatternData& p, double* w) const override {
        const cv::Mat& patch_feature = m_patch_feature[p.Name()];
        std::vector<std::vector<REAL>> costTables(num_clusters, std::vector<REAL>(per_cluster+2, 0));
        for (size_t i = 0; i < num_clusters; ++i) {
            for (size_t j = 0; j < per_cluster; ++j) {
                costTables[i][j+1] = doubleToREAL(m_scale*w[i*per_cluster+j]);
            }
        }
        cv::Point base, pt;
        const cv::Point patch_size(1.0, 1.0);
        for (base.y = 0; base.y + patch_size.y < p.m_image.rows; ++base.y) {
            for (base.x = 0; base.x + patch_size.x < p.m_image.cols; ++base.x) {
                std::vector<CRF::NodeId> vars;
                int feature = patch_feature.at<int>(base);
                for (pt.y = base.y; pt.y <= base.y + patch_size.y; ++pt.y) {
                    for (pt.x = base.x; pt.x <= base.x + patch_size.x; ++pt.x) {
                        vars.push_back(pt.y*p.m_image.cols + pt.x);
                    }
                }
                crf.AddClique(vars, costTables[feature]);
            }
        }
    }
    virtual Constr CollectConstrs(size_t feature_base, double constraint_scale) const override {
        typedef std::vector<std::pair<size_t, double>> LHS;
        typedef double RHS;
        Constr ret;
        Assgn max_assgn = (1 << clique_size) - 1;
        Assgn all_zeros = 0;
        Assgn all_ones = (1 << clique_size) - 1;
        for (size_t cluster = 0; cluster < num_clusters; ++cluster) {
            size_t base = feature_base + cluster*per_cluster;
            for (Assgn s = 0; s < max_assgn; ++s) {
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
                                LHS lhs = {{base+si-1, constraint_scale}, {base+t-1, constraint_scale}};
                                if (s != all_zeros) lhs.push_back(std::make_pair(base+s-1, -constraint_scale));
                                if (ti != all_ones) lhs.push_back(std::make_pair(base+ti-1, -constraint_scale));
                                RHS rhs = 0;
                                ret.push_back(std::make_pair(lhs, rhs));
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
        Assgn max_assgn = NumFeatures();
        Assgn all_zeros = 0;
        Assgn all_ones = (1 << clique_size) - 1;
        for (size_t cluster = 0; cluster < num_clusters; ++cluster) {
            size_t base = feature_base + cluster*per_cluster;
            for (Assgn s = 0; s < max_assgn; ++s) {
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
                                double violation = -w[base+si-1] - w[base+t-1];
                                if (s != all_zeros) violation += w[base+s-1];
                                if (ti != all_ones) violation += w[base+ti-1];
                                if (violation > 0) total_violation += violation;
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
    virtual void SaveEvaluation(const std::string& output_dir) const {
        std::string outfile = output_dir + "/contrast-submodular-feature.dat";
        std::ofstream os(outfile, std::ios_base::binary);
        boost::archive::binary_oarchive ar(os);
        ar & m_bgdUnaries;
        ar & m_fgdUnaries;
    }
    virtual void LoadEvaluation(const std::string& output_dir) {
        std::string infile = output_dir + "/contrast-submodular-feature.dat";
        std::ifstream is(infile, std::ios_base::binary);
        boost::archive::binary_iarchive ar(is);
        ar & m_bgdUnaries;
        ar & m_fgdUnaries;
    }
    private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) { 
        //std::cout << "Serializing ContrastSubmodularFeature\n";
        ar & boost::serialization::base_object<FeatureGroup>(*this);
        ar & m_scale;
    }
    mutable std::map<std::string, cv::Mat> m_patch_feature;
};

BOOST_CLASS_EXPORT_GUID(ContrastSubmodularFeature, "ContrastSubmodularFeature")

#endif

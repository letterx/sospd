#include "feature.hpp"
#include <cmath>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include "svm_c++.hpp"
#include "image_manip.hpp"

class DummyFeature : public FeatureGroup {
    virtual size_t NumFeatures() const { return 1; }
    virtual std::vector<FVAL> Psi(const PatternData& p, const LabelData& l) const {
        std::vector<FVAL> psi = { 1.0 };
        return psi;
    }
    virtual void AddToCRF(CRF& crf, const PatternData& p, double* w) const {

    }
};

class SubmodularFeature : public FeatureGroup {
    public: 
    typedef FeatureGroup::Constr Constr;
    typedef std::function<void(const std::vector<unsigned char>&)> PatchFn;
    typedef uint32_t Assgn;

    static constexpr Assgn clique_size = 4;
    double m_scale;

    SubmodularFeature() : m_scale(1.0) { }
    explicit SubmodularFeature(double scale) : m_scale(scale) { }

    virtual size_t NumFeatures() const { return (1 << clique_size) - 2; }
    virtual std::vector<FVAL> Psi(const PatternData& p, const LabelData& l) const {
        Assgn all_zeros = 0;
        Assgn all_ones = (1 << clique_size) - 1;
        std::vector<FVAL> psi(NumFeatures(), 0);
        PatchFn f = [&](const std::vector<unsigned char>& labels) 
        {
            Assgn a = 0;
            for (size_t i = 0; i < clique_size; ++i) {
                if (labels[i] == cv::GC_FGD || labels[i] == cv::GC_PR_FGD)
                    a |= 1 << i;
            }
            if (a != all_zeros && a != all_ones)
                psi[a-1] += m_scale;
        };
        ImageIteratePatch(l.m_gt, cv::Point(1.0, 1.0), f);

        for (auto& v : psi)
            v = -v;
        return psi;
    }
    virtual void AddToCRF(CRF& crf, const PatternData& p, double* w) const {
        std::vector<REAL> costTable(NumFeatures()+2, 0);
        for (size_t i = 0; i < NumFeatures(); ++i) {
            costTable[i+1] = doubleToREAL(m_scale*w[i]);
        }
        typedef std::function<void(const std::vector<CRF::NodeId>&)> Fn;
        Fn f = [&](const std::vector<CRF::NodeId>& vars)
        {
            ASSERT(vars.size() == clique_size);
            crf.AddClique(vars, costTable);
        };
        ImageIteriPatch(p.m_image, cv::Point(1.0, 1.0), f);
    }
    virtual Constr CollectConstrs(size_t feature_base, double constraint_scale) const {
        typedef std::vector<std::pair<size_t, double>> LHS;
        typedef double RHS;
        Constr ret;
        Assgn max_assgn = (1 << clique_size) - 1;
        Assgn all_zeros = 0;
        Assgn all_ones = (1 << clique_size) - 1;
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
                            LHS lhs = {{feature_base+si-1, constraint_scale}, {feature_base+t-1, constraint_scale}};
                            if (s != all_zeros) lhs.push_back(std::make_pair(feature_base+s-1, -constraint_scale));
                            if (ti != all_ones) lhs.push_back(std::make_pair(feature_base+ti-1, -constraint_scale));
                            RHS rhs = 0;
                            ret.push_back(std::make_pair(lhs, rhs));
                        }
                    }
                }
            }
        }
        return ret;
    }
    virtual double Violation(size_t base, double* w) const {
        size_t num_constraints = 0;
        double total_violation = 0;
        Assgn max_assgn = NumFeatures();
        Assgn all_zeros = 0;
        Assgn all_ones = (1 << clique_size) - 1;
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
        //std::cout << "Num constraints: " << num_constraints <<"\n";
        //std::cout << "Min violation: " << min_violation << "\n";
        return total_violation;
    }
    private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) { 
        //std::cout << "Serializing SubmodularFeature\n";
        ar & boost::serialization::base_object<FeatureGroup>(*this);
        ar & m_scale;
    }
};

BOOST_CLASS_EXPORT_GUID(SubmodularFeature, "SubmodularFeature")

class PairwiseFeature : public FeatureGroup {
    public:
    typedef FeatureGroup::Constr Constr;
    double m_scale;

    PairwiseFeature() : m_scale(1.0) { }
    explicit PairwiseFeature(double scale) : m_scale(scale) { }

    virtual size_t NumFeatures() const { return 1; }
    virtual std::vector<FVAL> Psi(const PatternData& p, const LabelData& l) const {
        std::vector<FVAL> psi = {0.0};
        auto constPairwise = [&](const unsigned char& l1, const unsigned char& l2) {
            psi[0] += m_scale*LabelDiff(l1, l2);
        };
        ImageIterate(l.m_gt, cv::Point(1.0, 0.0), constPairwise);
        ImageIterate(l.m_gt, cv::Point(0.0, 1.0), constPairwise);

        for (auto& v : psi)
            v = -v;
        return psi;
    }
    virtual void AddToCRF(CRF& crf, const PatternData& p, double* w) const {
        auto constPairwise = [&](long l1, long l2) {
            crf.AddPairwiseTerm(l1, l2, 0, doubleToREAL(m_scale*w[0]), doubleToREAL(m_scale*w[0]), 0);
        };
        ImageIteri(p.m_image, cv::Point(1.0, 0.0), constPairwise);
        ImageIteri(p.m_image, cv::Point(0.0, 1.0), constPairwise);
    }
    virtual Constr CollectConstrs(size_t feature_base, double constraint_scale) const {
        Constr ret;
        std::pair<std::vector<std::pair<size_t, double>>, double> c = {{{feature_base, constraint_scale}}, 0.0};
        ret.push_back(c);
        return ret;
    }
    private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) { 
        //std::cout << "Serializing PairwiseFeature\n";
        ar & boost::serialization::base_object<FeatureGroup>(*this);
        ar & m_scale;
    }
};

BOOST_CLASS_EXPORT_GUID(PairwiseFeature, "PairwiseFeature")

class ContrastPairwiseFeature : public FeatureGroup {
    public:
    typedef FeatureGroup::Constr Constr;
    double m_scale;

    ContrastPairwiseFeature() : m_scale(1.0) { }
    explicit ContrastPairwiseFeature(double scale) : m_scale(scale) { }

    virtual size_t NumFeatures() const { return 1; }
    virtual std::vector<FVAL> Psi(const PatternData& p, const LabelData& l) const {
        std::vector<FVAL> psi = {0.0};
        std::function<void(const double&, const double&, const unsigned char&, const unsigned char&)>
            gradientPairwise = [&](const double& d1, const double& d2, const unsigned char& l1, const unsigned char& l2)
            {
                psi[0] += m_scale*d1*LabelDiff(l1, l2);
            };
        ImageIterate(p.m_downW, l.m_gt, cv::Point(0.0, 1.0), gradientPairwise);
        ImageIterate(p.m_rightW, l.m_gt, cv::Point(1.0, 0.0), gradientPairwise);

        for (auto& v : psi)
            v = -v;
        return psi;
    }
    virtual void AddToCRF(CRF& crf, const PatternData& p, double* w) const {
        cv::Mat gradWeight;
        std::function<void(const cv::Point&, const cv::Point&)> gradientPairwise = 
            [&](const cv::Point& p1, const cv::Point& p2)
            {
                REAL weight = doubleToREAL(m_scale*w[0]*gradWeight.at<double>(p1));
                CRF::NodeId i1 = p1.y*p.m_image.cols + p1.x;
                CRF::NodeId i2 = p2.y*p.m_image.cols + p2.x;
                crf.AddPairwiseTerm(i1, i2, 0, weight, weight, 0);
            };
        gradWeight = p.m_downW;
        ImageIterp(p.m_image, cv::Point(0.0, 1.0), gradientPairwise);
        gradWeight = p.m_rightW;
        ImageIterp(p.m_image, cv::Point(1.0, 0.0), gradientPairwise);
    }
    virtual Constr CollectConstrs(size_t feature_base, double constraint_scale) const {
        Constr ret;
        std::pair<std::vector<std::pair<size_t, double>>, double> c = {{{feature_base, constraint_scale}}, 0.0};
        ret.push_back(c);
        return ret;
    }
    private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) { 
        //std::cout << "Serializing ContrastPairwiseFeature\n";
        ar & boost::serialization::base_object<FeatureGroup>(*this);
        ar & m_scale;
    }
};

BOOST_CLASS_EXPORT_GUID(ContrastPairwiseFeature, "ContrastPairwiseFeature")


class GMMFeature : public FeatureGroup {
    public:
    double m_scale;
    GMMFeature() : m_scale(1.0) { }
    explicit GMMFeature(double scale) : m_scale(0.1*scale) { }

    static constexpr double prob_epsilon = 0.00001;

    virtual size_t NumFeatures() const { return 3; }
    virtual std::vector<FVAL> Psi(const PatternData& p, const LabelData& l) const {
        std::vector<FVAL> psi = {0.0, 0.0, 0.0};
        ImageCIterate3_1(p.m_image, l.m_gt, 
            [&](const cv::Vec3b& color, const unsigned char& label) {
                double bgd_prob = p.m_bgdGMM(color);
                double fgd_prob = p.m_fgdGMM(color);
                if (bgd_prob < prob_epsilon) bgd_prob = prob_epsilon;
                if (fgd_prob < prob_epsilon) fgd_prob = prob_epsilon;
                psi[0] += -log(bgd_prob)*m_scale*LabelDiff(label, cv::GC_FGD);
                psi[0] += -log(fgd_prob)*m_scale*LabelDiff(label, cv::GC_BGD);
                ASSERT(!std::isnan(psi[0]));
                ASSERT(std::isfinite(psi[0]));
            });
        ImageCIterate(p.m_tri, l.m_gt,
            [&](const unsigned char& tri_label, const unsigned char& label) {
                if (tri_label == cv::GC_BGD)
                    psi[1] += m_scale*LabelDiff(label, cv::GC_BGD);
                if (tri_label == cv::GC_FGD)
                    psi[2] += m_scale*LabelDiff(label, cv::GC_FGD);
            });
        for (auto& v : psi)
            v = -v;
        return psi;
    }
    virtual void AddToCRF(CRF& crf, const PatternData& p, double* w) const {
        cv::Point pt;
        for (pt.y = 0; pt.y < p.m_image.rows; ++pt.y) {
            for (pt.x = 0; pt.x < p.m_image.cols; ++pt.x) {
                const cv::Vec3b& color = p.m_image.at<cv::Vec3b>(pt);
                CRF::NodeId id = pt.y * p.m_image.cols + pt.x;
                double bgd_prob = p.m_bgdGMM(color);
                double fgd_prob = p.m_fgdGMM(color);
                if (bgd_prob < prob_epsilon) bgd_prob = prob_epsilon;
                if (fgd_prob < prob_epsilon) fgd_prob = prob_epsilon;
                double E0 = w[0]*-log(bgd_prob)*m_scale;
                double E1 = w[0]*-log(fgd_prob)*m_scale;
                if (p.m_tri.at<unsigned char>(pt) == cv::GC_BGD) E1 += w[1]*m_scale;
                if (p.m_tri.at<unsigned char>(pt) == cv::GC_FGD) E0 += w[2]*m_scale;
                crf.AddUnaryTerm(id, doubleToREAL(E0), doubleToREAL(E1));
            }
        }
    }
    private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        //std::cout << "Serializing GMMFeature\n";
        ar & boost::serialization::base_object<FeatureGroup>(*this);
        ar & m_scale;
    }
};

BOOST_CLASS_EXPORT_GUID(GMMFeature, "GMMFeature")
                

std::vector<boost::shared_ptr<FeatureGroup>> GetFeaturesFromParam(STRUCT_LEARN_PARM* sparm) {
    std::vector<boost::shared_ptr<FeatureGroup>> features;
    features.push_back(boost::shared_ptr<FeatureGroup>(new GMMFeature(sparm->feature_scale)));
    if (sparm->pairwise_feature) {
        std::cout << "Adding PairwiseFeature\n";
        features.push_back(boost::shared_ptr<FeatureGroup>(new PairwiseFeature(sparm->feature_scale)));
    }
    if (sparm->contrast_pairwise_feature) {
        std::cout << "Adding ContrastPairwiseFeature\n";
        features.push_back(boost::shared_ptr<FeatureGroup>(new ContrastPairwiseFeature(sparm->feature_scale)));
    }
    if (sparm->submodular_feature) {
        std::cout << "Adding SubmodularFeature\n";
        features.push_back(boost::shared_ptr<FeatureGroup>(new SubmodularFeature(sparm->feature_scale)));
    }
    return features;
}

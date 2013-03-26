#include <cmath>
#include "svm_c++.hpp"
#include "image_manip.hpp"
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

inline double LabelDiff(unsigned char l1, unsigned char l2) {
    if (l1 == cv::GC_BGD || l1 == cv::GC_PR_BGD) {
        if (l2 == cv::GC_FGD || l2 == cv::GC_PR_FGD)
            return 1.0;
        else if (l2 == (unsigned char)-1)
            return 0.5;
        else 
            return 0.0;
    } else if (l1 == cv::GC_FGD || l1 == cv::GC_PR_FGD) {
        if (l2 == cv::GC_BGD || l2 == cv::GC_PR_BGD)
            return 1.0;
        else if (l2 == (unsigned char)-1)
            return 0.5;
        else 
            return 0.0;
    } else if (l1 == l2)
        return 0.0;
    else
        return 0.5;
}

PatternData::PatternData(const std::string& name, const cv::Mat& image, const cv::Mat& trimap) 
    : m_name(name), 
    m_image(image),
    m_bgdModel(),
    m_bgdGMM(m_bgdModel),
    m_fgdModel(),
    m_fgdGMM(m_fgdModel)
{
    ConvertToMask(trimap, m_tri);

    learnGMMs(m_image, m_tri, m_bgdGMM, m_fgdGMM);

    m_beta = calcBeta(m_image);
    m_downW.create(m_image.rows, m_image.cols, CV_64FC1);
    m_rightW.create(m_image.rows, m_image.cols, CV_64FC1);
    std::function<void(const cv::Vec3b&, const cv::Vec3b&, double&, double&)> calcExpDiff = 
        [&](const cv::Vec3b& color1, const cv::Vec3b& color2, double& d1, double& d2) {
            cv::Vec3d c1 = color1;
            cv::Vec3d c2 = color2;
            cv::Vec3d diff = c1-c2;
            d1 = exp(-m_beta*diff.dot(diff));
            //d1 = abs(diff[0]) + abs(diff[1]) + abs(diff[2]);
    };
    ImageIterate(m_image, m_downW, cv::Point(0.0, 1.0), calcExpDiff);
    ImageIterate(m_image, m_rightW, cv::Point(1.0, 0.0), calcExpDiff);

}

LabelData::LabelData(const std::string& name, const cv::Mat& gt)
    : m_name(name)
{ 
    ConvertGreyToMask(gt, m_gt);
}

bool LabelData::operator==(const LabelData& l) const {
    ASSERT(m_gt.size() == l.m_gt.size());
    cv::MatConstIterator_<unsigned char> i1, e1, i2;
    i1 = m_gt.begin<unsigned char>();
    e1 = m_gt.end<unsigned char>();
    i2 = l.m_gt.begin<unsigned char>();
    for (; i1 != e1; ++i1, ++i2) {
        if (*i1 != *i2) {
            return false;
        }
    }
    return true;
}

double LabelData::Loss(const LabelData& l, double scale) const {
    ASSERT(m_gt.size() == l.m_gt.size());
    double loss = 0;
    ImageCIterate(m_gt, l.m_gt, 
            [&](const unsigned char& c1, const unsigned char& c2) {
                loss += LabelDiff(c1, c2);
            });
    loss /= (m_gt.rows*m_gt.cols);
    return loss*scale;
}

class DummyFeature : public FG {
    virtual size_t NumFeatures() const { return 1; }
    virtual std::vector<FVAL> Psi(const PatternData& p, const LabelData& l) const {
        std::vector<FVAL> psi = { 1.0 };
        return psi;
    }
    virtual void AddToCRF(CRF& crf, const PatternData& p, double* w) const {

    }
};

class SubmodularFeature : public FG {
    public: 
    typedef FG::Constr Constr;
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
        ar & boost::serialization::base_object<FG>(*this);
        ar & m_scale;
    }
};

BOOST_CLASS_EXPORT_GUID(SubmodularFeature, "SubmodularFeature")

class PairwiseFeature : public FG {
    public:
    typedef FG::Constr Constr;
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
        ar & boost::serialization::base_object<FG>(*this);
        ar & m_scale;
    }
};

BOOST_CLASS_EXPORT_GUID(PairwiseFeature, "PairwiseFeature")

class ContrastPairwiseFeature : public FG {
    public:
    typedef FG::Constr Constr;
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
        ar & boost::serialization::base_object<FG>(*this);
        ar & m_scale;
    }
};

BOOST_CLASS_EXPORT_GUID(ContrastPairwiseFeature, "ContrastPairwiseFeature")


class GMMFeature : public FG {
    public:
    double m_scale;
    GMMFeature() : m_scale(1.0) { }
    explicit GMMFeature(double scale) : m_scale(scale) { }

    virtual size_t NumFeatures() const { return 3; }
    virtual std::vector<FVAL> Psi(const PatternData& p, const LabelData& l) const {
        std::vector<FVAL> psi = {0.0, 0.0, 0.0};
        ImageCIterate3_1(p.m_image, l.m_gt, 
            [&](const cv::Vec3b& color, const unsigned char& label) {
                double bgd_prob = p.m_bgdGMM(color);
                double fgd_prob = p.m_fgdGMM(color);
                if (bgd_prob < 0.0000001) bgd_prob = 0.0000001;
                if (fgd_prob < 0.0000001) fgd_prob = 0.0000001;
                psi[0] += -log(bgd_prob)*m_scale*LabelDiff(label, cv::GC_FGD);
                psi[0] += -log(fgd_prob)*m_scale*LabelDiff(label, cv::GC_BGD);
                ASSERT(!std::isnan(psi[0]));
                ASSERT(std::isfinite(psi[0]));
                ASSERT(std::isnormal(psi[0]));
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
                if (bgd_prob < 0.0000001) bgd_prob = 0.0000001;
                if (fgd_prob < 0.0000001) fgd_prob = 0.0000001;
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
        ar & boost::serialization::base_object<FG>(*this);
        ar & m_scale;
    }
};

BOOST_CLASS_EXPORT_GUID(GMMFeature, "GMMFeature")
                

ModelData::ModelData() { }

void ModelData::InitFeatures(STRUCT_LEARN_PARM* sparm) {
    m_features.push_back(boost::shared_ptr<FG>(new GMMFeature(sparm->feature_scale)));
    if (sparm->pairwise_feature) {
        std::cout << "Adding PairwiseFeature\n";
        m_features.push_back(boost::shared_ptr<FG>(new PairwiseFeature(sparm->feature_scale)));
    }
    if (sparm->contrast_pairwise_feature) {
        std::cout << "Adding ContrastPairwiseFeature\n";
        m_features.push_back(boost::shared_ptr<FG>(new ContrastPairwiseFeature(sparm->feature_scale)));
    }
    if (sparm->submodular_feature) {
        std::cout << "Adding SubmodularFeature\n";
        m_features.push_back(boost::shared_ptr<FG>(new SubmodularFeature(sparm->feature_scale)));
    }
}

long ModelData::NumFeatures() const {
    long n = 0;
    for (auto fgp : m_features) {
        n += fgp->NumFeatures();
    }
    return n;
}

void ModelData::InitializeCRF(CRF& crf, const PatternData& p) const {
    crf.AddNode(p.m_image.rows*p.m_image.cols);

}

void ModelData::AddLossToCRF(CRF& crf, const PatternData& p, const LabelData& l, double scale) const {
    double mult = scale/(p.m_image.rows*p.m_image.cols);
    cv::Point pt;
    for (pt.y = 0; pt.y < p.m_image.rows; ++pt.y) {
        for (pt.x = 0; pt.x < p.m_image.cols; ++pt.x) {
            CRF::NodeId id = pt.y * p.m_image.cols + pt.x;
            double E0 = 0;
            double E1 = 0;
            if (l.m_gt.at<unsigned char>(pt) == cv::GC_BGD) E1 -= 1.0*mult;
            if (l.m_gt.at<unsigned char>(pt) == cv::GC_FGD) E0 -= 1.0*mult;
            crf.AddUnaryTerm(id, doubleToREAL(E0), doubleToREAL(E1));
        }
    }
}

LabelData* ModelData::ExtractLabel(const CRF& crf, const PatternData& x) const {
    LabelData* lp = new LabelData;
    lp->m_name = x.m_name;
    lp->m_gt.create(x.m_image.rows, x.m_image.cols, CV_8UC1);
    CRF::NodeId id = 0;
    ImageIterate(lp->m_gt, 
        [&](unsigned char& c) { 
            //ASSERT(crf.GetLabel(id) >= 0);
            if (crf.GetLabel(id) == 0) c = cv::GC_BGD;
            else if (crf.GetLabel(id) == 1) c = cv::GC_FGD;
            else c = -1;
            id++;
        });
    //x.m_tri.copyTo(lp->m_gt);
    //cv::Mat bgdModel, fgdModel;
    //cv::grabCut(x.m_image, lp->m_gt, cv::Rect(), bgdModel, fgdModel, 10, cv::GC_INIT_WITH_MASK);
    return lp;
}

LabelData* ModelData::Classify(const PatternData& x, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const {
    CRF crf;
    SubmodularFlow sf;
    HigherOrderWrapper ho;
    if (sparm->crf == 0) {
        crf.Wrap(&sf);
    } else {
        crf.Wrap(&ho);
    }
    InitializeCRF(crf, x);
    size_t feature_base = 1;
    for (auto fgp : m_features) {
        double violation = fgp->Violation(feature_base, sm->w);
        double w2 = 0;
        for (size_t i = feature_base; i < feature_base + fgp->NumFeatures(); ++i) 
            w2 += sm->w[i] * sm->w[i];
        if (violation > 0.0001 * w2)
            std::cout << "\n***\n*** Max Violation: " << violation << "\t|w|^2: " << w2 << "\n***\n";
        fgp->AddToCRF(crf, x, sm->w + feature_base );
        feature_base += fgp->NumFeatures();
    }
    crf.Solve();
    return ExtractLabel(crf, x);
}

LabelData* ModelData::FindMostViolatedConstraint(const PatternData& x, const LabelData& y, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const {
    CRF crf;
    SubmodularFlow sf;
    HigherOrderWrapper ho;
    if (sparm->crf == 0) {
        crf.Wrap(&sf);
    } else {
        crf.Wrap(&ho);
    }
    InitializeCRF(crf, x);
    size_t feature_base = 1;
    for (auto fgp : m_features) {
        double violation = fgp->Violation(feature_base, sm->w);
        double w2 = 0;
        for (size_t i = feature_base; i < feature_base + fgp->NumFeatures(); ++i) 
            w2 += sm->w[i] * sm->w[i];
        if (violation > 0.0001 * w2)
            std::cout << "\n***\n*** Max Violation: " << violation << "\t|w|^2: " << w2 << "\n***\n";
        fgp->AddToCRF(crf, x, sm->w + feature_base );
        feature_base += fgp->NumFeatures();
    }
    AddLossToCRF(crf, x, y, sparm->loss_scale);
    crf.Solve();
    return ExtractLabel(crf, x);
}

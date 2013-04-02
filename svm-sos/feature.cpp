#include "feature.hpp"
#include <cmath>
#include <queue>
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

    virtual size_t NumFeatures() const { return 3; }
    virtual std::vector<FVAL> Psi(const PatternData& p, const LabelData& l) const {
        std::vector<FVAL> psi = {0.0, 0.0, 0.0};
        cv::Point pt;
        for (pt.y = 0; pt.y < p.m_image.rows; ++pt.y) {
            for (pt.x = 0; pt.x < p.m_image.cols; ++pt.x) {
                unsigned char label = l.m_gt.at<unsigned char>(pt);
                psi[0] += p.m_bgdUnaries.at<double>(pt)*m_scale*LabelDiff(label, cv::GC_FGD);
                psi[0] += p.m_fgdUnaries.at<double>(pt)*m_scale*LabelDiff(label, cv::GC_BGD);
                ASSERT(!std::isnan(psi[0]));
                ASSERT(std::isfinite(psi[0]));
            }
        }
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
                CRF::NodeId id = pt.y * p.m_image.cols + pt.x;
                double E0 = w[0]*p.m_bgdUnaries.at<double>(pt)*m_scale;
                double E1 = w[0]*p.m_fgdUnaries.at<double>(pt)*m_scale;
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

class DistanceFeature : public FeatureGroup {
    public:
    double m_scale;
    DistanceFeature() : m_scale(1.0) { }
    explicit DistanceFeature(double scale) : m_scale(0.1*scale) { }

    static constexpr int numBins = 10;

    virtual size_t NumFeatures() const { return numBins*numBins*2; }
    virtual std::vector<FVAL> Psi(const PatternData& p, const LabelData& l) const {
        std::vector<FVAL> psi(NumFeatures(), 0);
        cv::Point pt;
        for (pt.y = 0; pt.y < p.m_image.rows; ++pt.y) {
            for (pt.x = 0; pt.x < p.m_image.cols; ++pt.x) {
                unsigned char label = l.m_gt.at<unsigned char>(pt);
                int feature = p.m_dist_feature.at<int>(pt);
                psi[feature] += m_scale*LabelDiff(label, cv::GC_FGD);
                psi[feature+numBins*numBins] += m_scale*LabelDiff(label, cv::GC_BGD);
            }
        }
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
                CRF::NodeId id = pt.y * p.m_image.cols + pt.x;
                int feature = p.m_dist_feature.at<int>(pt);
                double E0 = m_scale*w[feature];
                double E1 = m_scale*w[feature+numBins*numBins];
                crf.AddUnaryTerm(id, doubleToREAL(E0), doubleToREAL(E1));
            }
        }
    }
    private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        //std::cout << "Serializing DistanceFeature\n";
        ar & boost::serialization::base_object<FeatureGroup>(*this);
        ar & m_scale;
    }
};

BOOST_CLASS_EXPORT_GUID(DistanceFeature, "DistanceFeature")
                
void CalcUnaries(PatternData& p) {
    static constexpr double prob_epsilon = 0.00001;

    cv::Point pt;
    for (pt.y = 0; pt.y < p.m_image.rows; ++pt.y) {
        for (pt.x = 0; pt.x < p.m_image.cols; ++pt.x) {
            const cv::Vec3b& color = p.m_image.at<cv::Vec3b>(pt);
            double bgd_prob = p.m_bgdGMM(color);
            double fgd_prob = p.m_fgdGMM(color);
            if (bgd_prob < prob_epsilon) bgd_prob = prob_epsilon;
            if (fgd_prob < prob_epsilon) fgd_prob = prob_epsilon;
            p.m_bgdUnaries.at<double>(pt) = -log(bgd_prob);
            p.m_fgdUnaries.at<double>(pt) = -log(fgd_prob);
        }
    }
}

static void BFS(const cv::Mat& tri, std::queue<cv::Point>& queue, cv::Mat& dist) {
    const std::vector<cv::Point> offsets = {cv::Point(0, 1), cv::Point(1, 0), cv::Point(0, -1), cv::Point(-1, 0)};
    while (!queue.empty()) {
        cv::Point p = queue.front();
        int p_dist = dist.at<int>(p);
        queue.pop();
        for (const cv::Point& o : offsets) {
            cv::Point q = p+o;
            if (q.x >= 0 && q.x < tri.cols && q.y >= 0 && q.y < tri.rows) {
                int& q_dist = dist.at<int>(q);
                if (q_dist > p_dist + 1) {
                    q_dist = p_dist + 1;
                    queue.push(q);
                }
            }
        }
    }
}

uint32_t CalcFeature(int fgdDist, int bgdDist, int fgdBins, int bgdBins) {
    int i, j;
    // Quadratic binning for distance feature
    for (i = 0; i < fgdBins-1; ++i) {
        if (fgdDist <= i*i) break;
    }
    for (j = 0; j < bgdBins-1; ++j) {
        if (bgdDist <= j*j) break;
    }
    return i*bgdBins + j;
}


void CalcDistances(const cv::Mat& tri, cv::Mat& fgdDist, cv::Mat& bgdDist, cv::Mat& distFeature) {
    typedef std::queue<cv::Point> Queue;
    Queue fgdQueue;
    Queue bgdQueue;
    cv::Point p;
    for (p.y = 0; p.y < tri.rows; p.y++) {
        for (p.x = 0; p.x < tri.cols; ++p.x) {
            fgdDist.at<int>(p) = std::numeric_limits<int>::max();
            bgdDist.at<int>(p) = std::numeric_limits<int>::max();
            unsigned char label = tri.at<unsigned char>(p);
            if (label == cv::GC_FGD) {
                fgdQueue.push(p);
                fgdDist.at<int>(p) = 0;
            }
            if (label == cv::GC_BGD) {
                bgdQueue.push(p);
                bgdDist.at<int>(p) = 0;
            }
        }
    }

    ASSERT(!fgdQueue.empty());
    ASSERT(!bgdQueue.empty());
    BFS(tri, fgdQueue, fgdDist);
    BFS(tri, bgdQueue, bgdDist);

    for (p.y = 0; p.y < tri.rows; p.y++) {
        for (p.x = 0; p.x < tri.cols; ++p.x) {
            int f_dist = fgdDist.at<int>(p);
            int b_dist = bgdDist.at<int>(p);
            distFeature.at<uint32_t>(p) = CalcFeature(f_dist, b_dist, 10, 10);
        }
    }
}

std::vector<boost::shared_ptr<FeatureGroup>> GetFeaturesFromParam(STRUCT_LEARN_PARM* sparm) {
    std::vector<boost::shared_ptr<FeatureGroup>> features;
    features.push_back(boost::shared_ptr<FeatureGroup>(new GMMFeature(sparm->feature_scale)));
    if (sparm->distance_unary)
        features.push_back(boost::shared_ptr<FeatureGroup>(new DistanceFeature(sparm->feature_scale)));
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


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


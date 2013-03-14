#include <cmath>
#include "svm_c++.hpp"
#include "image_manip.hpp"

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

double LabelData::Loss(const LabelData& l) const {
    ASSERT(m_gt.size() == l.m_gt.size());
    double loss = 0;
    ImageCIterate(m_gt, l.m_gt, 
            [&](const unsigned char& c1, const unsigned char& c2) {
                if ((c1 == cv::GC_BGD || c1 == cv::GC_PR_BGD) && (c2 == cv::GC_FGD || c2 == cv::GC_PR_FGD)) {
                    loss += 1.0;
                } else if ((c1 == cv::GC_FGD || c1 == cv::GC_PR_FGD) && (c2 == cv::GC_BGD || c2 == cv::GC_PR_BGD)) {
                    loss += 1.0;
                }
            });
    loss /= (m_gt.rows*m_gt.cols);
    return loss;
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

class PairwiseFeature : public FG {
    public:
    typedef FG::Constr Constr;
    static constexpr double scale = 0.01;
    virtual size_t NumFeatures() const { return 2; }
    virtual std::vector<FVAL> Psi(const PatternData& p, const LabelData& l) const {
        std::vector<FVAL> psi = {0.0, 0.0};
        auto constPairwise = [&](const unsigned char& l1, const unsigned char& l2) {
            if (l1 != l2) psi[0] += scale;
        };
        ImageIterate(l.m_gt, cv::Point(1.0, 0.0), constPairwise);
        ImageIterate(l.m_gt, cv::Point(0.0, 1.0), constPairwise);
        std::function<void(const double&, const double&, const unsigned char&, const unsigned char&)>
            gradientPairwise = [&](const double& d1, const double& d2, const unsigned char& l1, const unsigned char& l2)
            {
                if (l1 != l2) psi[1] += d1;
            };
        ImageIterate(p.m_downW, l.m_gt, cv::Point(0.0, 1.0), gradientPairwise);
        ImageIterate(p.m_rightW, l.m_gt, cv::Point(1.0, 0.0), gradientPairwise);

        for (auto& v : psi)
            v = -v;
        return psi;
    }
    virtual void AddToCRF(CRF& crf, const PatternData& p, double* w) const {
        auto constPairwise = [&](long l1, long l2) {
            crf.AddPairwiseTerm(l1, l2, 0, doubleToREAL(scale*w[0]), doubleToREAL(scale*w[0]), 0);
        };
        ImageIteri(p.m_image, cv::Point(1.0, 0.0), constPairwise);
        ImageIteri(p.m_image, cv::Point(0.0, 1.0), constPairwise);
        cv::Mat gradWeight;
        std::function<void(const cv::Point&, const cv::Point&)> gradientPairwise = 
            [&](const cv::Point& p1, const cv::Point& p2)
            {
                REAL weight = doubleToREAL(w[1]*gradWeight.at<double>(p1));
                CRF::NodeId i1 = p1.y*p.m_image.cols + p1.x;
                CRF::NodeId i2 = p2.y*p.m_image.cols + p2.x;
                crf.AddPairwiseTerm(i1, i2, 0, weight, weight, 0);
            };
        gradWeight = p.m_downW;
        ImageIterp(p.m_image, cv::Point(0.0, 1.0), gradientPairwise);
        gradWeight = p.m_rightW;
        ImageIterp(p.m_image, cv::Point(1.0, 0.0), gradientPairwise);
    }
    virtual Constr CollectConstrs(size_t feature_base) const {
        Constr ret;
        std::pair<std::vector<std::pair<size_t, double>>, double> c = {{{feature_base, 1.0}}, 0.0};
        ret.push_back(c);
        c = {{{feature_base+1, 1.0}}, 0.0};
        return ret;
    }
};



class GMMFeature : public FG {
    virtual size_t NumFeatures() const { return 3; }
    virtual std::vector<FVAL> Psi(const PatternData& p, const LabelData& l) const {
        std::vector<FVAL> psi = {0.0, 0.0, 0.0};
        ImageCIterate3_1(p.m_image, l.m_gt, 
            [&](const cv::Vec3b& color, const unsigned char& label) {
                double prob;
                if (label == cv::GC_BGD || label == cv::GC_PR_BGD) {
                    prob = p.m_bgdGMM(color);
                } else if (label == cv::GC_FGD || label == cv::GC_PR_FGD) {
                    prob = p.m_fgdGMM(color);
                } else {
                    ASSERT(false /* should never reach here? */);
                    prob = 0;
                }
                if (prob < 0.0000001) prob = 0.0000001;
                psi[0] += -log(prob)*0.001;
                ASSERT(!std::isnan(psi[0]));
                ASSERT(std::isfinite(psi[0]));
                ASSERT(std::isnormal(psi[0]));
            });
        ImageCIterate(p.m_tri, l.m_gt,
            [&](const unsigned char& tri_label, const unsigned char& label) {
                if (tri_label == cv::GC_BGD 
                    && (label == cv::GC_FGD || label == cv::GC_PR_FGD)) 
                    psi[1] += 1.0;
                if (tri_label == cv::GC_FGD
                    && (label == cv::GC_BGD || label == cv::GC_PR_BGD))
                    psi[2] += 1.0;
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
                double E0 = w[0]*-log(p.m_bgdGMM(color))*0.001;
                double E1 = w[0]*-log(p.m_fgdGMM(color))*0.001;
                if (p.m_tri.at<unsigned char>(pt) == cv::GC_BGD) E1 += w[1];
                if (p.m_tri.at<unsigned char>(pt) == cv::GC_FGD) E0 += w[2];
                crf.AddUnaryTerm(id, doubleToREAL(E0), doubleToREAL(E1));
            }
        }
    }
};

                

ModelData::ModelData() {
    m_features.push_back(std::shared_ptr<FG>(new GMMFeature));
    m_features.push_back(std::shared_ptr<FG>(new PairwiseFeature));

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

void ModelData::AddLossToCRF(CRF& crf, const PatternData& p, const LabelData& l) const {
    double mult = 1.0/(p.m_image.rows*p.m_image.cols);
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
    lp->m_gt.create(x.m_image.rows, x.m_image.cols, CV_8UC1);
    CRF::NodeId id = 0;
    ImageIterate(lp->m_gt, 
        [&](unsigned char& c) { 
            ASSERT(crf.GetLabel(id) >= 0);
            if (crf.GetLabel(id) == 0) c = cv::GC_BGD;
            else c = cv::GC_FGD;
            id++;
        });
    //x.m_tri.copyTo(lp->m_gt);
    //cv::Mat bgdModel, fgdModel;
    //cv::grabCut(x.m_image, lp->m_gt, cv::Rect(), bgdModel, fgdModel, 10, cv::GC_INIT_WITH_MASK);
    return lp;
}

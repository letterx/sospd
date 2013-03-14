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

class GMMFeature : public FG {
    virtual size_t NumFeatures() const { return 3; }
    virtual std::vector<FVAL> Psi(const PatternData& p, const LabelData& l) const {
        std::vector<FVAL> psi = {0.0, 0.0, 0.0};
        ImageCIterate3_1(p.m_image, l.m_gt, 
            [&](const cv::Vec3b& color, const unsigned char& label) {
                if (label == cv::GC_BGD || label == cv::GC_PR_BGD) {
                    psi[0] += -log(p.m_bgdGMM(color));
                } else if (label == cv::GC_FGD || label == cv::GC_PR_FGD) {
                    psi[0] += -log(p.m_fgdGMM(color));
                } else {
                    ASSERT(false /* should never reach here? */);
                }
            });
        ImageCIterate(p.m_tri,
            [&](const unsigned char& label) {
                if (label == cv::GC_BGD) psi[1] += 1.0;
                if (label == cv::GC_FGD) psi[2] += 1.0;
            });
        return psi;
    }
    virtual void AddToCRF(CRF& crf, const PatternData& p, double* w) const {
        cv::Point pt;
        for (pt.y = 0; pt.y < p.m_image.rows; ++pt.y) {
            for (pt.x = 0; pt.x < p.m_image.cols; ++pt.x) {
                const cv::Vec3b& color = p.m_image.at<cv::Vec3b>(pt);
                CRF::NodeId id = pt.y * p.m_image.cols + pt.x;
                double E0 = w[0]*-log(p.m_bgdGMM(color));
                double E1 = w[0]*-log(p.m_fgdGMM(color));
                if (p.m_tri.at<unsigned char>(pt) == cv::GC_BGD) E1 += w[1];
                if (p.m_tri.at<unsigned char>(pt) == cv::GC_FGD) E0 += w[2];
                crf.AddUnaryTerm(id, E0, E1);
            }
        }
    }
};

                

ModelData::ModelData() {
    m_features.push_back(std::shared_ptr<FG>(new GMMFeature));

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

}

LabelData* ModelData::ExtractLabel(const CRF& crf, const PatternData& x) const {
    LabelData* lp = new LabelData;
    lp->m_gt.create(x.m_image.rows, x.m_image.cols, CV_8UC1);
    CRF::NodeId id = 0;
    ImageIterate(lp->m_gt, 
        [&](unsigned char& c) { 
            if (crf.GetLabel(id) == 0) c = cv::GC_BGD;
            else c = cv::GC_FGD;
            id++;
        });
    //x.m_tri.copyTo(lp->m_gt);
    //cv::Mat bgdModel, fgdModel;
    //cv::grabCut(x.m_image, lp->m_gt, cv::Rect(), bgdModel, fgdModel, 10, cv::GC_INIT_WITH_MASK);
    return lp;
}

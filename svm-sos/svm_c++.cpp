#include "svm_c++.hpp"
#include "image_manip.hpp"

PatternData::PatternData(const cv::Mat& image, const cv::Mat& trimap) 
    : m_image(image)
{
    ConvertToMask(trimap, m_tri);
}

LabelData::LabelData(const cv::Mat& gt)
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
    cv::MatConstIterator_<unsigned char> i1, e1, i2;
    i1 = m_gt.begin<unsigned char>();
    e1 = m_gt.end<unsigned char>();
    i2 = l.m_gt.begin<unsigned char>();
    for (; i1 != e1; ++i1, ++i2) {
        if ((*i1 == cv::GC_BGD || *i1 == cv::GC_PR_BGD) && (*i2 == cv::GC_FGD || *i2 == cv::GC_PR_FGD)) {
            loss++;
        } else if ((*i1 == cv::GC_FGD || *i1 == cv::GC_PR_FGD) && (*i2 == cv::GC_BGD || *i2 == cv::GC_PR_BGD)) {
            loss++;
        }
    }
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

ModelData::ModelData() {
    m_features.push_back(std::shared_ptr<FG>(new DummyFeature));

}

long ModelData::NumFeatures() const {
    long n = 0;
    for (auto fgp : m_features) {
        n += fgp->NumFeatures();
    }
    return n;
}

void ModelData::InitializeCRF(CRF& crf, const PatternData& p) const {

}

void ModelData::AddLossToCRF(CRF& crf, const PatternData& p, const LabelData& l) const {

}

LabelData* ModelData::ExtractLabel(const CRF& crf) const {
    LabelData* lp = new LabelData;

    return lp;
}

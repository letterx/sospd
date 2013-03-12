#include "svm_c++.hpp"

bool LabelData::operator==(const LabelData& l) const {
    ASSERT(m_gt.size() == l.m_gt.size());
    for (size_t i = 0; i < m_gt.size(); ++i) {
        if (m_gt[i] != l.m_gt[i])
            return false;
    }
    return true;
}

double LabelData::Loss(const LabelData& l) const {
    ASSERT(m_gt.size() == l.m_gt.size());
    double loss = 0;
    for (size_t i = 0; i < m_gt.size(); ++i) {
        if (m_gt[i] != l.m_gt[i]) loss += 1.0;
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

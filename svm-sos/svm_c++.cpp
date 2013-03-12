#include "svm_c++.hpp"


void ModelData::InitializeCRF(CRF& crf, const PatternData& p) const {

}

void ModelData::AddLossToCRF(CRF& crf, const PatternData& p, const LabelData& l) const {

}

LabelData* ModelData::ExtractLabel(const CRF& crf) const {
    LabelData* lp = new LabelData;

    return lp;
}

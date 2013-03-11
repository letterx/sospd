#ifndef _SVM_FEATURE_HPP_
#define _SVM_FEATURE_HPP_

extern "C" {
#include "svm_light/svm_common.h"
#include "svm_struct/svm_struct_common.h"
}
#include <vector>

template <typename PatternData, typename LabelData, typename CRF>
class FeatureGroup {
    public:
        virtual const size_t NumFeatures() const = 0;
        virtual std::vector<FVAL> Psi(const PatternData& p, const LabelData& l) const = 0;
        virtual void AddToCRF(CRF& c, const PatternData& p) const = 0;
};



#endif


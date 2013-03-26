#ifndef _FEATURE_HPP_
#define _FEATURE_HPP_

extern "C" {
#include "svm_light/svm_common.h"
#include "svm_struct/svm_struct_common.h"
}
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/serialization/access.hpp>

class PatternData;
class LabelData;
class CRF;

class FeatureGroup {
    public:
        virtual size_t NumFeatures() const = 0;
        virtual std::vector<FVAL> Psi(const PatternData& p, const LabelData& l) const = 0;
        virtual void AddToCRF(CRF& c, const PatternData& p, double* w) const = 0;
        typedef std::vector<std::pair<std::vector<std::pair<size_t, double>>, double>> Constr;
        virtual Constr CollectConstrs(size_t base, double constraint_scale) const { return Constr(); }
        virtual double Violation(size_t base, double* w) const { return 0.0; }
    private:
        friend class boost::serialization::access;
        template <typename Archive>
        void serialize(const Archive& ar, const unsigned int version) { }
};

std::vector<boost::shared_ptr<FeatureGroup>> GetFeaturesFromParam(STRUCT_LEARN_PARM* sparm);

#endif

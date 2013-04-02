#ifndef _FEATURE_HPP_
#define _FEATURE_HPP_

extern "C" {
#include "svm_light/svm_common.h"
#include "svm_struct/svm_struct_common.h"
}
#include <vector>
#include <string>
#include <boost/serialization/access.hpp>

class CRF;

template <typename PatternData, typename LabelData>
class FeatureGroup {
    public:
        typedef std::vector<std::pair<std::vector<std::pair<size_t, double>>, double>> Constr;

        virtual size_t NumFeatures() const = 0;
        virtual std::vector<FVAL> Psi(const PatternData& p, const LabelData& l) const = 0;
        virtual void AddToCRF(CRF& c, const PatternData& p, double* w) const = 0;
        virtual Constr CollectConstrs(size_t base, double constraint_scale) const { return Constr(); }
        virtual double Violation(size_t base, double* w) const { return 0.0; }

        virtual void Train(const std::vector<PatternData*>& patterns, const std::vector<LabelData*>& labels) { }
        virtual void LoadTraining(const std::string& train_dir) { }
        virtual void SaveTraining(const std::string& train_dir) const { }

        virtual void Evaluate(const std::vector<PatternData*>& patterns) { }
        virtual void LoadEvaluation(const std::string& train_dir) { }
        virtual void SaveEvaluation(const std::string& train_dir) const { }
    private:
        friend class boost::serialization::access;
        template <typename Archive>
        void serialize(const Archive& ar, const unsigned int version) { }
};

/*
std::vector<boost::shared_ptr<FeatureGroup>> GetFeaturesFromParam(STRUCT_LEARN_PARM* sparm);
void CalcUnaries(PatternData& p);
void CalcDistances(const cv::Mat& tri, cv::Mat& fgdDist, cv::Mat& bgdDist, cv::Mat& closerMap);
*/

#endif

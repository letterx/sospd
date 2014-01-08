#ifndef _SVM_CXX_HPP_
#define _SVM_CXX_HPP_

extern "C" {
#include "svm_light/svm_common.h"
#include "svm_struct/svm_struct_common.h"
}
#include <string>
#include <vector>
#include <memory>
#include "stats.hpp"
#include "feature.hpp"

/* Forward declarations of PatternData and LabelData
 * These will be specified by the user, per-application
 */
class PatternData;
class LabelData;
class Parameters;
class Optimizer;

/* SVM_Cpp_Base: abstract base class providing interface for the svm_c++ api
 *
 * SVM_Struct requires several functions to be defined by the user. This 
 * abstract base class contains the pure-virtual members that must be provided
 * by the user, as well as some impure virtual members where there is a 
 * sensible default. 
 *
 * To use the svm_c++ library, provide a subclass implementing the required
 * functionality, as well as providing definitions of the structs PatternData
 * and LabelData above. User must also implement newUserApplication() to return
 * an instance of their subclass.
 */
class SVM_Cpp_Base {
    public:
        SVM_Cpp_Base() : m_testStats() { }
        virtual ~SVM_Cpp_Base() { }

        typedef std::vector<PatternData*> PatternVec;
        typedef std::vector<LabelData*> LabelVec;
        typedef std::vector<std::unique_ptr<FeatureGroup>> FeatureVec;

        /* Must be defined by the user.
         * This function should allocate the user-defined subclass.
         */
        static std::unique_ptr<SVM_Cpp_Base> newUserApplication();

        /* 
         * Reads examples from a file at fname, allocating patterns and labels
         * outputting them in the provided vectors.
         */
        virtual void readExamples(const std::string& fname,
                PatternVec& patterns, LabelVec& labels) = 0;

        /*
         * Perform any necessary initialization for the features
         */
        virtual void initFeatures(const Parameters& params) = 0;

        /*
         * Return the features used by the application
         */
        virtual const FeatureVec& features() const = 0;

        /*
         * Classify a given pattern, according to the current parameter 
         * vector w.
         */
        virtual LabelData* classify(const PatternData& p, 
                const double* w) const = 0;

        /* 
         * Given a pattern p and correct label l, find the most violated
         * constraint according to the current parameter vector w.
         */
        virtual LabelData* findMostViolatedConstraint(const PatternData& p, 
                const LabelData& l, const double* w) const = 0;

        /* 
         * Calculate the loss between two labels.
         */
        virtual double loss(const LabelData& l1, const LabelData& l2) const = 0;

        virtual bool finalizeIteration() const = 0;
        virtual void evalPrediction(const PatternData& p, 
                const LabelData& y, const LabelData& ypred) const = 0;
        virtual const Parameters& params() const = 0;

        long numFeatures() const;
        void trainFeatures(const std::string& train_file, 
                const std::string& eval_file, 
                const std::string& output_dir);

        TestStats m_testStats;

    private:
        /*
         * If the user-defined subclass has any data members that need saving,
         * it must also implement serialize, and export itself to the
         * serialization library with BOOST_CLASS_EXPORT_GUID, so that we can
         * serialize it from a pointer-to-base-class.
         */
        friend class boost::serialization::access;
        template <typename Archive>
        void serialize(Archive& ar, unsigned int version) {
            ar & m_testStats;
        }
};

extern std::unique_ptr<SVM_Cpp_Base> g_application;

#endif


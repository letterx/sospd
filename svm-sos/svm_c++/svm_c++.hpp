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

/* 
 * Forward declarations of PatternData and LabelData
 * These will be specified by the user, per-application
 */
struct PatternData;
struct LabelData;
struct Optimizer;

/*
 * Other forward declarations
 */
namespace boost { namespace program_options { class options_description; } }
class FeatureGroup;

/* 
 * SVM_Cpp_Base: abstract base class providing interface for the svm_c++ api
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

        /* 
         * This function allocates the user-defined subclass.
         * MUST be defined by the user.
         */
        static std::unique_ptr<SVM_Cpp_Base> newUserApplication();

        /*
         * These 2 structs allow us to use std::unique_ptr with these types.
         *
         * Because we have forward declared PatternData and LabelData,
         * we don't have access to their destructors, so can't delete them.
         * Definitions of operator() must be provided by the user. In most 
         * cases, simply calling the macro SVM_CPP_DEFINE_DEFAULT_DELETERS
         * at any point after where PatternData and LabelData are complete
         * will suffice.
         */
        struct PatternDeleter {
            void operator()(PatternData*) const;
        };
        struct LabelDeleter {
            void operator()(LabelData*) const;
        };
#define SVM_CPP_DEFINE_DEFAULT_DELETERS \
        void SVM_Cpp_Base::PatternDeleter::operator()(PatternData* p) const { delete p; } \
        void SVM_Cpp_Base::LabelDeleter::operator()(LabelData* p) const { delete p; }

        typedef std::unique_ptr<PatternData, PatternDeleter> PatternPtr;
        typedef std::unique_ptr<LabelData, LabelDeleter> LabelPtr;
        typedef std::unique_ptr<FeatureGroup> FeaturePtr;
        typedef std::vector<PatternPtr> PatternVec;
        typedef std::vector<LabelPtr> LabelVec;
        typedef std::vector<FeaturePtr> FeatureVec;

        /* 
         * Reads examples from a file at fname, allocating patterns and labels
         * outputting them in the provided vectors.
         */
        virtual void readExamples(const std::string& fname,
                PatternVec& patterns, LabelVec& labels) = 0;

        /*
         * Perform any necessary initialization for the features
         */
        virtual void initFeatures() = 0;

        /*
         * Return the features used by the application
         */
        virtual const FeatureVec& features() const = 0;

        /*
         * Classify a given pattern, according to the current parameter 
         * vector w.
         */
        virtual LabelPtr classify(const PatternData& p, 
                const double* w) const = 0;

        /* 
         * Given a pattern p and correct label l, find the most violated
         * constraint according to the current parameter vector w.
         */
        virtual LabelPtr findMostViolatedConstraint(const PatternData& p, 
                const LabelData& l, const double* w) const = 0;

        /* 
         * Calculate the loss between two labels.
         */
        virtual double loss(const LabelData& l1, const LabelData& l2) const = 0;

        virtual bool finalizeIteration() const = 0;
        virtual void evalPrediction(const PatternData& p, 
                const LabelData& y, const LabelData& ypred) const = 0;

        long numFeatures() const;
        void trainFeatures(const std::string& train_file, 
                const std::string& eval_file, 
                const std::string& output_dir);

        /*
         * Program options 
         */
        boost::program_options::options_description
            getBaseLearnParams();
        void parseBaseLearnParams(int argc, char** argv);
        void printLearnHelp();
        boost::program_options::options_description
            getBaseClassifyParams();
        void parseBaseClassifyParams(int argc, char** argv);
        void printClassifyHelp();

        virtual boost::program_options::options_description
            getLearnParams() = 0;
        virtual void parseLearnParams(const std::vector<std::string>& args) = 0;
        virtual boost::program_options::options_description
            getClassifyParams() = 0;
        virtual void parseClassifyParams(const std::vector<std::string>& args) = 0;


        TestStats m_testStats;
        std::string m_statsFile;

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

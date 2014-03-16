#ifndef _INTERACTIVE_SEG_APP_HPP_
#define _INTERACTIVE_SEG_APP_HPP_

#include "svm_c++.hpp"
#include "feature.hpp"
#include "gmm.hpp"
#include "crf.hpp"

#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/program_options.hpp>

struct PatternData {
    public:
        PatternData(const std::string& name, const cv::Mat& im, const cv::Mat& tri);
        cv::Mat m_image;
        cv::Mat m_tri;
        std::string m_name;

        const std::string& Name() const { return m_name; }
};

struct LabelData {
    public:
        LabelData() = default;
        LabelData(const std::string& name) : m_name(name) { }
        LabelData(const std::string& name, const cv::Mat& gt);

        bool operator==(const LabelData& l) const;

        cv::Mat m_gt;
        std::string m_name;

        const std::string& Name() const { return m_name; }
};

struct Optimizer : public CRF { };

class InteractiveSegApp : public SVM_Cpp_Base {
    public:
        struct Parameters {
            // The following are saved/loaded in model serialization
            std::string eval_dir;
            bool all_features;
            int grabcut_classify;
            int grabcut_unary;
            bool distance_unary;
            bool color_patch;
            bool pairwise_feature;
            bool contrast_pairwise_feature;
            bool submodular_feature;
            bool contrast_submodular_feature;
            // These parameters are classify-specific
            bool show_images;
            std::string output_dir;
            std::string stats_file;
            int crf;

            unsigned int Version() const { return 1; }
        };
        template <typename Archive>
        void SerializeParams(Archive& ar, const unsigned int version) {
            ar & m_params.eval_dir;
            ar & m_params.all_features;
            if (version < 1) {
                bool dummy_grabcut_classify;
                ar & dummy_grabcut_classify;
            }
            ar & m_params.grabcut_unary;
            ar & m_params.distance_unary;
            ar & m_params.color_patch;
            ar & m_params.pairwise_feature;
            ar & m_params.contrast_pairwise_feature;
            ar & m_params.submodular_feature;
            ar & m_params.contrast_submodular_feature;
        }

        InteractiveSegApp() = default;
        InteractiveSegApp(const Parameters& params);
        virtual void readExamples(const std::string& file, PatternVec& patterns, LabelVec& labels) override;
        virtual void initFeatures() override;
        virtual const FeatureVec& features() const { return m_features; }
        virtual LabelPtr classify(const PatternData& x, const double* w) const override;
        virtual LabelPtr findMostViolatedConstraint(const PatternData& x, const LabelData& y, const double* w) const override;
        virtual double loss(const LabelData& y, const LabelData& ybar) const override;
        //bool FinalizeIteration(double eps, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const;
        virtual bool finalizeIteration() const override;
        virtual void evalPrediction(const PatternData& x, const LabelData& y, const LabelData& ypred) const override;
        const Parameters& params() const { return m_params; }

        virtual boost::program_options::options_description getLearnParams() override;
        virtual void parseLearnParams(const std::vector<std::string>& args) override;
        virtual boost::program_options::options_description getClassifyParams() override;
        virtual void parseClassifyParams(const std::vector<std::string>& args) override;
    private:
        void initializeCRF(CRF& crf, const PatternData& x) const;
        void AddLossToCRF(CRF& crf, const PatternData& x, const LabelData& y, double scale) const;
        LabelPtr ExtractLabel(const CRF& crf, const PatternData& x) const;
        static boost::program_options::options_description GetCommonOptions();

        Parameters m_params;
        FeatureVec m_features;
};

inline double LabelDiff(unsigned char l1, unsigned char l2) {
    if (l1 == cv::GC_BGD || l1 == cv::GC_PR_BGD) {
        if (l2 == cv::GC_FGD || l2 == cv::GC_PR_FGD)
            return 1.0;
        else if (l2 == (unsigned char)-1)
            return 0.5;
        else 
            return 0.0;
    } else if (l1 == cv::GC_FGD || l1 == cv::GC_PR_FGD) {
        if (l2 == cv::GC_BGD || l2 == cv::GC_PR_BGD)
            return 1.0;
        else if (l2 == (unsigned char)-1)
            return 0.5;
        else 
            return 0.0;
    } else if (l1 == l2)
        return 0.0;
    else
        return 0.5;
}

#endif

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

class public PatternData {
    public:
        PatternData(const std::string& name, const cv::Mat& im, const cv::Mat& tri);
        cv::Mat m_image;
        cv::Mat m_tri;
        std::string m_name;
};

class LabelData {
    public:
        LabelData() = default;
        LabelData(const std::string& name) : m_name(name) { }
        LabelData(const std::string& name, const cv::Mat& gt);

        bool operator==(const LabelData& l) const;

        cv::Mat m_gt;
        std::string m_name;
};

class InteractiveSegApp {
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

        typedef FeatureGroup<PatternData, LabelData, CRF> FG;

        InteractiveSegApp(const Parameters& params);
        void InitFeatures(const Parameters& p);
        void ReadExamples(const std::string& file, std::vector<PatternData*>& patterns, std::vector<LabelData*>& labels);
        long NumFeatures() const;
        const std::vector<boost::shared_ptr<FG>>& Features() const { return m_features; }
        LabelData* Classify(const PatternData& x, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const;
        LabelData* FindMostViolatedConstraint(const PatternData& x, const LabelData& y, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const;
        double Loss(const LabelData& y, const LabelData& ybar, double loss_scale) const;
        void EvalPrediction(const PatternData& x, const LabelData& y, const LabelData& ypred) const;
        bool FinalizeIteration(double eps, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const;
        const Parameters& Params() const { return m_params; }

        static boost::program_options::options_description GetLearnOptions();
        static boost::program_options::options_description GetClassifyOptions();
        static Parameters ParseLearnOptions(const std::vector<std::string>& args);
        static Parameters ParseClassifyOptions(const std::vector<std::string>& args);
    private:
        void InitializeCRF(CRF& crf, const PatternData& x) const;
        void AddLossToCRF(CRF& crf, const PatternData& x, const LabelData& y, double scale) const;
        LabelData* ExtractLabel(const CRF& crf, const PatternData& x) const;
        static boost::program_options::options_description GetCommonOptions();

        Parameters m_params;
        std::vector<boost::shared_ptr<FG>> m_features;
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

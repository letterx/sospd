#ifndef _BINARY_SEG_APP_HPP_
#define _BINARY_SEG_APP_HPP_

#include "svm_c++.hpp"
#include "feature.hpp"
#include "gmm.hpp"
#include "crf.hpp"

#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/program_options.hpp>

class BS_PatternData : public PatternData {
    public:
        BS_PatternData(const std::string& name, const cv::Mat& im, const cv::Mat& tri);
        cv::Mat m_image;
        /*
        double m_beta;
        cv::Mat m_fgdUnaries;
        cv::Mat m_bgdUnaries;
        cv::Mat m_downW;
        cv::Mat m_rightW;
        cv::Mat m_fgdDist;
        cv::Mat m_bgdDist;
        cv::Mat m_dist_feature;
        */
};

class BS_LabelData : public LabelData{
    public:
        BS_LabelData() = default;
        BS_LabelData(const std::string& name) : LabelData(name) { }
        BS_LabelData(const std::string& name, const cv::Mat& gt);

        bool operator==(const BS_LabelData& l) const;

        cv::Mat m_gt;
};

class BinarySegApp;

template <>
struct AppTraits<BinarySegApp> {
    typedef BS_PatternData PatternData;
    typedef BS_LabelData LabelData;
    typedef FeatureGroup<BS_PatternData, BS_LabelData, CRF> FG;
};

class BinarySegApp : public SVM_App<BinarySegApp> {
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

        typedef FeatureGroup<BS_PatternData, BS_LabelData, CRF> FG;

        BinarySegApp(const Parameters& params);
        void InitFeatures(const Parameters& p);
        void ReadExamples(const std::string& file, std::vector<BS_PatternData*>& patterns, std::vector<BS_LabelData*>& labels);
        long NumFeatures() const;
        const std::vector<boost::shared_ptr<FG>>& Features() const { return m_features; }
        BS_LabelData* Classify(const BS_PatternData& x, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const;
        BS_LabelData* FindMostViolatedConstraint(const BS_PatternData& x, const BS_LabelData& y, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const;
        double Loss(const BS_LabelData& y, const BS_LabelData& ybar, double loss_scale) const;
        void EvalPrediction(const BS_PatternData& x, const BS_LabelData& y, const BS_LabelData& ypred) const;
        bool FinalizeIteration(double eps, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const;
        const Parameters& Params() const { return m_params; }

        static boost::program_options::options_description GetLearnOptions();
        static boost::program_options::options_description GetClassifyOptions();
        static Parameters ParseLearnOptions(const std::vector<std::string>& args);
        static Parameters ParseClassifyOptions(const std::vector<std::string>& args);
    private:
        void InitializeCRF(CRF& crf, const BS_PatternData& x) const;
        void AddLossToCRF(CRF& crf, const BS_PatternData& x, const BS_LabelData& y, double scale) const;
        BS_LabelData* ExtractLabel(const CRF& crf, const BS_PatternData& x) const;
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

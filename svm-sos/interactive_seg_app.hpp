#ifndef _INTERACTIVE_SEG_APP_HPP_
#define _INTERACTIVE_SEG_APP_HPP_

#include "svm_c++.hpp"
#include "feature.hpp"
#include "gmm.hpp"

#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>

class IS_PatternData : public PatternData {
    public:
        IS_PatternData(const std::string& name, const cv::Mat& im, const cv::Mat& tri);
        cv::Mat m_image;
        cv::Mat m_tri;
        /*
        cv::Mat m_bgdModel;
        GMM m_bgdGMM;
        cv::Mat m_fgdModel;
        GMM m_fgdGMM;
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

class IS_LabelData : public LabelData{
    public:
        IS_LabelData() = default;
        IS_LabelData(const std::string& name) : LabelData(name) { }
        IS_LabelData(const std::string& name, const cv::Mat& gt);

        bool operator==(const IS_LabelData& l) const;

        cv::Mat m_gt;
};

class InteractiveSegApp;

template <>
struct AppTraits<InteractiveSegApp> {
    typedef IS_PatternData PatternData;
    typedef IS_LabelData LabelData;
    typedef FeatureGroup<IS_PatternData, IS_LabelData> FG;
};

class CRF;

class InteractiveSegApp : public SVM_App<InteractiveSegApp> {
    public:
        struct Parameters {
            bool show_images;
            std::string output_dir;
        };

        typedef FeatureGroup<IS_PatternData, IS_LabelData> FG;

        InteractiveSegApp() : SVM_App<InteractiveSegApp>(this) { }
        void ReadExamples(const std::string& file, std::vector<PatternData*>& patterns, std::vector<LabelData*>& labels);
        long NumFeatures() const;
        const std::vector<boost::shared_ptr<FG>>& Features() const { return m_features; }
        IS_LabelData* Classify(const IS_PatternData& x, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const;
        IS_LabelData* FindMostViolatedConstraint(const IS_PatternData& x, const IS_LabelData& y, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const;
        double Loss(const IS_LabelData& y, const IS_LabelData& ybar, double loss_scale) const;
        void EvalPrediction(const IS_PatternData& x, const IS_LabelData& y, const IS_LabelData& ypred) const;
    private:
        void InitFeatures(const Parameters& p);
        void InitializeCRF(CRF& crf, const IS_PatternData& x) const;
        void AddLossToCRF(CRF& crf, const IS_PatternData& x, const IS_LabelData& y, double scale) const;
        IS_LabelData* ExtractLabel(const CRF& crf, const IS_PatternData& x) const;

        Parameters m_params;
        std::vector<boost::shared_ptr<FG>> m_features;
};

#endif

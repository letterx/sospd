#ifndef _SEMANTIC_SEG_APP_HPP_
#define _SEMANTIC_SEG_APP_HPP_

#include "svm_c++.hpp"
#include "feature.hpp"
#include "alpha-expansion.hpp"

#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/program_options.hpp>

class Sem_PatternData : public PatternData {
    public:
        Sem_PatternData(const std::string& name, const cv::Mat& im);
        cv::Mat m_image;
};

class Sem_LabelData : public LabelData{
    public:
        Sem_LabelData() = default;
        Sem_LabelData(const std::string& name) : LabelData(name) { }
        Sem_LabelData(const std::string& name, const cv::Mat& gt);

        bool operator==(const Sem_LabelData& l) const;

        cv::Mat m_gt;
};

class SemanticSegApp;

template <>
struct AppTraits<SemanticSegApp> {
    typedef Sem_PatternData PatternData;
    typedef Sem_LabelData LabelData;
    typedef FeatureGroup<Sem_PatternData, Sem_LabelData, MultiLabelCRF> FG;
};

class MultiLabelCRF;

class SemanticSegApp : public SVM_App<SemanticSegApp> {
    public:
        struct Parameters {
            bool show_images;
            std::string eval_dir;
            std::string output_dir;
            std::string stats_file;
            bool all_features;
            int crf;
            bool pairwise_feature;
            bool contrast_pairwise_feature;
            bool submodular_feature;
            bool contrast_submodular_feature;
        };

        typedef FeatureGroup<Sem_PatternData, Sem_LabelData, MultiLabelCRF> FG;

        SemanticSegApp(const Parameters& params);
        void ReadExamples(const std::string& file, std::vector<Sem_PatternData*>& patterns, std::vector<Sem_LabelData*>& labels);
        long NumFeatures() const;
        const std::vector<boost::shared_ptr<FG>>& Features() const { return m_features; }
        Sem_LabelData* Classify(const Sem_PatternData& x, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const;
        Sem_LabelData* FindMostViolatedConstraint(const Sem_PatternData& x, const Sem_LabelData& y, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const;
        double Loss(const Sem_LabelData& y, const Sem_LabelData& ybar, double loss_scale) const;
        void EvalPrediction(const Sem_PatternData& x, const Sem_LabelData& y, const Sem_LabelData& ypred) const;
        bool FinalizeIteration(STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const;

        static boost::program_options::options_description GetLearnOptions();
        static boost::program_options::options_description GetClassifyOptions();
        static Parameters ParseLearnOptions(const std::vector<std::string>& args);
        static Parameters ParseClassifyOptions(const std::vector<std::string>& args);
        const Parameters& Params() const { return m_params; }
    private:
        typedef int Label;
        void InitFeatures(const Parameters& p);
        void InitializeCRF(MultiLabelCRF& crf, const Sem_PatternData& x) const;
        void AddLossToCRF(MultiLabelCRF& crf, const Sem_PatternData& x, const Sem_LabelData& y, double scale) const;
        Sem_LabelData* ExtractLabel(const MultiLabelCRF& crf, const Sem_PatternData& x) const;
        void ConvertColorToLabel(const cv::Mat& color_image, cv::Mat& label_image) const;
        void ConvertLabelToColor(const cv::Mat& label_image, cv::Mat& color_image) const;
        void ValidateExample(const cv::Mat& image, const cv::Mat& gt) const;
        static boost::program_options::options_description GetCommonOptions();

        Parameters m_params;
        std::vector<boost::shared_ptr<FG>> m_features;
        std::map<cv::Vec3b, Label> m_label_map;
        std::vector<cv::Vec3b> m_color_vec;
        Label m_num_labels;
};

#endif

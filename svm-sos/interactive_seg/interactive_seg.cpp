#include <cmath>
#include <fstream>
#include "interactive_seg.hpp"
#include "svm_c++.hpp"
#include "image_manip.hpp"
#include "crf.hpp"
#include "feature.hpp"
#include "submodular-feature.hpp"
#include "gmm-feature.hpp"
#include "distance-feature.hpp"
#include "pairwise-feature.hpp"
#include "contrast-submodular.hpp"
#include "color-patch-feature.hpp"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

InteractiveSegApp::InteractiveSegApp(const Parameters& params) 
: m_params(params) 
{ }

void InteractiveSegApp::ReadExamples(const std::string& file, std::vector<PatternData*>& patterns, std::vector<LabelData*>& labels) {
std::ifstream main_file(file);

std::string images_dir;
std::string trimap_dir;
std::string gt_dir;
std::string line;

size_t n = 0;

do {
    std::getline(main_file, images_dir);
} while (images_dir[0] == '#');
do {
    std::getline(main_file, trimap_dir);
} while (trimap_dir[0] == '#');
do {
    std::getline(main_file, gt_dir);
} while (gt_dir[0] == '#');
          
while (main_file.good()) {
    std::getline(main_file, line);
    if (!line.empty() && line[0] != '#') {
        n++;
        if (n % 10 == 0) {
            std::cout << ".";
            std::cout.flush();
        }
        cv::Mat image = cv::imread(images_dir + line, CV_LOAD_IMAGE_COLOR);
        cv::Mat trimap = cv::imread(trimap_dir + line, CV_LOAD_IMAGE_COLOR);
        cv::Mat gt = cv::imread(gt_dir + line, CV_LOAD_IMAGE_GRAYSCALE);
        ValidateExample(image, trimap, gt);
        patterns.push_back(new PatternData(line, image, trimap));
        labels.push_back(new LabelData(line, gt));
    }
}
main_file.close();
}


PatternData::PatternData(const std::string& name, const cv::Mat& image, const cv::Mat& trimap) 
: m_name(name)
    , m_image(image)
{
    ConvertToMask(trimap, m_tri);
}

LabelData::LabelData(const std::string& name, const cv::Mat& gt)
    : LabelData(name)
{ 
    ConvertGreyToMask(gt, m_gt);
}

bool LabelData::operator==(const LabelData& l) const {
    ASSERT(m_gt.size() == l.m_gt.size());
    cv::Point pt;
    for (pt.y = 0; pt.y < m_gt.rows; ++pt.y) {
        for (pt.x = 0; pt.x < m_gt.cols; ++pt.x) {
            if (m_gt.at<unsigned char>(pt) != l.m_gt.at<unsigned char>(pt))
                return false;
        }
    }
    return true;
}

double InteractiveSegApp::Loss(const LabelData& l1, const LabelData& l2, double scale) const {
    ASSERT(l1.m_gt.size() == l2.m_gt.size());
    double loss = 0;
    cv::Point pt;
    for (pt.y = 0; pt.y < l1.m_gt.rows; ++pt.y) {
        for (pt.x = 0; pt.x < l1.m_gt.cols; ++pt.x) {
            loss += LabelDiff(l1.m_gt.at<unsigned char>(pt), l2.m_gt.at<unsigned char>(pt));
        }
    }
    loss /= (l1.m_gt.rows*l1.m_gt.cols);
    return loss*scale;
}


void InteractiveSegApp::InitFeatures(const Parameters& param) {
    std::cout << "\nFeatures: ";
    constexpr double feature_scale = 0.01;
    m_features.push_back(boost::shared_ptr<FeatureGroup>(new GMMFeature(feature_scale, param.grabcut_unary)));
    std::cout << "GMMFeature ";
    if (param.distance_unary || param.all_features) {
        m_features.push_back(boost::shared_ptr<FeatureGroup>(new DistanceFeature(feature_scale)));
        std::cout << "DistanceFeature ";
    }
    if (param.color_patch || param.all_features) {
        m_features.push_back(boost::shared_ptr<FeatureGroup>(new ColorPatchFeature(feature_scale)));
        std::cout << "ColorPatchFeature ";
    }
    if (param.submodular_feature || param.all_features) {
        m_features.push_back(boost::shared_ptr<FeatureGroup>(new SubmodularFeature(feature_scale)));
        std::cout << "SubmodularFeature ";
    }
    if (param.pairwise_feature || param.all_features) {
        m_features.push_back(boost::shared_ptr<FeatureGroup>(new PairwiseFeature(feature_scale)));
        std::cout << "PairwiseFeature ";
    }
    if (param.contrast_pairwise_feature || param.all_features) {
        m_features.push_back(boost::shared_ptr<FeatureGroup>(new ContrastPairwiseFeature(feature_scale)));
        std::cout << "ContrastPairwiseFeature ";
    }
    if (param.contrast_submodular_feature || param.all_features) {
        m_features.push_back(boost::shared_ptr<FeatureGroup>(new ContrastSubmodularFeature(feature_scale)));
        std::cout << "ContrastSubmodularFeature ";
    }
    std::cout << "\n";

    if (param.eval_dir != std::string("")) {
        for (auto fp : m_features) {
            fp->LoadEvaluation(param.eval_dir);
        }
    }
}

long InteractiveSegApp::NumFeatures() const {
    long n = 0;
    for (auto fgp : m_features) {
        n += fgp->NumFeatures();
    }
    return n;
}

void InteractiveSegApp::InitializeCRF(CRF& crf, const PatternData& p) const {
    crf.AddNode(p.m_image.rows*p.m_image.cols);

}

void InteractiveSegApp::AddLossToCRF(CRF& crf, const PatternData& p, const LabelData& l, double scale) const {
    double mult = scale/(p.m_image.rows*p.m_image.cols);
    cv::Point pt;
    for (pt.y = 0; pt.y < p.m_image.rows; ++pt.y) {
        for (pt.x = 0; pt.x < p.m_image.cols; ++pt.x) {
            CRF::NodeId id = pt.y * p.m_image.cols + pt.x;
            double E0 = 0;
            double E1 = 0;
            if (l.m_gt.at<unsigned char>(pt) == cv::GC_BGD) E1 -= 1.0*mult;
            if (l.m_gt.at<unsigned char>(pt) == cv::GC_FGD) E0 -= 1.0*mult;
            crf.AddUnaryTerm(id, doubleToREAL(E0), doubleToREAL(E1));
        }
    }
}

LabelData* InteractiveSegApp::ExtractLabel(const CRF& crf, const PatternData& x) const {
    LabelData* lp = new LabelData(x.Name());
    lp->m_gt.create(x.m_image.rows, x.m_image.cols, CV_8UC1);
    CRF::NodeId id = 0;
    ImageIterate(lp->m_gt, 
        [&](unsigned char& c) { 
            //ASSERT(crf.GetLabel(id) >= 0);
            if (crf.GetLabel(id) == 0) c = cv::GC_BGD;
            else if (crf.GetLabel(id) == 1) c = cv::GC_FGD;
            else c = -1;
            id++;
        });
    return lp;
}

LabelData* InteractiveSegApp::Classify(const PatternData& x, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const {
    if (m_params.grabcut_classify) {
        const_cast<std::string&>(this->m_test_stats.m_model_file) = std::string("grabcut.model");
        LabelData* y = new LabelData(x.Name());
        x.m_tri.copyTo(y->m_gt);
        cv::Mat bgdModel;
        cv::Mat fgdModel;
        cv::grabCut(x.m_image, y->m_gt, cv::Rect(), bgdModel, fgdModel, m_params.grabcut_classify);
        return y;
    } else {
        CRF crf;
        SubmodularFlow sf;
        SubmodularIBFS ibfs;
        HigherOrderWrapper ho;
        if (m_params.crf == 0) {
            crf.Wrap(&sf);
        } else if (m_params.crf == 2) {
            crf.Wrap(&ibfs);
        } else {
            crf.Wrap(&ho);
        }
        InitializeCRF(crf, x);
        size_t feature_base = 1;
        for (auto fgp : m_features) {
            fgp->AddToCRF(crf, x, sm->w + feature_base );
            feature_base += fgp->NumFeatures();
        }
        crf.Solve();
        return ExtractLabel(crf, x);
    }
}

LabelData* InteractiveSegApp::FindMostViolatedConstraint(const PatternData& x, const LabelData& y, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const {
    CRF crf;
    SubmodularFlow sf;
    HigherOrderWrapper ho;
    SubmodularIBFS ibfs;
    if (m_params.crf == 0) {
        crf.Wrap(&sf);
    } else if (m_params.crf == 2) {
        crf.Wrap(&ibfs);
    } else {
        crf.Wrap(&ho);
    }
    InitializeCRF(crf, x);
    size_t feature_base = 1;
    for (auto fgp : m_features) {
        fgp->AddToCRF(crf, x, sm->w + feature_base );
        feature_base += fgp->NumFeatures();
    }
    AddLossToCRF(crf, x, y, sparm->loss_scale);
    crf.Solve();
    return ExtractLabel(crf, x);
}

void InteractiveSegApp::EvalPrediction(const PatternData& x, const LabelData& y, const LabelData& ypred) const {
    const std::string& name = ypred.Name();
    cv::Mat color_image = MaskToColor(ypred.m_gt, x.m_image);
    if (m_params.show_images)
        ShowImage(color_image);

    if (m_params.output_dir != std::string("")) {
        std::string out_filename = m_params.output_dir + "/" + name;
        cv::imwrite(out_filename, (ypred.m_gt)*(255/3));
        out_filename = m_params.output_dir + "/fancy-" + name;
        cv::Mat write_image;
        color_image.convertTo(write_image, CV_8U, 255.0);
        cv::imwrite(out_filename, write_image);
    }
}

bool InteractiveSegApp::FinalizeIteration(double eps, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const {
    size_t feature_base = 1;
    for (auto fgp : m_features) {
        double violation = fgp->Violation(feature_base, sm->w);
        double w2 = 0;
        for (size_t i = feature_base; i < feature_base + fgp->NumFeatures(); ++i) 
            w2 += sm->w[i] * sm->w[i];
        if (violation > 0.001 * w2) {
            if (eps < sparm->epsilon)
                sparm->epsilon *= 0.49999;
            std::cout << "Forcing algorithm to continue: Max Violation = " << violation << ", |w|^2 = " << w2 << " New eps = " << sparm->epsilon << "\n";
            return true;
        }
        feature_base += fgp->NumFeatures();
    }
    return false;
}

namespace po = boost::program_options;

po::options_description InteractiveSegApp::GetCommonOptions() {
    po::options_description desc("Interactive Segmentation Options");
    desc.add_options()
        ("crf", po::value<std::string>(), "[ho | sf] -> Set CRF optimizer. (default sf)")
        ("stats-file", po::value<std::string>(), "Output file for statistics")
        ("eval-dir", po::value<std::string>(), "Directory for feature evaluation caching")
    ;

    return desc;
}

po::options_description InteractiveSegApp::GetLearnOptions() {
    po::options_description desc = GetCommonOptions();
    desc.add_options()
        ("all-features", po::value<bool>(), "Turn on all features (for use with feature-train)")
        ("grabcut-unary", po::value<int>(), "[0..] Use n iterations of grabcut to initialize GMM unary features (default 0)")
        ("distance-unary", po::value<int>(), "[0,1] If 1, use distance features for unary potentials")
        ("color-patch", po::value<bool>(), "[0,1] If 1, use color patch features for unary potentials")
        ("pairwise", po::value<int>(), "[0, 1] -> Use pairwise edge features. (default 0)")
        ("contrast-pairwise", po::value<int>(), "[0, 1] -> Use contrast-sensitive pairwise features. (default 0)")
        ("submodular", po::value<int>(), "[0, 1] -> Use submodular features. (default 0)")
        ("contrast-submodular", po::value<bool>(), "[0, 1] -> Use contrast-submodular features. (default 1)")
    ;
    return desc;
}

po::options_description InteractiveSegApp::GetClassifyOptions() {
    po::options_description desc = GetCommonOptions();
    desc.add_options()
        ("grabcut", po::value<int>(), "[0..] -> If nonzero, run n iterations of grabcut as the classifier instead. (default 0)")
        ("show", po::value<int>(), "[0,1] -> If nonzero, display each image after it is classified. (default 0)")
        ("output-dir", po::value<std::string>(), "Write predicted images to directory.")
    ;
    return desc;
}


InteractiveSegApp::Parameters InteractiveSegApp::ParseLearnOptions(const std::vector<std::string>& args) {
    Parameters params;
    params.all_features = false;
    params.grabcut_classify = 0;
    params.crf = 0;
    params.grabcut_unary = 0;
    params.distance_unary = 1;
    params.color_patch = 1;
    params.pairwise_feature = 0;
    params.contrast_pairwise_feature = 0;
    params.submodular_feature = 0;
    params.contrast_submodular_feature = 1;
    params.stats_file = std::string();

    po::options_description desc = GetLearnOptions();
    po::variables_map vm;
    po::store(po::command_line_parser(args).options(desc).run(), vm);

    if (vm.count("crf")) {
        std::string type = vm["crf"].as<std::string>();
        if (type == "sf") {
            std::cout << "SubmodularFlow optimizer\n";
            params.crf = 0;
        } else if (type == "ho") {
            std::cout << "HigherOrder optimizer\n";
            params.crf = 1;
        } else if (type == "ibfs") {
            std::cout << "SubmodularIBFS optimizer\n";
            params.crf = 2;
        } else {
            std::cout << "Unrecognized optimizer\n";
            exit(-1);
        }
    }
    if (vm.count("grabcut-unary")) 
        params.grabcut_unary = vm["grabcut-unary"].as<int>();
    if (vm.count("distance-unary"))
        params.distance_unary = vm["distance-unary"].as<int>();
    if (vm.count("color-patch"))
        params.color_patch = vm["color-patch"].as<bool>();
    if (vm.count("pairwise")) {
        params.pairwise_feature = vm["pairwise"].as<int>();
        std::cout << "Pairwise Feature = " << params.pairwise_feature << "\n";
    }
    if (vm.count("contrast-pairwise")) {
        params.contrast_pairwise_feature = vm["contrast-pairwise"].as<int>();
        std::cout << "Contrast Pairwise Feature = " << params.contrast_pairwise_feature << "\n";
    }
    if (vm.count("submodular")) {
        params.submodular_feature = vm["submodular"].as<int>();
        std::cout << "Submodular Feature = " << params.submodular_feature << "\n";
    }
    if (vm.count("contrast-submodular")) {
        params.contrast_submodular_feature = vm["contrast-submodular"].as<bool>();
    }
    if (vm.count("stats-file")) {
        params.stats_file = vm["stats-file"].as<std::string>();
    }
    if (vm.count("eval-dir")) {
        params.eval_dir = vm["eval-dir"].as<std::string>();
    }
    if (vm.count("all-features"))
        params.all_features = vm["all-features"].as<bool>();
    return params;
}

InteractiveSegApp::Parameters InteractiveSegApp::ParseClassifyOptions(const std::vector<std::string>& args) {
    Parameters params;

    params.show_images = false;
    params.grabcut_classify = 0;
    params.crf = 0;
    params.grabcut_unary = 0;
    params.output_dir = std::string();
    params.stats_file = std::string();

    po::options_description desc = GetClassifyOptions();
    po::variables_map vm;
    po::store(po::command_line_parser(args).options(desc).run(), vm);

    if (vm.count("crf")) {
        std::string type = vm["crf"].as<std::string>();
        if (type == "sf") {
            std::cout << "SubmodularFlow optimizer\n";
            params.crf = 0;
        } else if (type == "ho") {
            std::cout << "HigherOrder optimizer\n";
            params.crf = 1;
        } else if (type == "ibfs") {
            std::cout << "SubmodularIBFS optimizer\n";
            params.crf = 2;
        } else {
            std::cout << "Unrecognized optimizer\n";
            exit(-1);
        }
    }
    if (vm.count("show")) {
        params.show_images = vm["show"].as<int>();
        std::cout << "Show Images = " << params.show_images << "\n";
    }
    if (vm.count("grabcut")) {
        params.grabcut_classify = vm["grabcut"].as<int>();
        std::cout << "Grabcut iterations = " << params.grabcut_classify << "\n";
    }
    if (vm.count("output-dir")) {
        params.output_dir = vm["output-dir"].as<std::string>();
    }
    if (vm.count("stats-file")) {
        params.stats_file = vm["stats-file"].as<std::string>();
    }
    if (vm.count("eval-dir")) {
        params.eval_dir = vm["eval-dir"].as<std::string>();
    }
    return params;
}

#include <cmath>
#include <fstream>
#include "binary_seg_app.hpp"
#include "svm_c++.hpp"
#include "image_manip.hpp"
#include "crf.hpp"
#include "feature.hpp"
#include "binary-unary-feature.hpp"
#include "binary-submodular-feature.hpp"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <opencv2/imgproc/imgproc.hpp>

BinarySegApp::BinarySegApp(const Parameters& params) 
    : SVM_App<BinarySegApp>(this),
    m_params(params) 
{ }

void BinarySegApp::ReadExamples(const std::string& file, std::vector<BS_PatternData*>& patterns, std::vector<BS_LabelData*>& labels) {
    std::ifstream main_file(file);

    std::string images_dir;
    std::string gt_dir;
    std::string line;

    size_t n = 0;

    do {
        std::getline(main_file, images_dir);
    } while (images_dir[0] == '#');
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
            cv::Mat image = cv::imread(images_dir + line, CV_LOAD_IMAGE_GRAYSCALE);
            cv::Mat gt = cv::imread(gt_dir + line, CV_LOAD_IMAGE_GRAYSCALE);
            ValidateExample(image, gt);
            patterns.push_back(new BS_PatternData(line, image));
            labels.push_back(new BS_LabelData(line, gt));
        }
    }
    main_file.close();
}


BS_PatternData::BS_PatternData(const std::string& name, const cv::Mat& image) 
    : PatternData(name), m_image(image)
{ }

BS_LabelData::BS_LabelData(const std::string& name, const cv::Mat& gt)
    : LabelData(name), m_gt(gt)
{ }

bool BS_LabelData::operator==(const BS_LabelData& l) const {
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

double BinarySegApp::Loss(const BS_LabelData& l1, const BS_LabelData& l2, double scale) const {
    ASSERT(l1.m_gt.size() == l2.m_gt.size());
    double loss = 0;
    cv::Point pt;
    for (pt.y = 0; pt.y < l1.m_gt.rows; ++pt.y) {
        for (pt.x = 0; pt.x < l1.m_gt.cols; ++pt.x) {
            if (l1.m_gt.at<unsigned char>(pt) != l2.m_gt.at<unsigned char>(pt))
                loss += 1.0;
        }
    }
    loss /= (l1.m_gt.rows*l1.m_gt.cols);
    return loss*scale;
}


void BinarySegApp::InitFeatures(const Parameters& param) {
    std::cout << "\nFeatures: ";
    constexpr double feature_scale = 0.01;
    m_features.push_back(boost::shared_ptr<FG>(new BinaryUnaryFeature(feature_scale)));
    std::cout << "UnaryFeature ";
    m_features.push_back(boost::shared_ptr<FG>(new BinarySubmodularFeature(feature_scale)));
    std::cout << "SubmodularFeature ";
    /*
    m_features.push_back(boost::shared_ptr<FG>(new GMMFeature(feature_scale, param.grabcut_unary)));
    std::cout << "GMMFeature ";
    if (param.distance_unary || param.all_features) {
        m_features.push_back(boost::shared_ptr<FG>(new DistanceFeature(feature_scale)));
        std::cout << "DistanceFeature ";
    }
    if (param.color_patch || param.all_features) {
        m_features.push_back(boost::shared_ptr<FG>(new ColorPatchFeature(feature_scale)));
        std::cout << "ColorPatchFeature ";
    }
    if (param.submodular_feature || param.all_features) {
        m_features.push_back(boost::shared_ptr<FG>(new SubmodularFeature(feature_scale)));
        std::cout << "SubmodularFeature ";
    }
    if (param.pairwise_feature || param.all_features) {
        m_features.push_back(boost::shared_ptr<FG>(new PairwiseFeature(feature_scale)));
        std::cout << "PairwiseFeature ";
    }
    if (param.contrast_pairwise_feature || param.all_features) {
        m_features.push_back(boost::shared_ptr<FG>(new ContrastPairwiseFeature(feature_scale)));
        std::cout << "ContrastPairwiseFeature ";
    }
    if (param.contrast_submodular_feature || param.all_features) {
        m_features.push_back(boost::shared_ptr<FG>(new ContrastSubmodularFeature(feature_scale)));
        std::cout << "ContrastSubmodularFeature ";
    }
    */
    std::cout << "\n";

    if (param.eval_dir != std::string("")) {
        for (auto fp : m_features) {
            fp->LoadEvaluation(param.eval_dir);
        }
    }
}

long BinarySegApp::NumFeatures() const {
    long n = 0;
    for (auto fgp : m_features) {
        n += fgp->NumFeatures();
    }
    return n;
}

void BinarySegApp::InitializeCRF(CRF& crf, const BS_PatternData& p) const {
    crf.AddNode(p.m_image.rows*p.m_image.cols);

}

void BinarySegApp::AddLossToCRF(CRF& crf, const BS_PatternData& p, const BS_LabelData& l, double scale) const {
    double mult = scale/(p.m_image.rows*p.m_image.cols);
    cv::Point pt;
    for (pt.y = 0; pt.y < p.m_image.rows; ++pt.y) {
        for (pt.x = 0; pt.x < p.m_image.cols; ++pt.x) {
            CRF::NodeId id = pt.y * p.m_image.cols + pt.x;
            double E0 = 0;
            double E1 = 0;
            if (l.m_gt.at<unsigned char>(pt) == BGD) E1 -= 1.0*mult;
            if (l.m_gt.at<unsigned char>(pt) == FGD) E0 -= 1.0*mult;
            crf.AddUnaryTerm(id, doubleToREAL(E0), doubleToREAL(E1));
        }
    }
}

BS_LabelData* BinarySegApp::ExtractLabel(const CRF& crf, const BS_PatternData& x) const {
    BS_LabelData* lp = new BS_LabelData(x.Name());
    lp->m_gt.create(x.m_image.rows, x.m_image.cols, CV_8UC1);
    CRF::NodeId id = 0;
    ImageIterate(lp->m_gt, 
        [&](unsigned char& c) { 
            //ASSERT(crf.GetLabel(id) >= 0);
            if (crf.GetLabel(id) == 0) c = BGD;
            else if (crf.GetLabel(id) == 1) c = FGD;
            else c = -1;
            id++;
        });
    return lp;
}

BS_LabelData* BinarySegApp::Classify(const BS_PatternData& x, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const {
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

BS_LabelData* BinarySegApp::FindMostViolatedConstraint(const BS_PatternData& x, const BS_LabelData& y, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const {
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

void BinarySegApp::EvalPrediction(const BS_PatternData& x, const BS_LabelData& y, const BS_LabelData& ypred) const {
    const std::string& name = ypred.Name();
    if (m_params.show_images)
        ShowImage(ypred.m_gt);

    if (m_params.output_dir != std::string("")) {
        std::string out_filename = m_params.output_dir + "/" + name;
        cv::imwrite(out_filename, ypred.m_gt);
    }
}

bool BinarySegApp::FinalizeIteration(double eps, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const {
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

void BinarySegApp::ValidateExample(const cv::Mat& im, const cv::Mat& gt) {
    ASSERT(im.rows == gt.rows);
    ASSERT(im.cols == gt.cols);
    ASSERT(im.type() == CV_8UC1);
    ASSERT(gt.type() == CV_8UC1);
}

namespace po = boost::program_options;

po::options_description BinarySegApp::GetCommonOptions() {
    po::options_description desc("Binary Segmentation Options");
    desc.add_options()
        ("crf", po::value<std::string>(), "[ho | sf] -> Set CRF optimizer. (default sf)")
        ("stats-file", po::value<std::string>(), "Output file for statistics")
        ("eval-dir", po::value<std::string>(), "Directory for feature evaluation caching")
    ;

    return desc;
}

po::options_description BinarySegApp::GetLearnOptions() {
    po::options_description desc = GetCommonOptions();
    desc.add_options()
        ("all-features", po::value<bool>(), "Turn on all features (for use with feature-train)")
        ("pairwise", po::value<int>(), "[0, 1] -> Use pairwise edge features. (default 0)")
        ("submodular", po::value<int>(), "[0, 1] -> Use submodular features. (default 0)")
    ;
    return desc;
}

po::options_description BinarySegApp::GetClassifyOptions() {
    po::options_description desc = GetCommonOptions();
    desc.add_options()
        ("show", po::value<int>(), "[0,1] -> If nonzero, display each image after it is classified. (default 0)")
        ("output-dir", po::value<std::string>(), "Write predicted images to directory.")
    ;
    return desc;
}


BinarySegApp::Parameters BinarySegApp::ParseLearnOptions(const std::vector<std::string>& args) {
    Parameters params;
    params.eval_dir = std::string();
    params.all_features = false;
    params.pairwise_feature = 0;
    params.submodular_feature = 0;
    params.crf = 0;
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
    if (vm.count("pairwise")) {
        params.pairwise_feature = vm["pairwise"].as<int>();
        std::cout << "Pairwise Feature = " << params.pairwise_feature << "\n";
    }
    if (vm.count("submodular")) {
        params.submodular_feature = vm["submodular"].as<int>();
        std::cout << "Submodular Feature = " << params.submodular_feature << "\n";
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

BinarySegApp::Parameters BinarySegApp::ParseClassifyOptions(const std::vector<std::string>& args) {
    Parameters params;

    params.eval_dir = std::string();
    params.show_images = false;
    params.output_dir = std::string();
    params.stats_file = std::string();
    params.crf = 0;

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

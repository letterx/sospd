#include <cmath>
#include <fstream>
#include "semantic_seg_app.hpp"
#include "svm_c++.hpp"
#include "image_manip.hpp"
#include "crf.hpp"
#include "feature.hpp"
#include "alpha-expansion.hpp"
#include "ale-feature.hpp"
#include "color-patch-multi-feature.hpp"
#include "potts-feature.hpp"
#include "contrast-submodular-multi.hpp"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

SemanticSegApp::SemanticSegApp(const Parameters& params) 
    : SVM_App<SemanticSegApp>(this),
    m_params(params) 
{ }

void SemanticSegApp::ReadExamples(const std::string& file, std::vector<Sem_PatternData*>& patterns, std::vector<Sem_LabelData*>& labels) {
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
            cv::Mat image = cv::imread(images_dir + line, CV_LOAD_IMAGE_COLOR);
            cv::Mat gt = cv::imread(gt_dir + line, CV_LOAD_IMAGE_COLOR);
            ValidateExample(image, gt);
            cv::Mat label_gt;
            ConvertColorToLabel(gt, label_gt);
            patterns.push_back(new Sem_PatternData(line, image));
            labels.push_back(new Sem_LabelData(line, label_gt));
        }
    }
    main_file.close();
}


Sem_PatternData::Sem_PatternData(const std::string& name, const cv::Mat& image) 
    : PatternData(name),
    m_image(image)
{ }

Sem_LabelData::Sem_LabelData(const std::string& name, const cv::Mat& gt)
    : LabelData(name), m_gt(gt) 
{ }

bool Sem_LabelData::operator==(const Sem_LabelData& l) const {
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

double SemanticSegApp::Loss(const Sem_LabelData& l1, const Sem_LabelData& l2, double scale) const {
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


void SemanticSegApp::InitFeatures(const Parameters& param) {
    std::cout << "\nFeatures: "; 
    constexpr double feature_scale = 0.01;
    //m_features.push_back(boost::shared_ptr<FG>(new ClusterColorFeature(feature_scale, m_num_labels)));
    //std::cout << "ColorClusterFeature ";
    m_features.push_back(boost::shared_ptr<FG>(new ALE_Feature(feature_scale, m_num_labels)));
    std::cout << "ALE_Feature ";
    m_features.push_back(boost::shared_ptr<FG>(new ColorPatchMultiFeature(feature_scale, m_num_labels)));
    std::cout << "ColorPatchMulti ";
    m_features.push_back(boost::shared_ptr<FG>(new ContrastSubmodularMultiFeature(feature_scale, m_num_labels)));
    std::cout << "ContrastSubmodular ";
    /*
    m_features.push_back(boost::shared_ptr<FG>(new GMMFeature(feature_scale)));
    if (param.distance_unary || param.all_features)
        m_features.push_back(boost::shared_ptr<FG>(new DistanceFeature(feature_scale)));
    if (param.submodular_feature || param.all_features)
        m_features.push_back(boost::shared_ptr<FG>(new SubmodularFeature(feature_scale)));
    if (param.pairwise_feature || param.all_features)
        m_features.push_back(boost::shared_ptr<FG>(new PairwiseFeature(feature_scale)));
    if (param.contrast_pairwise_feature || param.all_features)
        m_features.push_back(boost::shared_ptr<FG>(new ContrastPairwiseFeature(feature_scale)));
    if (param.contrast_submodular_feature || param.all_features) {
        std::cout << "Using Contrast Submodular Feature!\n";
        m_features.push_back(boost::shared_ptr<FG>(new ContrastSubmodularFeature(feature_scale)));
    }
    */
    std::cout << "\n";

    if (param.eval_dir != std::string("")) {
        for (auto fp : m_features) {
            fp->LoadEvaluation(param.eval_dir);
        }
    }
}

long SemanticSegApp::NumFeatures() const {
    long n = 0;
    for (auto fgp : m_features) {
        n += fgp->NumFeatures();
    }
    return n;
}

void SemanticSegApp::InitializeCRF(MultiLabelCRF& crf, const Sem_PatternData& p) const {
    crf.AddNode(p.m_image.rows*p.m_image.cols);
}

void SemanticSegApp::AddLossToCRF(MultiLabelCRF& crf, const Sem_PatternData& p, const Sem_LabelData& l, double scale) const {
    double mult = scale/(p.m_image.rows*p.m_image.cols);
    cv::Point pt;
    for (pt.y = 0; pt.y < p.m_image.rows; ++pt.y) {
        for (pt.x = 0; pt.x < p.m_image.cols; ++pt.x) {
            MultiLabelCRF::NodeId id = pt.y * p.m_image.cols + pt.x;
            std::vector<REAL> costs(m_num_labels, doubleToREAL(-mult));
            costs[l.m_gt.at<Label>(pt)] = 0;
            crf.AddUnaryTerm(id, costs);
        }
    }
}

Sem_LabelData* SemanticSegApp::ExtractLabel(const MultiLabelCRF& crf, const Sem_PatternData& x) const {
    Sem_LabelData* lp = new Sem_LabelData(x.Name());
    lp->m_gt.create(x.m_image.rows, x.m_image.cols, CV_32SC1);
    MultiLabelCRF::NodeId id = 0;
    cv::Point pt;
    for (pt.y = 0; pt.y < x.m_image.rows; ++pt.y) {
        for (pt.x = 0; pt.x < x.m_image.cols; ++pt.x) {
            lp->m_gt.at<Label>(pt) = crf.GetLabel(id);
            id++;
        }
    }
    return lp;
}

Sem_LabelData* SemanticSegApp::Classify(const Sem_PatternData& x, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const {
    MultiLabelCRF crf(m_num_labels);
    InitializeCRF(crf, x);
    size_t feature_base = 1;
    for (auto fgp : m_features) {
        fgp->AddToCRF(crf, x, sm->w + feature_base );
        feature_base += fgp->NumFeatures();
    }
    crf.Solve();
    return ExtractLabel(crf, x);
}

Sem_LabelData* SemanticSegApp::FindMostViolatedConstraint(const Sem_PatternData& x, const Sem_LabelData& y, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const {
    MultiLabelCRF crf(m_num_labels);
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

bool SemanticSegApp::FinalizeIteration(double eps, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const {
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
            if (sparm->epsilon < 0.0001)
                return false;
            return true;
        }
        feature_base += fgp->NumFeatures();
    }
    return false;
}

void SemanticSegApp::EvalPrediction(const Sem_PatternData& x, const Sem_LabelData& y, const Sem_LabelData& ypred) const {
    const std::string& name = ypred.Name();
    cv::Mat color_image;
    ConvertLabelToColor(ypred.m_gt, color_image);
    if (m_params.show_images) {
        cv::namedWindow("Display window", CV_WINDOW_AUTOSIZE);
        cv::imshow("Display window", color_image);
        cv::waitKey(0);
    }

    if (m_params.output_dir != std::string("")) {
        std::string out_filename = m_params.output_dir + "/" + name;
        cv::imwrite(out_filename, color_image);
    }
}

void SemanticSegApp::ValidateExample(const cv::Mat& image, const cv::Mat& gt) const {
    ASSERT(image.data != NULL);
    ASSERT(image.type() == CV_8UC3);
    ASSERT(gt.data != NULL);
    ASSERT(gt.type() == CV_8UC3);
    ASSERT(image.rows == gt.rows);
    ASSERT(image.cols == gt.cols);
}

// The following two functions from Lubor Ladicky's code
void SemanticSegApp::ConvertColorToLabel(const cv::Vec3b& color, unsigned char& label) const {
    label = 0;
    for(int i = 0; i < 8; i++) label = (label << 3) | (((color[2] >> i) & 1) << 0) | (((color[1] >> i) & 1) << 1) | (((color[0] >> i) & 1) << 2);
    label--;
}

void SemanticSegApp::ConvertLabelToColor(unsigned char label, cv::Vec3b& color) const {
    label++;
    color[0] = color[1] = color[2] = 0;
    for(int i = 0; label > 0; i++, label >>= 3)
    {
        color[2] |= (unsigned char) (((label >> 0) & 1) << (7 - i));
        color[1] |= (unsigned char) (((label >> 1) & 1) << (7 - i));
        color[0] |= (unsigned char) (((label >> 2) & 1) << (7 - i));
    }
}

void SemanticSegApp::ConvertColorToLabel(const cv::Mat& color_image, cv::Mat& label_image) const {
    label_image.create(color_image.rows, color_image.cols, CV_32SC1);
    cv::Point p;
    for (p.y = 0; p.y < label_image.rows; ++p.y) {
        for (p.x = 0; p.x < label_image.cols; ++p.x) {
            const cv::Vec3b& color = color_image.at<cv::Vec3b>(p);
            unsigned char label;
            ConvertColorToLabel(color, label);
            label_image.at<Label>(p) = label;
        }
    }
}

void SemanticSegApp::ConvertLabelToColor(const cv::Mat& label_image, cv::Mat& color_image) const {
    color_image.create(label_image.rows, label_image.cols, CV_8UC3);
    cv::Point p;
    for (p.y = 0; p.y < label_image.rows; ++p.y) {
        for (p.x = 0; p.x < label_image.cols; ++p.x) {
            Label l = label_image.at<Label>(p);
            ConvertLabelToColor(l, color_image.at<cv::Vec3b>(p));
        }
    }
}

namespace po = boost::program_options;

po::options_description SemanticSegApp::GetCommonOptions() {
    po::options_description desc("Semantic Segmentation Options");
    desc.add_options()
        ("crf", po::value<std::string>(), "[ho | sf] -> Set CRF optimizer. (default sf)")
        ("stats-file", po::value<std::string>(), "Output file for statistics")
        ("eval-dir", po::value<std::string>(), "Directory for feature evaluation caching")
    ;

    return desc;
}

po::options_description SemanticSegApp::GetLearnOptions() {
    po::options_description desc = GetCommonOptions();
    desc.add_options()
        ("all-features", po::value<bool>(), "Turn on all features (for use with feature-train)")
    ;
    return desc;
}

po::options_description SemanticSegApp::GetClassifyOptions() {
    po::options_description desc = GetCommonOptions();
    desc.add_options()
        ("show", po::value<int>(), "[0,1] -> If nonzero, display each image after it is classified. (default 0)")
        ("output-dir", po::value<std::string>(), "Write predicted images to directory.")
    ;
    return desc;
}


SemanticSegApp::Parameters SemanticSegApp::ParseLearnOptions(const std::vector<std::string>& args) {
    Parameters params;
    params.all_features = false;
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
        } else {
            std::cout << "Unrecognized optimizer\n";
            exit(-1);
        }
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

SemanticSegApp::Parameters SemanticSegApp::ParseClassifyOptions(const std::vector<std::string>& args) {
    Parameters params;

    params.show_images = false;
    params.crf = 0;
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

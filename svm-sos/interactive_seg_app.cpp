#include <cmath>
#include <fstream>
#include "interactive_seg_app.hpp"
#include "svm_c++.hpp"
#include "image_manip.hpp"
#include "crf.hpp"
#include "feature.hpp"
#include "submodular-feature.hpp"
#include "gmm-feature.hpp"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

InteractiveSegApp::InteractiveSegApp(const Parameters& params) 
    : SVM_App<InteractiveSegApp>(this),
    m_params(params) 
{
    InitFeatures(params);        
}

void InteractiveSegApp::ReadExamples(const std::string& file, std::vector<IS_PatternData*>& patterns, std::vector<IS_LabelData*>& labels) {
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
            patterns.push_back(new IS_PatternData(line, image, trimap));
            labels.push_back(new IS_LabelData(line, gt));
        }
    }
    main_file.close();
}


IS_PatternData::IS_PatternData(const std::string& name, const cv::Mat& image, const cv::Mat& trimap) 
    : PatternData(name),
    m_image(image)
{
    ConvertToMask(trimap, m_tri);

    /*
    m_fgdDist.create(m_image.rows, m_image.cols, CV_32SC1);
    m_bgdDist.create(m_image.rows, m_image.cols, CV_32SC1);
    m_dist_feature.create(m_image.rows, m_image.cols, CV_32SC1);
    CalcDistances(m_tri, m_fgdDist, m_bgdDist, m_dist_feature);

    cv::Mat grabcutResult;
    m_tri.copyTo(grabcutResult);
    if (sparm->grabcut_unary)
        cv::grabCut(m_image, grabcutResult, cv::Rect(), m_bgdModel, m_fgdModel, sparm->grabcut_unary, cv::GC_INIT_WITH_MASK);

    learnGMMs(m_image, grabcutResult, m_bgdGMM, m_fgdGMM);

    m_fgdUnaries.create(m_image.rows, m_image.cols, CV_64FC1);
    m_bgdUnaries.create(m_image.rows, m_image.cols, CV_64FC1);
    CalcUnaries(*this);

    m_beta = calcBeta(m_image);
    m_downW.create(m_image.rows, m_image.cols, CV_64FC1);
    m_rightW.create(m_image.rows, m_image.cols, CV_64FC1);
    std::function<void(const cv::Vec3b&, const cv::Vec3b&, double&, double&)> calcExpDiff = 
        [&](const cv::Vec3b& color1, const cv::Vec3b& color2, double& d1, double& d2) {
            cv::Vec3d c1 = color1;
            cv::Vec3d c2 = color2;
            cv::Vec3d diff = c1-c2;
            d1 = exp(-m_beta*diff.dot(diff));
            //d1 = abs(diff[0]) + abs(diff[1]) + abs(diff[2]);
    };
    ImageIterate(m_image, m_downW, cv::Point(0.0, 1.0), calcExpDiff);
    ImageIterate(m_image, m_rightW, cv::Point(1.0, 0.0), calcExpDiff);
    */
}

IS_LabelData::IS_LabelData(const std::string& name, const cv::Mat& gt)
    : LabelData(name)
{ 
    ConvertGreyToMask(gt, m_gt);
}

bool IS_LabelData::operator==(const IS_LabelData& l) const {
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

double InteractiveSegApp::Loss(const IS_LabelData& l1, const IS_LabelData& l2, double scale) const {
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
    constexpr double feature_scale = 1.0;
    m_features.push_back(boost::shared_ptr<FG>(new GMMFeature(feature_scale)));
    if (param.submodular_feature)
        m_features.push_back(boost::shared_ptr<FG>(new SubmodularFeature(feature_scale)));

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

void InteractiveSegApp::InitializeCRF(CRF& crf, const IS_PatternData& p) const {
    crf.AddNode(p.m_image.rows*p.m_image.cols);

}

void InteractiveSegApp::AddLossToCRF(CRF& crf, const IS_PatternData& p, const IS_LabelData& l, double scale) const {
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

IS_LabelData* InteractiveSegApp::ExtractLabel(const CRF& crf, const IS_PatternData& x) const {
    IS_LabelData* lp = new IS_LabelData(x.Name());
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
    //x.m_tri.copyTo(lp->m_gt);
    //cv::Mat bgdModel, fgdModel;
    //cv::grabCut(x.m_image, lp->m_gt, cv::Rect(), bgdModel, fgdModel, 10, cv::GC_INIT_WITH_MASK);
    return lp;
}

IS_LabelData* InteractiveSegApp::Classify(const IS_PatternData& x, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const {
    if (sparm->grabcut_classify) {
        IS_LabelData* y = new IS_LabelData(x.Name());
        x.m_tri.copyTo(y->m_gt);
        cv::Mat bgdModel;
        cv::Mat fgdModel;
        cv::grabCut(x.m_image, y->m_gt, cv::Rect(), bgdModel, fgdModel, sparm->grabcut_classify);
        return y;
    } else {
        CRF crf;
        SubmodularFlow sf;
        HigherOrderWrapper ho;
        if (sparm->crf == 0) {
            crf.Wrap(&sf);
        } else {
            crf.Wrap(&ho);
        }
        InitializeCRF(crf, x);
        size_t feature_base = 1;
        for (auto fgp : m_features) {
            double violation = fgp->Violation(feature_base, sm->w);
            double w2 = 0;
            for (size_t i = feature_base; i < feature_base + fgp->NumFeatures(); ++i) 
                w2 += sm->w[i] * sm->w[i];
            if (violation > 0.0001 * w2) {
                std::cout << "*** Max Violation: " << violation << ", |w|^2: " << w2 << "***";
                std::cout.flush();
            }
            fgp->AddToCRF(crf, x, sm->w + feature_base );
            feature_base += fgp->NumFeatures();
        }
        crf.Solve();
        return ExtractLabel(crf, x);
    }
}

IS_LabelData* InteractiveSegApp::FindMostViolatedConstraint(const IS_PatternData& x, const IS_LabelData& y, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const {
    CRF crf;
    SubmodularFlow sf;
    HigherOrderWrapper ho;
    if (sparm->crf == 0) {
        crf.Wrap(&sf);
    } else {
        crf.Wrap(&ho);
    }
    InitializeCRF(crf, x);
    size_t feature_base = 1;
    for (auto fgp : m_features) {
        double violation = fgp->Violation(feature_base, sm->w);
        double w2 = 0;
        for (size_t i = feature_base; i < feature_base + fgp->NumFeatures(); ++i) 
            w2 += sm->w[i] * sm->w[i];
        if (violation > 0.0001 * w2) {
            std::cout << "*** Max Violation: " << violation << ", |w|^2: " << w2 << "***";
            std::cout.flush();
        }
        fgp->AddToCRF(crf, x, sm->w + feature_base );
        feature_base += fgp->NumFeatures();
    }
    AddLossToCRF(crf, x, y, sparm->loss_scale);
    crf.Solve();
    return ExtractLabel(crf, x);
}

void InteractiveSegApp::EvalPrediction(const IS_PatternData& x, const IS_LabelData& y, const IS_LabelData& ypred) const {
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
        ("grabcut-unary", po::value<int>(), "[0..] Use n iterations of grabcut to initialize GMM unary features (default 0)")
        ("distance-unary", po::value<int>(), "[0,1] If 1, use distance features for unary potentials")
        ("pairwise", po::value<int>(), "[0, 1] -> Use pairwise edge features. (default 0)")
        ("contrast-pairwise", po::value<int>(), "[0, 1] -> Use contrast-sensitive pairwise features. (default 0)")
        ("submodular", po::value<int>(), "[0, 1] -> Use submodular features. (default 1)")
        ("constraint-scale", po::value<double>(), "Scaling factor for constraint violations")
        ("feature-scale", po::value<double>(), "Scaling factor for Psi")
        ("loss-scale", po::value<double>(), "Scaling factor for Delta (loss function")
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
    params.grabcut_classify = 0;
    params.crf = 0;
    params.grabcut_unary = 0;
    params.distance_unary = 1;
    params.pairwise_feature = 0;
    params.contrast_pairwise_feature = 0;
    params.submodular_feature = 1;
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
    if (vm.count("grabcut-unary")) 
        params.grabcut_unary = vm["grabcut-unary"].as<int>();
    if (vm.count("distance-unary"))
        params.distance_unary = vm["distance-unary"].as<int>();
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
    if (vm.count("stats-file")) {
        params.stats_file = vm["stats-file"].as<std::string>();
    }
    if (vm.count("eval-dir")) {
        params.eval_dir = vm["eval-dir"].as<std::string>();
    }
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

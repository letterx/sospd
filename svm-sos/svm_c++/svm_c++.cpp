#include "svm_c++.hpp"
#include "feature.hpp"
#include <string>
#include <memory>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

std::unique_ptr<SVM_Cpp_Base> g_application{};

void SVM_Cpp_Base::trainFeatures(const std::string& train_file, 
        const std::string& eval_file, 
        const std::string& output_dir) {
    PatternVec train_patterns;
    LabelVec train_labels;
    PatternVec eval_patterns;
    LabelVec eval_labels;
    readExamples(train_file, train_patterns, train_labels);
    readExamples(eval_file, eval_patterns, eval_labels);
    //initFeatures(m_derived->Params());

    for (const auto& fgp : features()) {
        fgp->Train(train_patterns, train_labels);
    }
    for (const auto& fgp : features()) {
        fgp->Evaluate(eval_patterns);
        fgp->SaveEvaluation(output_dir);
    }
}

long SVM_Cpp_Base::numFeatures() const {
    long n = 0;
    for (const auto& fgp : features())
        n += fgp->NumFeatures();
    return n;
}

boost::program_options::options_description
SVM_Cpp_Base::getBaseLearnParams() {
    po::options_description desc("General learning options");
    desc.add_options()
        ("constraint-scale", po::value<double>(&m_constraint_scale)->default_value(1.0), "Scaling constant for enforcing constraints")
        ("feature-scale", po::value<double>(&m_feature_scale)->default_value(1.0), "Scaling constant for features Psi")
        ("loss-scale", po::value<double>(&m_loss_scale)->default_value(1.0), "Scaling constant for loss function Delta")
    ;
    return desc;
}

void SVM_Cpp_Base::parseBaseLearnParams(int argc, char** argv) {
    m_constraint_scale = 100000.0;
    m_loss_scale = 1000.0;

    po::options_description desc;
    desc.add(getBaseLearnParams());
    po::variables_map vm;
    try {
        po::parsed_options parsed = po::command_line_parser(argc, argv).
            options(desc).
            allow_unregistered().
            run();
        std::vector<std::string> passOnwards = po::collect_unrecognized(parsed.options, po::include_positional);
        po::store(parsed, vm);
        po::notify(vm);
        parseLearnParams(passOnwards);
    } catch (std::exception& e) {
        std::cout << "Parsing exception: " << e.what() << "\n";
        printLearnHelp();
    }
}

void SVM_Cpp_Base::printLearnHelp() {
    po::options_description desc;
    desc.add(getBaseLearnParams())
        .add(getLearnParams());
    std::cout << desc << "\n";
}


void SVM_Cpp_Base::parseBaseFeatureParams(int argc, char** argv, std::string& trainFile, std::string& evalFile, std::string& outputDir) {
    po::options_description desc;
    desc.add_options()
        ("train-file", po::value<std::string>(&trainFile)->required(), "Example file for training instances")
        ("eval-file", po::value<std::string>(&evalFile)->required(), "Example file for evaluated instances")
        ("output-dir", po::value<std::string>(&outputDir)->required(), "Output directory for evaluated instances")
        ("h,help", "Display this help message")
    ;

    try {
        po::variables_map vm;
        po::parsed_options parsed = po::command_line_parser(argc, argv).
            options(desc).
            allow_unregistered().
            run();
        std::vector<std::string> pass_onwards = po::collect_unrecognized(parsed.options, po::include_positional);
        po::store(parsed, vm);
        if (vm.count("help")) {
            std::cout << desc;
            exit(0);
        }
        po::notify(vm);

        parseFeatureParams(pass_onwards);
    } catch (std::exception& e) {
        std::cout << "Error: " << e.what() << "\n";
        std::cout << desc << "\n";
        exit(-1);
    }
}

boost::program_options::options_description
SVM_Cpp_Base::getBaseClassifyParams() {
    po::options_description desc("General classify options");
    desc.add_options();
    return desc;
}

void SVM_Cpp_Base::parseBaseClassifyParams(int argc, char** argv) {

}

void SVM_Cpp_Base::printClassifyHelp() {
    po::options_description desc;
    desc.add(getBaseClassifyParams())
        .add(getClassifyParams());
    std::cout << desc << "\n";
}

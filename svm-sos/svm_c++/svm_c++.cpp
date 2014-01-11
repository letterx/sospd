#include "svm_c++.hpp"
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
        ("constraint-scale", po::value<double>(), "Scaling constant for enforcing constraints")
        ("feature-scale", po::value<double>(), "Scaling constant for features Psi")
        ("loss-scale", po::value<double>(), "Scaling constant for loss function Delta")
    ;
    return desc;
}

void SVM_Cpp_Base::parseBaseLearnParams(int argc, char** argv) {

}

void SVM_Cpp_Base::printLearnHelp() {
    po::options_description desc;
    desc.add(getBaseLearnParams())
        .add(getLearnParams());
    std::cout << desc << "\n";
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

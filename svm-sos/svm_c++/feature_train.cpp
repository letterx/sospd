#include "feature.hpp"
#include "svm_c++.hpp"

int main (int argc, char** argv) {
    std::string train_file, eval_file, output_dir;
    g_application = SVM_Cpp_Base::newUserApplication();
    g_application->parseBaseFeatureParams(argc, argv, train_file, eval_file, output_dir);
    g_application->trainFeatures(train_file, eval_file, output_dir);
    return 0;
}

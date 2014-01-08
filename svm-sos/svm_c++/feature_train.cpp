#include "feature.hpp"
#include "svm_c++.hpp"
#include "svm_struct_options.hpp"

int main (int argc, char** argv) {
    std::string train_file, eval_file, output_dir;
    SVM_App_Base* app = ParseFeatureTrainParameters(argc, argv, train_file, eval_file, output_dir);
    app->train_features(train_file, eval_file, output_dir);
    return 0;
}

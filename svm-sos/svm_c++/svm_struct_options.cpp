#include "svm_struct_options.hpp"
#include <string>
#include <cstring>
#include <iostream>
#include <boost/program_options.hpp>
#include "svm_c++.hpp"
#include "interactive_seg_app.hpp"
#include "semantic_seg_app.hpp"
#include "binary_seg_app.hpp"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

static po::options_description GetStructLearnParameters() {
    po::options_description desc("General learning options");
    desc.add_options()
        ("app", po::value<std::string>(), "[interactive-seg] Chooses which application to use")
        ("constraint-scale", po::value<double>(), "Scaling constant for enforcing constraints")
        ("feature-scale", po::value<double>(), "Scaling constant for features Psi")
        ("loss-scale", po::value<double>(), "Scaling constant for loss function Delta")
    ;
    return desc;
}

static po::options_description GetStructClassifyParameters() {
    po::options_description desc("General classify options");
    desc.add_options()
        ("app", po::value<std::string>(), "[interactive-seg] Chooses which application to use")
    ;
    return desc;
}

SVM_App_Base* ParseStructLearnParameters(STRUCT_LEARN_PARM* sparm) {
    SVM_App_Base* app;

    sparm->constraint_scale = 100000.0;
    sparm->loss_scale = 1000.0;

    po::options_description desc = GetStructLearnParameters();
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(sparm->custom_argc, sparm->custom_argv).
        options(desc).
        allow_unregistered().
        run();
    std::vector<std::string> pass_onwards = po::collect_unrecognized(parsed.options, po::include_positional);
    po::store(parsed, vm);

    if (vm.count("app")) {
        if (vm["app"].as<std::string>() == std::string("interactive-seg"))
            app = new InteractiveSegApp(InteractiveSegApp::ParseLearnOptions(pass_onwards));
        else if (vm["app"].as<std::string>() == std::string("semantic-seg"))
            app = new SemanticSegApp(SemanticSegApp::ParseLearnOptions(pass_onwards));
        else if (vm["app"].as<std::string>() == std::string("binary-seg"))
            app = new BinarySegApp(BinarySegApp::ParseLearnOptions(pass_onwards));
        else {
            std::cout << "Unrecognized application: " << vm["app"].as<std::string>() << "\n";
            exit(-1);
        }
    } else {
        std::cout << "Must supply application!\n";
        exit(-1);
    }
    if (vm.count("constraint-scale"))
        sparm->constraint_scale = vm["constraint-scale"].as<double>();
    if (vm.count("feature-scale"))
        sparm->feature_scale = vm["feature-scale"].as<double>();
    if (vm.count("loss-scale"))
        sparm->loss_scale = vm["loss-scale"].as<double>();
    return app;
}

SVM_App_Base* ParseFeatureTrainParameters(int argc, char** argv, std::string& train_file, std::string& eval_file, std::string& output_dir) {
    SVM_App_Base* app;
    std::string app_name;
    
    try {

        po::options_description desc;
        desc.add_options()
            ("app", po::value<std::string>(&app_name)->required(), "Application to train features for")
            ("train-file", po::value<std::string>(&train_file)->required(), "Example file for training instances")
            ("eval-file", po::value<std::string>(&eval_file)->required(), "Example file for evaluated instances")
            ("output-dir", po::value<std::string>(&output_dir)->required(), "Output directory for evaluated instances")
        ;

        po::variables_map vm;
        po::parsed_options parsed = po::command_line_parser(argc, argv).
            options(desc).
            allow_unregistered().
            run();
        std::vector<std::string> pass_onwards = po::collect_unrecognized(parsed.options, po::include_positional);
        po::store(parsed, vm);
        po::notify(vm);

        if (vm["app"].as<std::string>() == std::string("interactive-seg"))
            app = new InteractiveSegApp(InteractiveSegApp::ParseLearnOptions(pass_onwards));
        else if (vm["app"].as<std::string>() == std::string("semantic-seg"))
            app = new SemanticSegApp(SemanticSegApp::ParseLearnOptions(pass_onwards));
        else if (vm["app"].as<std::string>() == std::string("binary-seg"))
            app = new BinarySegApp(BinarySegApp::ParseLearnOptions(pass_onwards));
        else {
            std::cout << "Unrecognized application: " << vm["app"].as<std::string>() << "\n";
                exit(-1);
        }
    } catch (std::exception& e) {
        std::cout << "Error: " << e.what() << "\n";
        exit(-1);
    }
    return app;
}

SVM_App_Base* ParseStructClassifyParameters(STRUCT_LEARN_PARM* sparm) {
    SVM_App_Base* app;

    po::options_description desc = GetStructClassifyParameters();
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(sparm->custom_argc, sparm->custom_argv).
        options(desc).
        allow_unregistered().
        run();
    std::vector<std::string> pass_onwards = po::collect_unrecognized(parsed.options, po::include_positional);
    po::store(parsed, vm);

    if (vm.count("app")) {
        if (vm["app"].as<std::string>() == std::string("interactive-seg"))
            app = new InteractiveSegApp(InteractiveSegApp::ParseClassifyOptions(pass_onwards));
        else if (vm["app"].as<std::string>() == std::string("semantic-seg"))
            app = new SemanticSegApp(SemanticSegApp::ParseClassifyOptions(pass_onwards));
        else if (vm["app"].as<std::string>() == std::string("binary-seg"))
            app = new BinarySegApp(BinarySegApp::ParseClassifyOptions(pass_onwards));
        else {
            std::cout << "Unrecognized application: " << vm["app"].as<std::string>() << "\n";
            exit(-1);
        }
    } else {
        std::cout << "Must supply application!\n";
        exit(-1);
    }
    return app;
}

void PrintStructLearnHelp() {
    po::options_description desc;
    desc.add(GetStructLearnParameters()).add(InteractiveSegApp::GetLearnOptions()).add(SemanticSegApp::GetLearnOptions()).add(BinarySegApp::GetLearnOptions());
    std::cout << desc << "\n";
}

void PrintStructClassifyHelp() {
    po::options_description desc;
    desc.add(GetStructClassifyParameters()).add(InteractiveSegApp::GetClassifyOptions()).add(SemanticSegApp::GetClassifyOptions()).add(BinarySegApp::GetClassifyOptions());
    std::cout << desc << "\n";
}

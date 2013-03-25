#include "svm_struct_options.hpp"
#include <string>
#include <cstring>
#include <iostream>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

static po::options_description GetCommonOptions() {
    po::options_description desc("Custom SVM options");
    desc.add_options()
        ("crf", po::value<std::string>(), "[ho | sf] -> Set CRF optimizer. (default sf)")
    ;

    return desc;
}

static po::options_description GetLearnOptions() {
    po::options_description desc = GetCommonOptions();
    desc.add_options()
        ("pairwise", po::value<int>(), "[0, 1] -> Use pairwise edge features. (default 0)")
        ("contrast-pairwise", po::value<int>(), "[0, 1] -> Use contrast-sensitive pairwise features. (default 0)")
        ("submodular", po::value<int>(), "[0, 1] -> Use submodular features. (default 1)")
    ;
    return desc;
}

static po::options_description GetClassifyOptions() {
    po::options_description desc = GetCommonOptions();
    desc.add_options()
        ("grabcut", po::value<int>(), "[0..] -> If nonzero, run n iterations of grabcut as the classifier instead. (default 0)")
        ("show", po::value<int>(), "[0,1] -> If nonzero, display each image after it is classified. (default 0)")
        ("output-dir", po::value<std::string>(), "Write predicted images to directory.")
    ;
    return desc;
}


void ParseStructLearnParameters(STRUCT_LEARN_PARM* sparm) {
    sparm->grabcut_classify = 0;
    sparm->crf = 0;
    sparm->pairwise_feature = 0;
    sparm->contrast_pairwise_feature = 0;
    sparm->submodular_feature = 1;

    po::options_description desc = GetLearnOptions();
    po::variables_map vm;
    po::store(po::parse_command_line(sparm->custom_argc, sparm->custom_argv, desc), vm);

    if (vm.count("crf")) {
        std::string type = vm["crf"].as<std::string>();
        if (type == "sf") {
            std::cout << "SubmodularFlow optimizer\n";
            sparm->crf = 0;
        } else if (type == "ho") {
            std::cout << "HigherOrder optimizer\n";
            sparm->crf = 1;
        } else {
            std::cout << "Unrecognized optimizer\n";
            exit(-1);
        }
    }
    if (vm.count("pairwise")) {
        sparm->pairwise_feature = vm["pairwise"].as<int>();
        std::cout << "Pairwise Feature = " << sparm->pairwise_feature << "\n";
    }
    if (vm.count("contrast-pairwise")) {
        sparm->contrast_pairwise_feature = vm["contrast-pairwise"].as<int>();
        std::cout << "Contrast Pairwise Feature = " << sparm->contrast_pairwise_feature << "\n";
    }
    if (vm.count("submodular")) {
        sparm->submodular_feature = vm["submodular"].as<int>();
        std::cout << "Submodular Feature = " << sparm->submodular_feature << "\n";
    }

}

void PrintStructLearnHelp() {
    std::cout << GetLearnOptions() << "\n";
}

void ParseStructClassifyParameters(STRUCT_LEARN_PARM* sparm) {
    sparm->show_images = false;
    sparm->grabcut_classify = 0;
    sparm->crf = 0;

    po::options_description desc = GetClassifyOptions();
    po::variables_map vm;
    po::store(po::parse_command_line(sparm->custom_argc, sparm->custom_argv, desc), vm);

    if (vm.count("crf")) {
        std::string type = vm["crf"].as<std::string>();
        if (type == "sf") {
            std::cout << "SubmodularFlow optimizer\n";
            sparm->crf = 0;
        } else if (type == "ho") {
            std::cout << "HigherOrder optimizer\n";
            sparm->crf = 1;
        } else {
            std::cout << "Unrecognized optimizer\n";
            exit(-1);
        }
    }
    if (vm.count("show")) {
        sparm->show_images = vm["show"].as<int>();
        std::cout << "Show Images = " << sparm->show_images << "\n";
    }
    if (vm.count("grabcut")) {
        sparm->grabcut_classify = vm["grabcut"].as<int>();
        std::cout << "Grabcut iterations = " << sparm->grabcut_classify << "\n";
    }
    if (vm.count("output-dir")) {
        strncpy(sparm->output_dir, vm["output-dir"].as<std::string>().c_str(), 256);
        if (sparm->output_dir[255] != 0) {
            std::cout << "Output-directory name too long!\n";
            exit(-1);
        }
    } else {
        sparm->output_dir[0] = 0;
    }
}

void PrintStructClassifyHelp() {
    std::cout << GetClassifyOptions() << "\n";
}

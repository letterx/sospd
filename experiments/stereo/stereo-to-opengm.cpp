/*
 * stereo.cpp
 *
 * Copyright 2014 Alexander Fix
 * See LICENSE.txt for license information
 *
 * Stereo inference using fusion proposals
 * Unary potentials and proposals are loaded from file
 */

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/program_options.hpp>

#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/graphicalmodel/graphicalmodel_hdf5.hxx"
#include "opengm/graphicalmodel/space/simplediscretespace.hxx"
#include "opengm/operations/adder.hxx"
#include "opengm/functions/sparsemarray.hxx"

typedef size_t Label;
typedef size_t VarId;

typedef double ValueType;
typedef opengm::SimpleDiscreteSpace<size_t, size_t> Space;
typedef opengm::SparseFunction<double, VarId, Label> SparseFunction;
typedef opengm::GraphicalModel<double, opengm::Adder, OPENGM_TYPELIST_2(opengm::ExplicitFunction<double>, SparseFunction ) , Space> Model;


std::vector<cv::Mat> g_proposals;
float g_kappa = 0.001;
float g_alpha = 10.0;
float g_scale = 20000;


double StereoEnergy(const Label buf[], const VarId nodes[]) {
    float disparity[3];
    double energy;
    for (int i = 0; i < 3; ++i)
        disparity[i] = g_proposals[buf[i]].at<float>(nodes[i]);
    if (std::abs(disparity[1] - disparity[0]) > g_alpha
            || std::abs(disparity[2] - disparity[1]) > g_alpha) {
        energy = g_kappa;
    } else {
        float curvature = disparity[0] - 2*disparity[1] + disparity[2];
        energy = std::min(curvature*curvature, g_kappa);
    }
    return energy*g_scale/g_kappa;
}


void SetupEnergy(Model& gm,
        const std::vector<cv::Mat>& proposals, 
        const std::vector<cv::Mat>& unary);
std::vector<cv::Mat> ReadUnaries(const std::string& unaryFilename);
std::vector<cv::Mat> ReadProposals(const std::string& proposalFilename);

int width = 0;
int height = 0;
int nproposals = 0;

int main(int argc, char **argv) {
    namespace po = boost::program_options;
    // Variables set by program options
    std::string basename;
    std::string unaryFilename;
    std::string proposalFilename;
    std::string outfilename;

    // Setup and parse options
    po::options_description options("Stereo arguments");
    options.add_options()
        ("help", "Display this help message")
        ("image",
         po::value<std::string>(&basename)->required(),
         "Name of image (without extension)")
        ("kappa", 
         po::value<float>(&g_kappa)->default_value(0.001),
         "Truncation for stereo prior")
        ("alpha",
         po::value<float>(&g_alpha)->default_value(10),
         "Max gradient for stereo prior")
        ("lambda",
         po::value<float>(&g_scale)->default_value(20000),
         "Scale for stereo prior")
    ;

    po::positional_options_description positionalOpts;
    positionalOpts.add("image", 1);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).
                options(options).positional(positionalOpts).run(), vm);
        if (vm.count("help")) {
            std::cout << options;
            exit(0);
        }
        po::notify(vm);
    } catch (std::exception& e) {
        std::cout << "Parsing error: " << e.what() << "\n";
        std::cout << "Usage: denoise [options] basename\n";
        std::cout << options;
        exit(-1);
    }

    unaryFilename = basename + ".unary";
    proposalFilename = basename + ".proposals";
    outfilename = basename + ".h5";

    // Read stored unaries and proposed moves
    std::cout << "Reading proposals...\n";
    g_proposals = ReadProposals(proposalFilename);
    std::cout << "Reading unaries...\n";
    std::vector<cv::Mat> unaries = ReadUnaries(unaryFilename);
    cv::Mat image(height, width, CV_32FC1);

    std::cout << "Setting up energy...\n";
    Space space(size_t(height*width), nproposals);
    Model gm(space);
    SetupEnergy(gm, g_proposals, unaries);

    std::cout << "Writing out model...\n";
    opengm::hdf5::save(gm, outfilename, "gm");  


    return 0;
}

void SetupEnergy(Model& gm,
        const std::vector<cv::Mat>& proposals, 
        const std::vector<cv::Mat>& unary) {
    const size_t nLabels = nproposals;
    const size_t cliqueShape[] = { nLabels, nLabels, nLabels };
    // For each 1x3 patch, add in a StereoClique
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width - 2; ++j) {
            SparseFunction f(cliqueShape, cliqueShape + 3, g_scale);
            size_t nodes[3] = { size_t(i*width+j), size_t(i*width+j+1), size_t(i*width+j+2) };
            size_t labels[3] = { 0, 0, 0 };
            for (labels[0] = 0; labels[0] < nLabels; ++labels[0]) {
                for (labels[1] = 0; labels[1] < nLabels; ++labels[1]) {
                    for (labels[2] = 0; labels[2] < nLabels; ++labels[2]) {
                        auto e = StereoEnergy(labels, nodes);
                        if (e < g_scale)
                            f.insert(labels, e);
                    }
                }
            }
            auto fid = gm.addFunction(f);
            gm.addFactor(fid, nodes, nodes+3);
        }
    }
    // For each 3x1 patch, add in a StereoClique
    for (int i = 0; i < height-2; ++i) {
        for (int j = 0; j < width; ++j) {
            SparseFunction f(cliqueShape, cliqueShape + 3, g_scale);
            size_t nodes[3] = { size_t(i*width+j), size_t((i+1)*width+j), size_t((i+2)*width+j) };
            size_t labels[3] = { 0, 0, 0 };
            for (labels[0] = 0; labels[0] < nLabels; ++labels[0]) {
                for (labels[1] = 0; labels[1] < nLabels; ++labels[1]) {
                    for (labels[2] = 0; labels[2] < nLabels; ++labels[2]) {
                        auto e = StereoEnergy(labels, nodes);
                        if (e < g_scale)
                            f.insert(labels, e);
                    }
                }
            }
            auto fid = gm.addFunction(f);
            gm.addFactor(fid, nodes, nodes+3);
        }
    }

    const size_t unaryShape[] = { nLabels };
    // Add the unary terms
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            VarId n = i*width + j;
            opengm::ExplicitFunction<double> f(unaryShape, unaryShape+1);
            for (Label l = 0; l < nLabels; ++l)
                f(&l) = unary[l].at<float>(i,j);
            auto fid = gm.addFunction(f);
            gm.addFactor(fid, &n, &n+1);
        }
    }
}

std::vector<cv::Mat> ReadProposals(const std::string& proposalFilename) {
    std::vector<cv::Mat> proposals;

    std::ifstream f(proposalFilename);
    std::string nproposals_s;
    std::getline(f, nproposals_s);
    nproposals = stoi(nproposals_s);
    for (int i = 0; i < nproposals; ++i) {
        std::string size_line;
        std::getline(f, size_line);
        sscanf(size_line.c_str(), "%d %d", &height, &width);
        cv::Mat m(width, height, CV_32FC1);
        for (int j = 0; j < width*height; ++j) {
            std::string line;
            std::getline(f, line);
            m.at<float>(j) = stod(line);
        }
        m = m.t();
        proposals.push_back(m);
    }
    return proposals;
}

std::vector<cv::Mat> ReadUnaries(const std::string& unaryFilename) {
    std::vector<cv::Mat> unaries;

    std::ifstream f(unaryFilename);
    std::string nproposals_s;
    std::getline(f, nproposals_s);
    if (stoi(nproposals_s) != nproposals) {
        std::cout << "Number of proposals in label file " \
            "does not match proposal file\n";
        exit(-1);
    }
    for (int i = 0; i < nproposals; ++i) {
        std::string size_line;
        std::getline(f, size_line);
        int size = std::stoi(size_line);
        if (size != width*height) {
            std::cout << "Size and width*height don't match in Unary file!\n";
            std::cout << "Size: " << size << "\tWidth: " 
                << width << "\tHeight: " << height << "\n";
            exit(-1);
        }
        cv::Mat m(width, height, CV_32FC1);
        for (int j = 0; j < width*height; ++j) {
            std::string line;
            std::getline(f, line);
            m.at<float>(j) = stod(line);
        }
        m = m.t();
        unaries.push_back(m);
    }
    return unaries;
}

/*
 * higher-order-example.cpp
 *
 * Copyright 2012 Alexander Fix
 * See LICENSE.txt for license information
 *
 * Provides an example implementation of the Field of Experts denoising
 * algorithm, using a blur-and-random fusion move algorithm
 */

#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/program_options.hpp>
#include "clique.hpp"
#include "foe-cliques.hpp"
#include "higher-order-energy.hpp"
#include "fusion-move.hpp"
#include "dgfm.hpp"

double sigma = 20.0;
REAL threshold = 100.0 * DoubleToREAL;
int thresholdIters = 20;

MultilabelEnergy SetupEnergy(const std::vector<Label>& image);
void FusionProposal(int niter, const std::vector<Label>& current, std::vector<Label>& proposed);
template <typename Optimizer>
void Optimize(Optimizer& opt, const MultilabelEnergy& energy_function, cv::Mat& image, std::vector<Label>& current, int iterations);

int width = 0;
int height = 0;

int main(int argc, char **argv) {
    namespace po = boost::program_options;
    // Variables set by program options
    std::string basename;
    std::string infilename;
    std::string outfilename;
    int iterations;
    std::string method;

    po::options_description options_desc("Denoising arguments");
    options_desc.add_options()
        ("help", "Display this help message")
        ("iters,i", po::value<int>(&iterations)->default_value(300), "Maximum number of iterations")
        ("image", po::value<std::string>(&basename)->required(), "Name of image (without extension)")
    ;

    po::positional_options_description popts_desc;
    popts_desc.add("image", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
            options(options_desc).positional(popts_desc).run(), vm);

    try {
        po::notify(vm);
    } catch (std::exception& e) {
        std::cout << "Parsing error: " << e.what() << "\n";
        std::cout << "Usage: denoise [options] basename\n";
        std::cout << options_desc;
        exit(-1);
    }
    infilename = basename + ".pgm";
    outfilename = basename + "-out.pgm";

    cv::Mat image = cv::imread(infilename.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    if (!image.data) {
        std::cout << "Could not load image: " << infilename << "\n";
        exit(-1);
    }

    width = image.cols;
    height = image.rows;

    std::vector<Label> current(image.data, image.data + width*height);

    MultilabelEnergy energy_function = SetupEnergy(current);

    /*
    FusionMove<4>::ProposalCallback pc(FusionProposal);
    FusionMove<4> fusion(&energy_function, pc, current);
    Optimize(fusion, energy_function, image, current, iterations);
    */

    {
        DualGuidedFusionMove dgfm(&energy_function);
        Optimize(dgfm, energy_function, image, current, iterations);
    }

    cv::imwrite(outfilename.c_str(), image); 

    REAL energy  = energy_function.ComputeEnergy(current);
    std::cout << "Final Energy: " << energy << std::endl;

    return 0;
}

template <typename Optimizer>
void Optimize(Optimizer& opt, const MultilabelEnergy& energy_function, cv::Mat& image, std::vector<Label>& current, int iterations) {
    // energies keeps track of last [thresholdIters] energy values to know
    // when we reach convergence
    REAL energies[thresholdIters];

    for (int i = 0; i < iterations; ++i) {
        std::cout << "Iteration " << i+1 << std::endl;

        REAL energy  = energy_function.ComputeEnergy(current); 
        // check if we've reached convergence
        if (i > thresholdIters 
                && energies[i%thresholdIters] - energy < threshold) {
            break;
        }
        // Do some statistic gathering
        energies[i%thresholdIters] = energy;
        std::cout << "    Current Energy: " << (double)energy / DoubleToREAL << std::endl;

        opt.Solve(1);
        for (int i = 0; i < width*height; ++i)
            current[i] = opt.GetLabel(i);
    }

    for (int i = 0; i < width*height; ++i)
        image.data[i] = current[i];
}

void FusionProposal(int niter, const std::vector<Label>& current, std::vector<Label>& proposed) {
    // Set up the RNGs
    static boost::mt19937 rng;
    static boost::uniform_int<> uniform255(0, 255);
    static boost::uniform_int<> uniform3sigma(-1.5*sigma, 1.5*sigma);
    static boost::variate_generator<boost::mt19937&, boost::uniform_int<> > noise255(rng, uniform255);
    static boost::variate_generator<boost::mt19937&, boost::uniform_int<> > noise3sigma(rng, uniform3sigma);

    proposed.resize(height*width);
    if (niter % 2 == 0) {
        // On even iterations, proposal is a gaussian-blurred version of the 
        // current image, plus a small amount of gaussian noise
        cv::Mat image(height, width, CV_32FC1);
        for (int i = 0; i < height*width; ++i)
            image.data[i] = current[i];
        cv::Mat blur(height, width, CV_32FC1);
        cv::Size ksize(0,0);
        cv::GaussianBlur(image, blur, ksize, 3.0, 3.0, cv::BORDER_REPLICATE);
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                int n = i*width+j;
                int p = blur.data[n] + noise3sigma();
                if (p > 255) p = 255;
                if (p < 0) p = 0;
                proposed[n] = (Label)p;
            }
        }
    } else {
        // On odd iterations, proposal is a uniform random image
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                proposed[i*width+j] = (Label)(noise255());
            }
        }
    }
}

MultilabelEnergy SetupEnergy(const std::vector<Label>& image) {
    MultilabelEnergy energy(256);
    energy.AddNode(width*height);
    
    // For each 2x2 patch, add in a Field of Experts clique
    for (int i = 0; i < height - 1; ++i) {
        for (int j = 0; j < width - 1; ++j) {
            int nodes[4];
            int bufIdx = 0;
            nodes[bufIdx++] = i*width + j;
            nodes[bufIdx++] = (i+1)*width + j;
            nodes[bufIdx++] = i*width + j+1;
            nodes[bufIdx++] = (i+1)*width + j+1;
            energy.AddClique(new FoEEnergy(nodes));
        }
    }
    // Add the unary terms
    std::vector<REAL> unary(256);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            NodeId n = i*width + j;
            for (int l = 0; l < 256; ++l)
                unary[l] = FoEUnaryEnergy(image[n], l, sigma);
            energy.AddUnaryTerm(n, unary);
        }
    }
    return energy;
}


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
#include "clique.hpp"
#include "foe-cliques.hpp"
#include "higher-order-energy.hpp"
#include "fusion-move.hpp"

double sigma = 20.0;
int kernelRadius;
std::vector<double> gaussianKernel;
REAL threshold = 100.0 * DoubleToREAL;
int thresholdIters = 20;

MultilabelEnergy SetupEnergy(const std::vector<Label>& image);
void FusionProposal(int niter, const std::vector<Label>& current, std::vector<Label>& proposed);

int width = 0;
int height = 0;

int main(int argc, char **argv) {
    // Parse command arguments
    if (argc != 3){
        std::cerr << "Usage: denoise infile outfile" << std::endl;
        exit(-1);
    }

    char *infilename = argv[1];
    char *outfilename = argv[2];

    cv::Mat image = cv::imread(infilename, CV_LOAD_IMAGE_GRAYSCALE);
    if (!image.data) {
        std::cout << "Could not load image: " << infilename << "\n";
        exit(-1);
    }

    width = image.cols;
    height = image.rows;

    std::vector<Label> current(image.data, image.data + width*height);

    MultilabelEnergy energy_function = SetupEnergy(current);

    // energies keeps track of last [thresholdIters] energy values to know
    // when we reach convergence
    REAL energies[thresholdIters];

    int iterations = 300;

    FusionMove<4>::ProposalCallback pc(FusionProposal);
    FusionMove<4> fusion(&energy_function, pc, current);

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

        fusion.Solve(1);
        for (int i = 0; i < width*height; ++i)
            current[i] = fusion.GetLabel(i);
    }

    for (int i = 0; i < width*height; ++i)
        image.data[i] = fusion.GetLabel(i);

    cv::imwrite(outfilename, image); 

    REAL energy  = energy_function.ComputeEnergy(current);
    std::cout << "Final Energy: " << energy << std::endl;

    return 0;
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


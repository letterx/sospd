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
#include "foe-cliques.hpp"
#include "image.hpp"
#include "higher-order.hpp"

double sigma = 20.0;
int kernelRadius;
std::vector<double> gaussianKernel;
REAL threshold = 100.0 * DoubleToREAL;
int thresholdIters = 20;

Image_uc GetProposedImage(const Image_uc& im, unsigned int iteration, Image_uc& blur);
CliqueSystem<REAL, unsigned char, 4> SetupCliques(const Image_uc& im);
void InitGaussKernel(double sigma, int& radius, std::vector<double>& kernel);
Image_uc ApplyGaussBlur(const Image_uc& im, int radius, const std::vector<double>& kernel);

int main(int argc, char **argv) {
    // Parse command arguments
    if (argc != 3){
        std::cerr << "Usage: denoise infile outfile" << std::endl;
        exit(-1);
    }

    // Initialize the gaussian kernel for blurring the image
    InitGaussKernel(sigma, kernelRadius, gaussianKernel);

    char *infilename = argv[1];
    char *outfilename = argv[2];

    Image_uc in = ImageFromFile(infilename);
    Image_uc current, blur;
    current.Copy(in);

    // Set up the clique system, which defines the energy to be minimized
    // by fusion move
    CliqueSystem<REAL, unsigned char, 4> cliques = SetupCliques(in);

    // energies keeps track of last [thresholdIters] energy values to know
    // when we reach convergence
    REAL energies[thresholdIters];

    int iterations = 300;

    for (int i = 0; i < iterations; ++i) {
        std::cout << "Iteration " << i+1 << std::endl;

        REAL energy  = cliques.Energy(current.Data()); 
        // check if we've reached convergence
        if (i > thresholdIters 
                && energies[i%thresholdIters] - energy < threshold) {
            break;
        }
        // Do some statistic gathering
        energies[i%thresholdIters] = energy;
        std::cout << "    Current Energy: " << (double)energy / DoubleToREAL << std::endl;

        // Real work here: get proposed image, then fuse it with current image
        Image_uc proposed;
        proposed = GetProposedImage(current, i, blur);

        FusionMove(current.Height()*current.Width(), current.Data(), proposed.Data(), current.Data(), cliques);
    }
    ImageToFile(current, outfilename);
    REAL energy  = cliques.Energy(current.Data());
    std::cout << "Final Energy: " << energy << std::endl;

    return 0;
}

Image_uc GetProposedImage(const Image_uc& im, unsigned int iteration, Image_uc& blur) {
    // Set up the RNGs
    static boost::mt19937 rng;
    static boost::uniform_int<> uniform255(0, 255);
    static boost::uniform_int<> uniform3sigma(-1.5*sigma, 1.5*sigma);
    static boost::variate_generator<boost::mt19937&, boost::uniform_int<> > noise255(rng, uniform255);
    static boost::variate_generator<boost::mt19937&, boost::uniform_int<> > noise3sigma(rng, uniform3sigma);

    Image_uc proposed(im.Height(), im.Width());
    if (iteration % 2 == 0) {
        // On even iterations, proposal is a gaussian-blurred version of the 
        // current image, plus a small amount of gaussian noise
        blur = ApplyGaussBlur(im, kernelRadius, gaussianKernel);
        for (int i = 0; i < im.Height(); ++i) {
            for (int j = 0; j < im.Width(); ++j) {
                int p = (int)blur(i, j) + (int)noise3sigma();
                if (p > 255) p = 255;
                if (p < 0) p = 0;
                proposed(i, j) = (unsigned char)p;
            }
        }
    } else {
        // On odd iterations, proposal is a uniform random image
        for (int i = 0; i < im.Height(); ++i) {
            for (int j = 0; j < im.Width(); ++j) {
                proposed(i, j) = (unsigned char)(noise255());
            }
        }
    }
    return proposed;
}

CliqueSystem<REAL, unsigned char, 4> SetupCliques(const Image_uc& im) {
    CliqueSystem<REAL, unsigned char, 4> cs;
    int height = im.Height();
    int width = im.Width();
    // For each 2x2 patch, add in a Field of Experts clique
    for (int i = 0; i < height - 1; ++i) {
        for (int j = 0; j < width - 1; ++j) {
            int buf[4];
            int bufIdx = 0;
            buf[bufIdx++] = i*width + j;
            buf[bufIdx++] = (i+1)*width + j;
            buf[bufIdx++] = i*width + j+1;
            buf[bufIdx++] = (i+1)*width + j+1;
            cs.AddClique(CliqueSystem<REAL, unsigned char, 4>::CliquePointer(new FoEEnergy(4, buf)));
        }
    }
    // Add the unary terms
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int buf[1];
            buf[0] = i*width + j;
            cs.AddClique(CliqueSystem<REAL, unsigned char, 4>::CliquePointer(new FoEUnaryEnergy(buf, im(i, j))));
        }
    }
    return cs;
}

// Calculate the gaussian kernel, given a std-dev sigma
void InitGaussKernel(double sigma, int& radius, std::vector<double>& kernel) {
    radius = ceil(3*sigma); 
    kernel.reserve(2*radius + 1);
    double pi = 4.0 * atan(1.0);
    double oneOverSqrt2PiSigmaSquared = 1.0 / (sqrt(2.0 * pi) * sigma);
    double oneOverTwoSigmaSquared = 1.0 / (2.0* sigma * sigma);
    for (int i = 0; i <= radius; ++i) {
        double value = oneOverSqrt2PiSigmaSquared 
            * exp(-(i*i)*oneOverTwoSigmaSquared);
        kernel[radius+i] = value;
        kernel[radius-i] = value;
    }
    double sum = 0.0;
    for (int i = 0; i < 2*radius + 1; ++i) {
        sum += kernel[i];
    }
    for (int i = 0; i < 2*radius + 1; ++i) {
        kernel[i] = kernel[i] / sum;
    }
}

// Blur an image, given a kernel and its size
Image_uc ApplyGaussBlur(const Image_uc& im, int radius, const std::vector<double>& kernel) {
    Image_uc vertical(im.Height(), im.Width());
    for (int i = 0; i < im.Height(); ++i) {
        for (int j = 0; j < im.Width(); ++j) {
            double acc = 0.0;
            for (int k = 0; k < 2*radius+1; ++k) {
                acc += kernel[k] * im(i + k - radius, j);
            }
            vertical(i, j) = (unsigned char)acc;
        }
    }
    Image_uc horizontal(im.Height(), im.Width());
    for (int i = 0; i < im.Height(); ++i) {
        for (int j = 0; j < im.Width(); ++j) {
            double acc = 0.0;
            for (int k = 0; k < 2*radius+1; ++k) {
                acc += kernel[k] * vertical(i, j + k - radius);
            }
            horizontal(i, j) = (unsigned char)acc;
        }
    }
    return horizontal;
}

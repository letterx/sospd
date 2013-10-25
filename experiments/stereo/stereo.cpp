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
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/program_options.hpp>
#include "clique.hpp"
#include "fusion-move.hpp"
#include "dgfm.hpp"

REAL threshold = 100;
int thresholdIters = 20;

struct IterationStat {
    int iter;
    REAL start_energy;
    REAL end_energy;
    double iter_time;
    double total_time;
};

class StereoClique : public Clique {
    public:
        StereoClique(const int* nodes, 
                const std::vector<cv::Mat>& proposals) 
        : m_proposals(proposals) {
            for (int i = 0; i < 3; ++i)
                m_nodes[i] = nodes[i];
        }

        virtual REAL Energy(const Label buf[]) const override;
        virtual const NodeId* Nodes() const override { return m_nodes; }
        virtual size_t Size() const override { return 3; }

        static float kappa;
        static float alpha;
        static float scale;
    protected:
        NodeId m_nodes[3];
        const std::vector<cv::Mat>& m_proposals;
};

float StereoClique::kappa = 0.001;
float StereoClique::alpha = 10.0;
float StereoClique::scale = 20000;


REAL StereoClique::Energy(const Label buf[]) const {
    float disparity[3];
    double energy;
    for (int i = 0; i < 3; ++i)
        disparity[i] = m_proposals[buf[i]].at<float>(m_nodes[i]);
    if (std::abs(disparity[1] - disparity[0]) > alpha
            || std::abs(disparity[2] - disparity[1]) > alpha) {
        energy = kappa;
    } else {
        float curvature = disparity[0] - 2*disparity[1] + disparity[2];
        energy = std::min(curvature*curvature, kappa);
    }
    return energy*scale/kappa;
}


MultilabelEnergy SetupEnergy(const std::vector<cv::Mat>& proposals,
        const std::vector<cv::Mat>& unaries);
void FusionProposal(int niter, const std::vector<Label>& current, std::vector<Label>& proposed);
std::vector<cv::Mat> ReadUnaries(const std::string& unary_filename);
std::vector<cv::Mat> ReadProposals(const std::string& proposal_filename);
void ShowImage(const cv::Mat& im);

template <typename Optimizer>
void Optimize(Optimizer& opt, 
        const MultilabelEnergy& energy_function, 
        const std::vector<cv::Mat>& proposals,
        cv::Mat& image, 
        std::vector<Label>& current, 
        int iterations, 
        std::vector<IterationStat>& stats);

int width = 0;
int height = 0;
int nproposals = 0;

int main(int argc, char **argv) {
    namespace po = boost::program_options;
    // Variables set by program options
    std::string basename;
    std::string unary_filename;
    std::string proposal_filename;
    std::string outfilename;
    std::string stats_filename;
    int iterations;
    std::string method;
    bool spd_lower_bound;

    po::options_description options_desc("Stereo arguments");
    options_desc.add_options()
        ("help", "Display this help message")
        ("iters,i", po::value<int>(&iterations)->default_value(300), "Maximum number of iterations")
        ("image", po::value<std::string>(&basename)->required(), "Name of image (without extension)")
        ("method,m", po::value<std::string>(&method)->default_value(std::string("spd-alpha")), "Optimization method")
        ("lower-bound", po::value<bool>(&spd_lower_bound)->default_value(true), "Use lower bound for SPD3")
        ("kappa", po::value<float>(&StereoClique::kappa)->default_value(0.001), "Truncation for stereo prior")
        ("alpha", po::value<float>(&StereoClique::alpha)->default_value(10), "Max gradient for stereo prior")
        ("lambda", po::value<float>(&StereoClique::scale)->default_value(20000), "Scale for stereo prior")
    ;

    po::positional_options_description popts_desc;
    popts_desc.add("image", 1);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).
                options(options_desc).positional(popts_desc).run(), vm);
        if (vm.count("help")) {
            std::cout << options_desc;
            exit(0);
        }
        po::notify(vm);
    } catch (std::exception& e) {
        std::cout << "Parsing error: " << e.what() << "\n";
        std::cout << "Usage: denoise [options] basename\n";
        std::cout << options_desc;
        exit(-1);
    }
    unary_filename = basename + ".unary";
    proposal_filename = basename + ".proposals";
    outfilename = basename + "-" + method + ".pgm";
    stats_filename = basename + "-" + method + ".stats";

    std::cout << "Reading proposals...\n";
    std::vector<cv::Mat> proposals = ReadProposals(proposal_filename);
    std::cout << "Reading unaries...\n";
    std::vector<cv::Mat> unaries = ReadUnaries(unary_filename);
    cv::Mat image(height, width, CV_32FC1);

    std::vector<IterationStat> stats;

    std::vector<Label> current(width*height, 0);
    std::cout << "Setting up energy...\n";
    MultilabelEnergy energy_function = SetupEnergy(proposals, unaries);

    std::cout << "Optimizing...\n";
    if (method == std::string("reduction")) {
        FusionMove<4>::ProposalCallback pc(FusionProposal);
        FusionMove<4> fusion(&energy_function, pc, current);
        Optimize(fusion, energy_function, proposals, image, current, iterations, stats);
    } else if (method == std::string("spd-alpha")) {
        DualGuidedFusionMove dgfm(&energy_function);
        dgfm.SetAlphaExpansion();
        dgfm.SetLowerBound(spd_lower_bound);
        Optimize(dgfm, energy_function, proposals, image, current, iterations, stats);
    } else if (method == std::string("spd-alpha-height")) {
        DualGuidedFusionMove dgfm(&energy_function);
        dgfm.SetHeightAlphaExpansion();
        dgfm.SetLowerBound(spd_lower_bound);
        Optimize(dgfm, energy_function, proposals, image, current, iterations, stats);
    } else {
        std::cout << "Unrecognized method: " << method << "!\n";
        exit(-1);
    }

    image.convertTo(image, CV_8U, .7, 0);
    //ShowImage(image);
    cv::imwrite(outfilename.c_str(), image); 

    std::ofstream statsfile(stats_filename);
    for (const IterationStat& s : stats) {
        statsfile << s.iter << "\t";
        statsfile << s.iter_time << "\t";
        statsfile << s.total_time << "\t";
        statsfile << s.start_energy << "\t";
        statsfile << s.end_energy << "\n";
    }
    statsfile.close();


    REAL energy  = energy_function.ComputeEnergy(current);
    std::cout << "Final Energy: " << energy << std::endl;

    return 0;
}

template <typename Optimizer>
void Optimize(Optimizer& opt, 
        const MultilabelEnergy& energy_function, 
        const std::vector<cv::Mat>& proposals, 
        cv::Mat& image, 
        std::vector<Label>& current, 
        int iterations, 
        std::vector<IterationStat>& stats) {
    // energies keeps track of last [thresholdIters] energy values to know
    // when we reach convergence
    REAL energies[thresholdIters];

    REAL last_energy = energy_function.ComputeEnergy(current);
    std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::chrono::system_clock::time_point iterStartTime = std::chrono::system_clock::now();
        IterationStat s;
        s.iter = i;
        std::cout << "Iteration " << i+1 << std::endl;

        s.start_energy = last_energy;
        // check if we've reached convergence
        if (i > thresholdIters 
                && energies[i%thresholdIters] - last_energy < threshold) {
            break;
        }
        // Do some statistic gathering
        energies[i%thresholdIters] = last_energy;
        std::cout << "    Current Energy: " << last_energy << std::endl;

        opt.Solve(1);

        std::chrono::system_clock::time_point iterStopTime = std::chrono::system_clock::now();
        std::chrono::duration<double> iterTime = iterStopTime - iterStartTime;
        s.iter_time = iterTime.count();
        std::chrono::duration<double> totalTime = iterStopTime - startTime;
        s.total_time = totalTime.count();

        std::vector<Label> next_labeling(width*height);
        for (int i = 0; i < width*height; ++i)
            next_labeling[i] = opt.GetLabel(i);
        REAL energy  = energy_function.ComputeEnergy(next_labeling); 
        if (energy < last_energy) {
            last_energy = energy;
            current = next_labeling;
        }
        s.end_energy = energy;
        stats.push_back(s);

        for (int i = 0; i < height; ++i)
            for (int j = 0; j < width; ++j)
                image.at<float>(i, j) = proposals[current[i*width+j]].at<float>(i, j);
    }

    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            image.at<float>(i, j) = proposals[current[i*width+j]].at<float>(i, j);
}

void FusionProposal(int niter, const std::vector<Label>& current, std::vector<Label>& proposed) {
    proposed.resize(height*width);
    Label alpha = niter % nproposals;
    for (Label& l : proposed)
        l = alpha;
}


MultilabelEnergy SetupEnergy(const std::vector<cv::Mat>& proposals, 
        const std::vector<cv::Mat>& unary) {
    MultilabelEnergy energy(nproposals);
    energy.AddNode(width*height);
    
    // For each 1x3 patch, add in a StereoClique
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width - 2; ++j) {
            int nodes[3] = { i*width+j, i*width+j+1, i*width+j+2 };
            energy.AddClique(new StereoClique(nodes, proposals));
        }
    }
    // For each 3x1 patch, add in a StereoClique
    for (int i = 0; i < height-2; ++i) {
        for (int j = 0; j < width; ++j) {
            int nodes[3] = { i*width+j, (i+1)*width+j, (i+2)*width+j };
            energy.AddClique(new StereoClique(nodes, proposals));
        }
    }
    // Add the unary terms
    std::vector<REAL> unary_buf(nproposals);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            NodeId n = i*width + j;
            for (int l = 0; l < nproposals; ++l)
                unary_buf[l] = REAL(std::round(unary[l].at<float>(i, j)));
            energy.AddUnaryTerm(n, unary_buf);
        }
    }
    return energy;
}

std::vector<cv::Mat> ReadProposals(const std::string& proposal_filename) {
    std::vector<cv::Mat> proposals;

    std::ifstream f(proposal_filename);
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

std::vector<cv::Mat> ReadUnaries(const std::string& unary_filename) {
    std::vector<cv::Mat> unaries;

    std::ifstream f(unary_filename);
    std::string nproposals_s;
    std::getline(f, nproposals_s);
    if (stoi(nproposals_s) != nproposals) {
        std::cout << "Number of proposals in label file does not match proposal file\n";
        exit(-1);
    }
    for (int i = 0; i < nproposals; ++i) {
        std::string size_line;
        std::getline(f, size_line);
        int size = std::stoi(size_line);
        if (size != width*height) {
            std::cout << "Size and width*height don't match in Unary file!\n";
            std::cout << "Size: " << size << "\tWidth: " << width << "\tHeight: " << height << "\n";
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

void ShowImage(const cv::Mat& im) {
    cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );
    cv::imshow( "Display window", im);                   

    cv::waitKey(0);
}

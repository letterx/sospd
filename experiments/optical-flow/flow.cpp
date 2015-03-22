/*
 * flow.cpp
 *
 * Copyright 2014 Alexander Fix
 * See LICENSE.txt for license information
 */

#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <unordered_map>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/program_options.hpp>
#include "multilabel-energy.hpp"
#include "higher-order-energy.hpp"
#include "fusion-move.hpp"
#include "sospd.hpp"

typedef MultilabelEnergy::Label Label;
typedef MultilabelEnergy::VarId VarId;

struct IterationStat {
    int iter;
    REAL startEnergy;
    REAL endEnergy;
    double iterTime;
    double totalTime;
};

MultilabelEnergy SetupEnergy( const cv::Mat& im1, const cv::Mat& im2,
        int gridWidth, int gridHeight, double offsetScale, int nSamples);
void AlphaProposal(int niter, const std::vector<Label>& current,
        std::vector<Label>& proposed);

static std::vector<Label> randomAlphaOrder;

template <typename Optimizer>
void Optimize(Optimizer& opt, 
        const MultilabelEnergy& energyFunction, 
        cv::Mat& image, 
        std::vector<Label>& current, 
        int iterations);

int width = 0;
int height = 0;
double maxTime = 0;
const double maxCost = 1e10;

std::vector<cv::Point2f> ComputeGridPoints(int gridWidth, int gridHeight) {
    std::vector<cv::Point2f> points;
    for (int i = 0; i <= gridHeight; ++i)
        for (int j = 0; j <= gridWidth; ++j)
            points.emplace_back(static_cast<double>(j*(width-1))/gridWidth, static_cast<double>(i*(height-1))/gridHeight);
    for (auto& p : points) {
        assert(p.x >= 0 && p.x <= width-1 && p.y >= 0 && p.y <= height-1);
    }
    return points;
}

std::vector<cv::Point2f> ComputeOffsets(double scale, int nsamples) {
    std::vector<cv::Point2f> offsets;
    const std::vector<cv::Point2f> directions = { 
        {0.0, 1.0}, {0.0, -1.0}, {1.0, 0.0}, {-1.0, 0.0},
        {1.0, 1.0}, {1.0, -1.0}, {-1.0, 1.0}, {-1.0, -1.0}
    };
    offsets.emplace_back(0.0, 0.0);
    for (auto d : directions) {
        for (int i = 1; i <= nsamples; ++i) {
            offsets.push_back(scale * i * d);
        }
    }
    return offsets;
}

std::vector<std::array<int, 3>> TriangulationIndices(int gridWidth, int gridHeight) {
    std::vector<std::array<int, 3>> indices;
    for (int i = 0; i < gridHeight; ++i) {
        for (int j = 0; j < gridWidth; ++j) {
            int baseIdx = i*(gridWidth+1) + j;
            indices.push_back(std::array<int,3>{ baseIdx, baseIdx+1, baseIdx+gridWidth+1 });
            indices.push_back(std::array<int,3>{ baseIdx+1, baseIdx+gridWidth+2, baseIdx+gridWidth+1 });
        }
    }
    return indices;
}

std::vector<cv::Point> TriPixels(const std::array<int, 3>& triIndices, const cv::Mat& im, const std::vector<cv::Point2f>& gridPoints) {
    std::vector<cv::Point> pixels;

    std::array<cv::Point, 3> intPoints { gridPoints[triIndices[0]], gridPoints[triIndices[1]], gridPoints[triIndices[2]] };
    int maxX = 0, minX = width;
    int maxY = 0, minY = height;
    for (int i = 0; i < 3; ++i) {
        maxX = std::max(maxX, intPoints[i].x);
        maxY = std::max(maxY, intPoints[i].y);
        minX = std::min(minX, intPoints[i].x);
        minY = std::min(minY, intPoints[i].y);
    }
    std::unordered_map<int, int> maxColInRow;
    std::unordered_map<int, int> minColInRow;
    for (int row = minY; row <= maxY; ++row) {
        maxColInRow[row] = 0;
        minColInRow[row] = width;
    }

    for (int start = 0; start < 3; ++start) {
        cv::LineIterator line{im, intPoints[start], intPoints[(start+1)%3]};
        for (int i = 0; i < line.count; ++i, ++line) {
            auto pt = line.pos();
            assert(pt.x <= maxX && pt.x >= minX);
            assert(pt.y <= maxY && pt.y >= minY);
            maxColInRow[pt.y] = std::max(maxColInRow[pt.y], pt.x);
            minColInRow[pt.y] = std::min(minColInRow[pt.y], pt.x);
        }
    }
    for (int row = minY; row <= maxY; ++row) {
        assert(minColInRow[row] <= maxColInRow[row]);
        for (int col = minColInRow[row]; col <= maxColInRow[row]; ++col) {
            pixels.emplace_back(col, row);
        }
    }
    return pixels;
}

std::vector<std::vector<cv::Point>> TriangulationPixels(const cv::Mat& im, int gridWidth, int gridHeight) {
    std::vector<std::vector<cv::Point>> pixelsPerTri;
    const auto gridPoints = ComputeGridPoints(gridWidth, gridHeight);
    const auto triIndices = TriangulationIndices(gridWidth, gridHeight);
    for (const auto& tri : triIndices)
        pixelsPerTri.emplace_back(TriPixels(tri, im, gridPoints));
    return pixelsPerTri;
}

std::pair<cv::Point, cv::Point> BoundingBox(std::array<cv::Point2f, 3> tri) {
    cv::Point bbMin { std::numeric_limits<int>::max(), std::numeric_limits<int>::max() };
    cv::Point bbMax { std::numeric_limits<int>::lowest(), std::numeric_limits<int>::lowest() };
    for (int i = 0; i < 3; ++i) {
        bbMin.x = std::min(bbMin.x, static_cast<int>(floor(tri[i].x)));
        bbMin.y = std::min(bbMin.y, static_cast<int>(floor(tri[i].y)));
        bbMax.x = std::max(bbMax.x, static_cast<int>(ceil(tri[i].x)));
        bbMax.y = std::max(bbMax.y, static_cast<int>(ceil(tri[i].y)));
    }
    return { bbMin, bbMax };
}

double WarpCost(const cv::Mat& im1,
        const cv::Mat& im2,
        std::array<int, 3> triIndices,
        const std::vector<cv::Point>& triPixels,
        const std::vector<cv::Point2f>& gridPoints,
        std::array<cv::Point2f, 3> triOffsets,
        cv::Mat& scratch) {
    std::array<cv::Point2f, 3> oldTri;
    for (int i = 0; i < 3; ++i)
        oldTri[i] = gridPoints[triIndices[i]];
    auto oldBB = BoundingBox(oldTri);
    for (auto& p : oldTri) {
        p = p - cv::Point2f{ static_cast<float>(oldBB.first.x), static_cast<float>(oldBB.first.y) };
        assert(p.x >= 0 && p.x <= oldBB.second.x && p.y >= 0 && p.y <= oldBB.second.y);
    }

    std::array<cv::Point2f, 3> newTri;
    for (int i = 0; i < 3; ++i)
        newTri[i] = gridPoints[triIndices[i]] + triOffsets[i];
    // Check if all translated points are inside bounds
    for (const auto& p : newTri) {
        if (p.x < 0 || p.x > width-1 || p.y < 0 || p.y > height-1)
            return maxCost;
    }
    auto newBB = BoundingBox(newTri);
    // Translate triangle relative to bounding-box
    for (auto& p : newTri) {
        p = p - cv::Point2f{ static_cast<float>(newBB.first.x), static_cast<float>(newBB.first.y) };
        assert(p.x >= 0 && p.x <= newBB.second.x-newBB.first.x && p.y >= 0 && p.y <= newBB.second.y-newBB.first.y);
    }

    cv::Mat dstWindow = scratch(cv::Range(oldBB.first.y, oldBB.second.y+1), cv::Range(oldBB.first.x, oldBB.second.x+1));
    cv::Mat srcWindow = im2(cv::Range(newBB.first.y, newBB.second.y+1), cv::Range(newBB.first.x, newBB.second.x+1));

    auto transform = cv::getAffineTransform(newTri.data(), oldTri.data());
    cv::warpAffine(srcWindow, dstWindow, transform, dstWindow.size());

    std::array<int, 3> sumColors1 { 0, 0, 0 };
    std::array<int, 3> sumColors2 { 0, 0, 0 };
    for (const auto& pix : triPixels) {
        auto& c1 = im1.at<cv::Vec3b>(pix);
        auto& c2 = scratch.at<cv::Vec3b>(pix);
        for (int i = 0; i < 3; ++i) {
            sumColors1[i] += c1[i];
            sumColors2[i] += c2[i];
        }
    }

    int nPix = triPixels.size();

    std::array<double, 3> normColors1;
    std::array<double, 3> normColors2;
    for (int i = 0; i < 3; ++i) {
        normColors1[i] = sumColors1[i] / nPix;
        normColors2[i] = sumColors2[i] / nPix;
    }

    std::array<double, 3> diffColors1;
    std::array<double, 3> diffColors2;
    double dot = 0;
    double squaredSum1 = 0;
    double squaredSum2 = 0;
    for (const auto& pix : triPixels) {
        auto& c1 = im1.at<cv::Vec3b>(pix);
        auto& c2 = scratch.at<cv::Vec3b>(pix);
        for (int i = 0; i < 3; ++i) {
            diffColors1[i] = c1[i] - normColors1[i];
            diffColors2[i] = c2[i] - normColors2[i];
            dot += diffColors1[i]*diffColors2[i];
            squaredSum1 += diffColors1[i] * diffColors1[i];
            squaredSum2 += diffColors2[i] * diffColors2[i];
        }
    }
    double cost = dot / (sqrt(squaredSum1) * sqrt(squaredSum2));
    assert(-1.0 <= cost && cost <= 1.0);

    return 1.0 - cost;
}

std::array<float, 3> ComputeAngles(const std::array<cv::Point2f, 3>& tri) {
    std::array<float, 3> angles;
    for (int i = 0; i < 3; ++i) {
        cv::Point2f leg1 = tri[(i+1)%3] - tri[i];
        cv::Point2f leg2 = tri[(i+2)%3] - tri[i];
        auto dot = leg1.dot(leg2);
        auto cosAngle = dot/(cv::norm(leg1)*cv::norm(leg2));
        assert(-1.0 <= cosAngle && cosAngle <= 1.0);
        angles[i] = acos(cosAngle);
    }
    return angles;
}

double AngleDeviationCost(std::array<int, 3> triIndices,
        const std::vector<cv::Point2f>& gridPoints,
        std::array<cv::Point2f, 3> triOffsets) {
    std::array<cv::Point2f, 3> oldTri;
    for (int i = 0; i < 3; ++i)
        oldTri[i] = gridPoints[triIndices[i]];

    std::array<cv::Point2f, 3> newTri;
    for (int i = 0; i < 3; ++i)
        newTri[i] = gridPoints[triIndices[i]] + triOffsets[i];

    auto oldAngles = ComputeAngles(oldTri);
    auto newAngles = ComputeAngles(newTri);

    double norm = 0;
    for (int i = 0; i < 3; ++i) {
        auto diff = oldAngles[i] - newAngles[i];
        norm += diff*diff;
    }
    return sqrt(norm);
}

int main(int argc, char **argv) {
    namespace po = boost::program_options;
    // Variables set by program options
    std::string im1name;
    std::string im2name;
    int iterations;
    std::string method;
    int gridWidth;
    int gridHeight;
    int nOffsetSamples;
    double offsetScale;

    po::options_description options("Denoising arguments");
    options.add_options()
        ("help", "Display this help message")
        ("iters,i", po::value<int>(&iterations)->default_value(300), 
         "Maximum number of iterations")
        ("im1", po::value<std::string>(&im1name)->required(),
         "Name of first image in pair")
        ("im2", po::value<std::string>(&im2name)->required(),
         "Name of second image in pair")
        ("method,m",
         po::value<std::string>(&method)
            ->default_value(std::string("reduction")),
         "Optimization method")
        ("grid", po::value<int>(&gridWidth)->default_value(10),
         "Number of points in side of triangle grid")
        ("oSamples", po::value<int>(&nOffsetSamples)->default_value(5),
         "Number of samples along each direction for offsets")
        ("oScale", po::value<double>(&offsetScale)->default_value(10.0),
         "Distance along each direction per sample")
    ;

    po::positional_options_description positionalOptions;
    positionalOptions.add("im1", 1);
    positionalOptions.add("im2", 2);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
            options(options).positional(positionalOptions).run(), vm);

    try {
        po::notify(vm);
    } catch (std::exception& e) {
        std::cout << "Parsing error: " << e.what() << "\n";
        std::cout << "Usage: flow [options] im1 im2\n";
        std::cout << options;
        exit(-1);
    }
    gridHeight = gridWidth;

    cv::Mat im1 = cv::imread(im1name.c_str(), CV_LOAD_IMAGE_COLOR);
    if (!im1.data) {
        std::cout << "Could not load image: " << im1name << "\n";
        exit(-1);
    }
    if (im1.type() != CV_8UC3) {
        std::cout << "Incorrect image format: " << im1name << "\n";
        exit(-1);
    }

    cv::Mat im2 = cv::imread(im2name.c_str(), CV_LOAD_IMAGE_COLOR);
    if (!im2.data) {
        std::cout << "Could not load image: " << im2name << "\n";
        exit(-1);
    }
    if (im2.type() != CV_8UC3) {
        std::cout << "Incorrect image format: " << im2name << "\n";
        exit(-1);
    }

    width = im1.cols;
    height = im1.rows;
    if (im2.cols != width || im2.rows != height) {
        std::cout << "Images must be same size!\n";
        exit(-1);
    }

    randomAlphaOrder = std::vector<Label>(256);
    for (int i = 0; i < 256; ++i)
        randomAlphaOrder[i] = i;
    std::random_shuffle(randomAlphaOrder.begin(), randomAlphaOrder.end());

    std::vector<Label> current(0, width*height);
    MultilabelEnergy energyFunction = SetupEnergy(im1, im2,
        gridWidth, gridHeight, offsetScale, nOffsetSamples);

/*
 *    std::vector<Label> orig = current;
 *
 *    if (method == std::string("reduction")) {
 *        FusionMove<4>::ProposalCallback pc{AlphaProposal};
 *        FusionMove<4> fusion(&energyFunction, pc, current);
 *        Optimize(fusion, energyFunction, im1, current, iterations);
 *    } else if (method == std::string("hocr")) {
 *        FusionMove<4>::ProposalCallback pc{AlphaProposal};
 *        FusionMove<4> fusion(&energyFunction, pc, current);
 *        fusion.SetMethod(FusionMove<4>::Method::HOCR);
 *        Optimize(fusion, energyFunction, im1, current, iterations);
 *    } else if (method == std::string("spd-alpha")) {
 *        SoSPD dgfm(&energyFunction);
 *        dgfm.SetProposalCallback(AlphaProposal);
 *        Optimize(dgfm, energyFunction, im1, current, iterations);
 *    } else {
 *        std::cout << "Unrecognized method: " << method << "!\n";
 *        exit(-1);
 *    }
 */

/*
 *    std::ofstream statsfile("stats.txt");
 *    for (const IterationStat& s : stats) {
 *        statsfile << s.iter << "\t";
 *        statsfile << s.iterTime << "\t";
 *        statsfile << s.totalTime << "\t";
 *        statsfile << s.startEnergy << "\t";
 *        statsfile << s.endEnergy << "\n";
 *    }
 *    statsfile.close();
 *
 */

    /*
     *REAL energy  = energyFunction.computeEnergy(current);
     *std::cout << "Final Energy: " << energy << std::endl;
     */

    return 0;
}

template <typename Optimizer>
void Optimize(Optimizer& opt, const MultilabelEnergy& energyFunction,
        cv::Mat& image, std::vector<Label>& current, int iterations,
        std::vector<IterationStat>& stats) {
    typedef std::chrono::system_clock Clock;
    typedef std::chrono::duration<double> Duration;

    REAL lastEnergy = energyFunction.computeEnergy(current);
    auto startTime = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto iterStartTime = Clock::now();
        IterationStat s;
        s.iter = i;
        std::cout << "Iteration " << i+1 << std::endl;

        s.startEnergy = lastEnergy;
        // check if we've reached convergence
        std::cout << "    Current Energy: " << lastEnergy << std::endl;

        opt.Solve(1);

        s.iterTime = Duration{ Clock::now() - iterStartTime }.count();
        s.totalTime = Duration{ Clock::now() - startTime }.count();

        std::vector<Label> nextLabeling(width*height);
        for (int i = 0; i < width*height; ++i)
            nextLabeling[i] = opt.GetLabel(i);
        REAL energy  = energyFunction.computeEnergy(nextLabeling); 
        if (energy < lastEnergy) {
            lastEnergy = energy;
            current = nextLabeling;
        }
        s.endEnergy = lastEnergy;
        stats.push_back(s);
        
        if (s.totalTime > maxTime && maxTime > 0)
            break;
    }

}

void AlphaProposal(int niter, const std::vector<Label>& current,
        std::vector<Label>& proposed) {
    Label alpha = randomAlphaOrder[niter%256];
    for (Label& l : proposed)
        l = alpha;
}

MultilabelEnergy SetupEnergy(
        const cv::Mat& im1, 
        const cv::Mat& im2,
        int gridWidth,
        int gridHeight,
        double offsetScale,
        int nSamples
        ) {
    auto offsets = ComputeOffsets(offsetScale, nSamples);
    auto gridPoints = ComputeGridPoints(gridWidth, gridHeight);
    auto triIndices = TriangulationIndices(gridWidth, gridHeight);
    auto triPixels = TriangulationPixels(im1, gridWidth, gridHeight);

    int nLabels = offsets.size();
    int nVars = gridPoints.size();
    int nTris = triIndices.size();

    MultilabelEnergy energy(nLabels);
    energy.addVar(nVars);

    cv::Mat scratch { im1.rows, im1.cols, im1.type(), 0.0 };

    for (int tri = 0; tri < nTris; ++tri) {
        std::cout << "Tri: " << tri << " / " << nTris << "\n";
        std::array<int, 3> triLabels;
        for (triLabels[0] = 0; triLabels[0] < nLabels; ++triLabels[0]) {
        for (triLabels[1] = 0; triLabels[1] < nLabels; ++triLabels[1]) {
        for (triLabels[2] = 0; triLabels[2] < nLabels; ++triLabels[2]) {
            std::array<cv::Point2f, 3> triOffsets;
            for (int i = 0; i < 3; ++i)
                triOffsets[i] = offsets[triLabels[i]];
            auto cost = WarpCost(im1, im2, triIndices[tri], triPixels[tri], 
                    gridPoints, triOffsets, scratch);
            cost += AngleDeviationCost(triIndices[tri], gridPoints, triOffsets);
        }
        }
        }
    }
    
    return energy;
}


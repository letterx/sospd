#include <assert.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/program_options.hpp>

#include "submodular-ibfs.hpp"

inline REAL doubleToREAL(double d) { return static_cast<REAL>(d * 10000.0); }

static constexpr unsigned char FGD = 255;
static constexpr unsigned char BGD = 0;

static const std::vector<double> patchCosts = {2.58834, 5.55897, 9.07768, 6.47293, 5.54426, 11.1949, 6.43006, 4.91923, 8.87701, 6.93001, 11.4572, 6.64993, 12.3735, 12.6612, 11.8139, 6.54825, 11.4572, 12.1504, 11.5626, 10.4276, 12.1504, 13.7598, 10.6243, 10.3258, 10.464, 10.1763, 13.7598, 9.26, 11.968, 13.7598, 12.1504, 6.58126, 8.97232, 13.7598, 11.6804, 12.3735, 6.94636, 13.7598, 6.63372, 6.5021, 13.7598, 13.0667, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 11.8139, 10.3586, 13.7598, 13.7598, 12.6612, 11.0518, 13.7598, 9.19546, 6.51201, 9.5257, 11.3619, 13.7598, 10.0962, 10.9872, 13.7598, 10.7153, 4.85369, 5.54615, 11.2749, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 12.6612, 6.98673, 5.7294, 12.1504, 7.01222, 13.7598, 13.7598, 13.7598, 11.0518, 11.968, 13.7598, 13.7598, 13.7598, 12.3735, 13.7598, 11.968, 13.7598, 10.2941, 10.6243, 12.1504, 7.07395, 13.0667, 13.7598, 11.6804, 6.64993, 12.3735, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 11.6804, 10.5409, 13.7598, 13.7598, 13.7598, 13.7598, 13.0667, 11.968, 12.6612, 12.3735, 12.3735, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 12.3735, 10.2633, 11.1208, 12.6612, 13.7598, 11.5626, 13.0667, 13.7598, 11.1208, 6.51916, 9.08698, 12.6612, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 11.6804, 12.1504, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 12.3735, 10.6688, 13.7598, 11.1949, 12.1504, 13.7598, 13.7598, 12.6612, 13.7598, 13.0667, 11.3619, 13.7598, 11.1208, 13.7598, 13.7598, 13.7598, 11.4572, 11.8139, 13.7598, 13.7598, 13.7598, 12.1504, 13.7598, 13.7598, 13.0667, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 13.0667, 13.7598, 13.7598, 13.7598, 11.8139, 13.7598, 11.1208, 11.5626, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 10.2941, 6.434, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 12.6612, 10.4276, 6.62253, 7.07395, 13.7598, 11.4572, 11.8139, 11.8139, 13.0667, 13.7598, 10.7153, 13.7598, 11.968, 13.7598, 12.6612, 13.7598, 13.7598, 13.7598, 9.48314, 7.13574, 11.2749, 5.73921, 13.0667, 12.1504, 11.968, 6.99246, 11.8139, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 12.1504, 10.464, 13.7598, 13.7598, 13.7598, 13.7598, 11.968, 12.6612, 11.5626, 13.7598, 11.968, 13.7598, 13.7598, 13.7598, 11.4572, 13.7598, 12.6612, 13.7598, 10.464, 11.0518, 13.7598, 10.5818, 11.5626, 13.7598, 10.8154, 5.62975, 5.5653, 13.7598, 12.6612, 13.7598, 12.1504, 13.7598, 13.7598, 12.6612, 13.7598, 13.7598, 13.7598, 13.0667, 13.7598, 13.7598, 13.7598, 10.9872, 12.6612, 12.1504, 13.7598, 12.3735, 13.7598, 13.7598, 13.7598, 13.0667, 13.7598, 13.0667, 13.7598, 12.1504, 13.7598, 13.7598, 13.7598, 10.2334, 6.93543, 13.0667, 12.6612, 13.7598, 5.72875, 13.7598, 7.04321, 11.0518, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 13.0667, 10.1763, 13.7598, 12.6612, 11.5626, 10.3925, 13.7598, 7.09031, 6.64585, 11.4572, 12.6612, 13.7598, 11.1208, 12.3735, 13.7598, 11.5626, 6.46543, 12.3735, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 12.6612, 13.7598, 13.7598, 13.0667, 13.7598, 13.7598, 13.7598, 11.6804, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 12.3735, 13.7598, 13.7598, 13.7598, 10.9266, 13.7598, 13.7598, 13.7598, 13.7598, 13.0667, 13.7598, 12.3735, 11.4572, 13.7598, 12.6612, 13.7598, 12.3735, 13.0667, 11.4572, 12.6612, 12.3735, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 12.1504, 11.3619, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 10.5818, 6.4743, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 13.0667, 10.9266, 13.0667, 13.7598, 13.7598, 13.0667, 13.7598, 13.7598, 13.7598, 10.4276, 10.6243, 13.0667, 11.968, 13.7598, 13.7598, 13.7598, 13.7598, 13.7598, 12.3735, 11.8139, 13.7598, 11.8139, 13.7598, 13.7598, 13.7598, 13.7598, 6.64341, 13.0667, 13.7598, 12.1504, 7.05295, 12.6612, 12.1504, 13.7598, 13.7598, 12.6612, 13.7598, 11.1949, 13.7598, 12.1504, 13.7598, 12.1504, 9.30546, 12.3735, 10.8694, 12.1504, 7.08904, 13.0667, 5.68019, 7.02166, 10.3258, 11.6804, 13.7598, 10.464, 11.2749, 13.7598, 10.3258, 5.61484, 4.91087, 13.7598, 13.7598, 11.3619, 12.6612, 13.7598, 10.7153, 7.39334, 6.49368, 11.1208, 13.7598, 13.0667, 10.5409, 11.968, 10.2334, 11.1949, 10.464, 12.3735, 11.3619, 13.7598, 13.0667, 13.7598, 13.7598, 13.7598, 6.50563, 6.62571, 11.0518, 7.0777, 10.3586, 11.1949, 13.7598, 10.464, 6.5564, 11.1208, 12.6612, 10.3258, 10.2941, 13.0667, 13.0667, 10.9872, 12.6612, 12.1504, 13.7598, 13.7598, 12.1504, 13.7598, 13.7598, 13.7598, 6.57594, 10.5017, 11.5626, 13.7598, 6.65649, 10.6243, 7.02998, 10.464, 4.79383, 6.45664, 10.0221, 5.61716, 6.51558, 11.1208, 5.63981, 0.225307, };

/* ReadExamples
 *
 * Not the most clear way to read in the inputs and expected outputs, but is 
 * left over from previous code base
 */
void ReadExamples(const std::string& file, 
        std::vector<std::string>& names,
        std::vector<cv::Mat>& patterns,
        std::vector<cv::Mat>& labels) {
    std::ifstream main_file(file.c_str());

    std::string images_dir;
    std::string gt_dir;
    std::string line;

    size_t n = 0;

    do {
        std::getline(main_file, images_dir);
    } while (images_dir[0] == '#');
    do {
        std::getline(main_file, gt_dir);
    } while (gt_dir[0] == '#');
              
    while (main_file.good()) {
        std::getline(main_file, line);
        if (!line.empty() && line[0] != '#') {
            n++;
            cv::Mat image = cv::imread(images_dir + line, CV_LOAD_IMAGE_GRAYSCALE);
            if (!image.data) {
                std::cout << "Could not load image: " << images_dir+line << "\n";
                exit(-1);
            }
            cv::Mat gt = cv::imread(gt_dir + line, CV_LOAD_IMAGE_GRAYSCALE);
            if (!gt.data) {
                std::cout << "Could not load image: " << gt_dir+line << "\n";
                exit(-1);
            }
            assert(image.rows == gt.rows);
            assert(image.cols == gt.cols);
            assert(image.type() == CV_8UC1);
            assert(gt.type() == CV_8UC1);
            names.push_back(line);
            patterns.push_back(image);
            labels.push_back(gt);
        }
    }
    main_file.close();
}

double Loss(const cv::Mat& im1, const cv::Mat& im2) {
    assert(im1.size() == im2.size());
    double loss = 0;
    cv::Point pt;
    for (pt.y = 0; pt.y < im1.rows; ++pt.y) {
        for (pt.x = 0; pt.x < im1.cols; ++pt.x) {
            if (im1.at<unsigned char>(pt) != im2.at<unsigned char>(pt))
                loss += 1.0;
        }
    }
    loss /= (im1.rows*im1.cols);
    return loss;
}

double ComputeEnergy(const cv::Mat& im, const cv::Mat& orig, 
        double unaryScale, double cliqueScale) {
    assert(im.size() == orig.size());
    double energy = 0;

    cv::Point pt;
    for (pt.y = 0; pt.y < im.rows; ++pt.y) {
        for (pt.x = 0; pt.x < im.cols; ++pt.x) {
            int denoised = im.at<unsigned char>(pt);
            int observed = orig.at<unsigned char>(pt);
            assert(denoised == observed || abs(denoised-observed) == 255);
            auto diff = denoised-observed;
            energy += unaryScale*diff*diff;
        }
    }

    typedef uint32_t Assgn;

    cv::Point base;
    cv::Point p;
    for (base.y = 0; base.y < im.rows-2; ++base.y) {
        for (base.x = 0; base.x < im.cols-2; ++base.x) {
            Assgn a = 0;
            int i = 0;
            for (p = base; p.y <= base.y + 2; ++p.y) {
                for (p.x = base.x; p.x <= base.x + 2; ++p.x) {
                    if (im.at<unsigned char>(p) == 255) {
                        a |= 1 << i;
                    } else {
                        assert(im.at<unsigned char>(p) == 0);
                    }
                    i++;
                }
            }
            energy += cliqueScale*patchCosts[a];
        }
    }

    return energy;
}

template <typename CRF>
void AddUnaryTerms(CRF& crf, const cv::Mat& inputImage, double unaryScale) {
    cv::Point pt;
    for (pt.y = 0; pt.y < inputImage.rows; ++pt.y) {
        for (pt.x = 0; pt.x < inputImage.cols; ++pt.x) {
            auto id = pt.y * inputImage.cols + pt.x;
            int value = inputImage.at<unsigned char>(pt);
            double E0 = unaryScale*value*value;
            double E1 = unaryScale*(255-value)*(255-value);
            crf.AddUnaryTerm(id, doubleToREAL(E0), doubleToREAL(E1));
        }
    }
}

template <typename CRF>
void AddCliqueTerms(CRF& crf, const cv::Mat& inputImage, double cliqueScale) {
    typedef uint32_t Assgn;
    cv::Point base;
    cv::Point p;
    std::vector<int> vars;
    std::vector<REAL> costTable;
    for (auto& c : patchCosts)
        costTable.push_back(doubleToREAL(cliqueScale*c));
    for (base.y = 0; base.y < inputImage.rows-2; ++base.y) {
        for (base.x = 0; base.x < inputImage.cols-2; ++base.x) {
            vars.clear();
            for (p = base; p.y <= base.y + 2; ++p.y) {
                for (p.x = base.x; p.x <= base.x + 2; ++p.x) {
                    vars.push_back(p.y*inputImage.cols + p.x);
                }
            }
            crf.AddClique(vars, costTable);
        }
    }
}

template <typename CRF>
cv::Mat GetResult(CRF& crf, const cv::Mat& inputImage) {
    cv::Mat result;
    result.create(inputImage.rows, inputImage.cols, CV_8UC1);
    int id = 0;
    cv::MatIterator_<unsigned char> iter, end;
    for (iter = result.begin<unsigned char>(), 
            end = result.end<unsigned char>();
            iter != end; ++iter) {
        if (crf.GetLabel(id) == 0) *iter = BGD;
        else if (crf.GetLabel(id) == 1) *iter = FGD;
        else *iter = -1;
        id++;
    }
    return result;
}

template <typename CRF>
cv::Mat SolveDenoise(CRF& crf, 
        const cv::Mat& inputImage,
        double unaryScale,
        double cliqueScale) {
    crf.AddNode(inputImage.rows*inputImage.cols);
    AddUnaryTerms(crf, inputImage, unaryScale);
    AddCliqueTerms(crf, inputImage, cliqueScale);
    crf.Solve();
    return GetResult(crf, inputImage);
}



int main(int argc, char** argv) {
    namespace po = boost::program_options;
    // Variables set by program options
    std::string inputFile;
    std::string outputDir;
    std::string method;
    std::string ubType;
    double unaryScale = 1.0;
    double cliqueScale = 1.0;

    po::options_description options("Denoising arguments");
    options.add_options()
        ("help", "Display this help message")
        ("inputFile", po::value<std::string>(&inputFile)->required(),
         "filename for list of inputs")
        ("outputDir", po::value<std::string>(&outputDir)->default_value("denoiseOutput"),
         "Directory to store outputs")
        ("method,m",
         po::value<std::string>(&method)
            ->default_value(std::string("submodular-ibfs")),
         "Optimization method")
        ("unaryScale,u", po::value<double>(&unaryScale)->default_value(1.0),
         "Weight on unary term")
        ("cliqueScale,c", po::value<double>(&cliqueScale)->default_value(1.0),
         "Weight on clique term")
        ("ubType", po::value<std::string>(&ubType)->default_value("pairwise"),
         "Upper Bound function to use")
    ;

    po::positional_options_description positionalOptions;
    positionalOptions.add("inputFile", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
            options(options).positional(positionalOptions).run(), vm);

    try {
        po::notify(vm);
    } catch (std::exception& e) {
        std::cout << "Parsing error: " << e.what() << "\n";
        std::cout << "Usage: denoise [options] inputFile\n";
        std::cout << options;
        exit(-1);
    }

    std::vector<std::string> imageNames;
    std::vector<cv::Mat> inputImages;
    std::vector<cv::Mat> groundTruthImages;
    ReadExamples(inputFile, imageNames, inputImages, groundTruthImages);

    int nExamples = imageNames.size();
    assert(imageNames.size() == inputImages.size());
    assert(imageNames.size() == groundTruthImages.size());

    double totalLoss = 0;
    for (int i = 0; i < nExamples; ++i) {
        std::cout << "Denoising " << imageNames[i] << "...";
        std::cout.flush();
        cv::Mat result;
        if (method == std::string("submodular-ibfs")) {
            SubmodularIBFSParams params;
            bool ubFnFound = false;
            for (const auto& tuple : SoSGraph::ubParamList) {
                if (ubType == std::get<1>(tuple)) {
                    params.ub = std::get<0>(tuple);
                    ubFnFound = true;
                }
            }
            if (!ubFnFound) {
                std::cout << "Unrecognized Upper Bound Function\n";
                exit(-1);
            }
            SubmodularIBFS crf {params};
            result = SolveDenoise(crf, inputImages[i], unaryScale, cliqueScale);
        }
        double loss = Loss(result, groundTruthImages[i]);
        totalLoss += loss;
        std::cout << "\tloss: " << loss;
        std::cout << "\tEnergy: " 
            << ComputeEnergy(result, inputImages[i], 
                    unaryScale, cliqueScale);
        std::cout << "\tStarting Energy: " 
            << ComputeEnergy(inputImages[i], inputImages[i], 
                    unaryScale, cliqueScale);
        std::cout << "\n";

        std::string outfilename = outputDir + "/" + imageNames[i];
        cv::imwrite(outfilename.c_str(), result); 
    }

    std::cout << "Average loss: " << totalLoss / nExamples << "\n";

    return 0;

}

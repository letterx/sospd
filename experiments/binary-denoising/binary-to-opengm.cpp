#include <assert.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/program_options.hpp>

#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/graphicalmodel/graphicalmodel_hdf5.hxx"
#include "opengm/graphicalmodel/space/simplediscretespace.hxx"
#include "opengm/functions/fieldofexperts.hxx"
#include "opengm/operations/adder.hxx"

typedef double ValueType;
typedef opengm::SimpleDiscreteSpace<size_t, size_t> Space;
typedef opengm::GraphicalModel<double, opengm::Adder, OPENGM_TYPELIST_1(opengm::ExplicitFunction<double> ) , Space> Model;

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

template <typename GM>
void AddUnaryTerms(GM& gm, const cv::Mat& inputImage, double unaryScale) {
    std::vector<Model::FunctionIdentifier> fids(2);
    for(size_t i=0; i<2; ++i){
       const size_t shape[] = {2};
       opengm::ExplicitFunction<double> f(shape, shape + 1);
       for(size_t s = 0; s < 2; ++s) {
          ValueType dist = 255*(ValueType(s) - ValueType(i));
          f(s) = unaryScale*dist*dist;
       }
       fids[i] = gm.addFunction(f);
    }

    cv::Point pt;
    for (pt.y = 0; pt.y < inputImage.rows; ++pt.y) {
        for (pt.x = 0; pt.x < inputImage.cols; ++pt.x) {
            auto id = pt.y * inputImage.cols + pt.x;
            int value = inputImage.at<unsigned char>(pt);
            size_t variableIndices[] = { static_cast<size_t>(pt.x + pt.y*inputImage.cols)};
            if (value == 0)
                gm.addFactor(fids[0], variableIndices, variableIndices+1);
            else
                gm.addFactor(fids[1], variableIndices, variableIndices+1);
        }
    }
}

template <typename GM>
void AddCliqueTerms(GM& gm, const cv::Mat& inputImage, double cliqueScale) {
    typedef uint32_t Assgn;
    const size_t shape[] = { 2, 2, 2, 2, 2, 2, 2, 2, 2 };
    opengm::ExplicitFunction<double> f(shape, shape+9);
    size_t cliqueLabels[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    for (uint32_t a = 0; a < (1 << 9); ++a) {
        for (int i = 0; i < 9; ++i) {
            if (a & (1 << i))
                cliqueLabels[i] = 1;
            else 
                cliqueLabels[i] = 0;
        }
        f(cliqueLabels) = cliqueScale*patchCosts[a];
    }
    auto fid = gm.addFunction(f);

    cv::Point base;
    cv::Point p;
    std::vector<size_t> vars;
    for (base.y = 0; base.y < inputImage.rows-2; ++base.y) {
        for (base.x = 0; base.x < inputImage.cols-2; ++base.x) {
            vars.clear();
            for (p = base; p.y <= base.y + 2; ++p.y) {
                for (p.x = base.x; p.x <= base.x + 2; ++p.x) {
                    vars.push_back(p.y*inputImage.cols + p.x);
                }
            }
            gm.addFactor(fid, vars.begin(), vars.end());
        }
    }
}


int main(int argc, char** argv) {
    namespace po = boost::program_options;
    // Variables set by program options
    std::string inputFile;
    std::string outputDir;
    double unaryScale = 1.0;
    double cliqueScale = 1.0;

    po::options_description options("Denoising arguments");
    options.add_options()
        ("help", "Display this help message")
        ("inputFile", po::value<std::string>(&inputFile)->required(),
         "filename for list of inputs")
        ("outputDir", po::value<std::string>(&outputDir)->default_value("denoiseOutput"),
         "Directory to store outputs")
        ("unaryScale,u", po::value<double>(&unaryScale)->default_value(1.0),
         "Weight on unary term")
        ("cliqueScale,c", po::value<double>(&cliqueScale)->default_value(1.0),
         "Weight on clique term")
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

    for (int i = 0; i < nExamples; ++i) {
        auto& im = inputImages[i];
        Space space(size_t(im.rows * im.cols), 2);
        Model gm(space);
        AddUnaryTerms(gm, im, unaryScale);
        AddCliqueTerms(gm, im, cliqueScale);

        std::string outfilename = outputDir + "/" + imageNames[i] + ".h5";
        opengm::hdf5::save(gm, outfilename, "gm");  
    }

    return 0;

}

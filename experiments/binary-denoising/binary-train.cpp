#include <assert.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/program_options.hpp>

static constexpr unsigned char FGD = 255;
static constexpr unsigned char BGD = 0;

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

void CountPatches(const cv::Mat& im, std::vector<int>& patchCounts, int pWidth, int pHeight) {
    cv::Point base;
    cv::Point p;
    for (base.y = 0; base.y < im.rows-pHeight+1; ++base.y) {
        for (base.x = 0; base.x < im.cols-pWidth+1; ++base.x) {
            uint32_t a = 0;
            int i = 0;
            for (p = base; p.y < base.y + pHeight; ++p.y) {
                for (p.x = base.x; p.x < base.x + pWidth; ++p.x) {
                    if (im.at<unsigned char>(p) == FGD) {
                        a |= 1 << i;
                    } else {
                        if (im.at<unsigned char>(p) != BGD) {
                            std::cout << p << "\n";
                            std::cout << static_cast<int>(im.at<unsigned char>(p)) << "\n";
                        }
                        assert(im.at<unsigned char>(p) == BGD);
                    }
                    i++;
                }
            }
            patchCounts[a]++;
        }
    }
}

int main(int argc, char** argv) {
    namespace po = boost::program_options;
    // Variables set by program options
    std::string inputFile;
    int pWidth = 0;
    int pHeight = 0;

    po::options_description options("Denoising arguments");
    options.add_options()
        ("help", "Display this help message")
        ("inputFile", po::value<std::string>(&inputFile)->required(),
         "filename for list of inputs")
        ("w,pWidth", po::value<int>(&pWidth)->default_value(3),
         "Patch width")
        ("h,pHeight", po::value<int>(&pHeight)->default_value(3),
         "Patch height")
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

    int patchSize = pWidth*pHeight;
    assert(patchSize > 0 && patchSize < 32);
    std::vector<int> patchCounts(1 << patchSize, 0);
    for (int i = 0; i < nExamples; ++i) {
        CountPatches(groundTruthImages[i], patchCounts, pWidth, pHeight);
    }

    int nPatches = 0;
    for (auto& c : patchCounts) {
        c = std::max(c, 1);
        nPatches += c;
    }
    std::cout << "Patch counts: ";
    for (auto c : patchCounts)
        std::cout << c << ", ";
    std::cout << "\n";
    std::cout << "patchProb = {";
    for (auto c : patchCounts)
        std::cout << -log(static_cast<double>(c)/static_cast<double>(nPatches)) << ", ";
    std::cout << "}\n";

    return 0;

}

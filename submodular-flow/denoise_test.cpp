#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include "spd2.hpp"

#define NUM_LABELS 8
int val[NUM_LABELS];

int main(){
    cv::Mat image = cv::imread("test.png", CV_LOAD_IMAGE_GRAYSCALE);
    if(! image.data )                              // Check for invalid input
    {
        printf("Could not open or find the image\n");
        return -1;
    }
    
    int n = image.rows;
    int m = image.cols;
    for (int i = 0; i < NUM_LABELS; ++i) val[i] = i * 256 / NUM_LABELS;
    
    SubmodularPrimalDual2 mrf = SubmodularPrimalDual2(NUM_LABELS);
    mrf.AddNode(n * m);
    for (int i = 0; i < m * n; ++i) {
        std::vector<REAL> cost;
        for (int j = 0; j < NUM_LABELS; ++j) {
            cost.push_back(abs(val[j] - image.data[i]));
        }
        mrf.AddUnaryTerm(i, cost);
    }
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j + 1 < m; ++j) {
            std::vector<NodeId> nodes;
            nodes.push_back(i * m + j);
            nodes.push_back(i * m + j + 1);
            SubmodularPrimalDual2::CliquePtr cp(new PottsClique(nodes, 0, 10));
            mrf.AddClique(cp);
        }
    }
    
    for (int i = 0; i + 1 < n; ++i) {
        for (int j = 0; j < m; ++j) {
            std::vector<NodeId> nodes;
            nodes.push_back(i * m + j);
            nodes.push_back(i * m + j + m);
            SubmodularPrimalDual2::CliquePtr cp(new PottsClique(nodes, 0, 10));
            mrf.AddClique(cp);
        }
    }
    
    for (int i = 0; i + 1 < n; ++i) {
        for (int j = 0; j + 1 < m; ++j) {
            std::vector<NodeId> nodes;
            nodes.push_back(i * m + j);
            nodes.push_back(i * m + j + 1);
            nodes.push_back(i * m + j + m);
            nodes.push_back(i * m + j + m + 1);
            SubmodularPrimalDual2::CliquePtr cp(new PottsClique(nodes, 0, 10));
            mrf.AddClique(cp);
        }
    }
    
    mrf.ComputeRho();
    std::cout << "Rho = " << mrf.GetRho() << std::endl;
    
    mrf.Solve();
    
    
    for (int i = 0; i < m * n; ++i) image.data[i] = val[mrf.GetLabel(i)];
    
    /*
    cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Display window", image );                   // Show our image inside it.

    cv::waitKey(0);    
    */
    return 0;
}

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include "spd2.hpp"
#include "dgfm.hpp"
#include "clique.hpp"

#define NUM_LABELS 8
int val[NUM_LABELS];

int main(){
    cv::Mat image = cv::imread("medium-test.pgm", CV_LOAD_IMAGE_GRAYSCALE);
    if(! image.data )                              // Check for invalid input
    {
        printf("Could not open or find the image\n");
        return -1;
    }
    
    int n = image.rows;
    int m = image.cols;
    for (int i = 0; i < NUM_LABELS; ++i) val[i] = i * 256 / NUM_LABELS;

    MultilabelEnergy mrf(NUM_LABELS);
    
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
            mrf.AddClique(new PottsClique<2>(nodes, 0, 10));
        }
    }
    
    for (int i = 0; i + 1 < n; ++i) {
        for (int j = 0; j < m; ++j) {
            std::vector<NodeId> nodes;
            nodes.push_back(i * m + j);
            nodes.push_back(i * m + j + m);
            mrf.AddClique(new PottsClique<2>(nodes, 0, 10));
        }
    }
    
    for (int i = 0; i + 1 < n; ++i) {
        for (int j = 0; j + 1 < m; ++j) {
            std::vector<NodeId> nodes;
            nodes.push_back(i * m + j);
            nodes.push_back(i * m + j + 1);
            nodes.push_back(i * m + j + m);
            nodes.push_back(i * m + j + m + 1);
            mrf.AddClique(new PottsClique<4>(nodes, 0, 10));
        }
    }
    std::cout << "Rho = " << mrf.Rho() << std::endl;
    
    
    DualGuidedFusionMove dgfm(&mrf);
    dgfm.SetExpansionSubmodular(true);
    dgfm.Solve();
    for (int i = 0; i < m * n; ++i) image.data[i] = val[dgfm.GetLabel(i)];
    
    
    /*
    cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Display window", image );                   // Show our image inside it.

    cv::waitKey(0);    
    */
    
    return 0;
}

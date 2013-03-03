#include <iostream>
#include "higher-order.hpp"
#include "QPBO.h"
#include "gen-random.hpp"

int main(int argc, char **argv) {
    typedef int64_t REAL;
    typedef HigherOrderEnergy<REAL, 4> HOE;
    typedef typename HOE::NodeId NodeId;

    HOE ho;
    const NodeId n = 100;
    const size_t m = 100;
    const size_t k = 4;

    GenRandom(ho, n, k, m, (REAL)100, (REAL)800, (REAL)1600, 0);


    QPBO<REAL> qr(n, 0);
    ho.ToQuadratic(qr);
    qr.Solve();

    size_t labeled = 0;
    size_t ones = 0;
    for (NodeId i = 0; i < n; ++i) {
        int label = qr.GetLabel(i);
        if (label >= 0)
            labeled++;
        if (label == 1)
            ones++;
    }

    std::cout << "Labeled: " << labeled << "\n";
    std::cout << "Ones:    " << ones << "\n";
    std::cout << "Energy:  " << qr.ComputeTwiceEnergy() << "\n";


    return 0;
}

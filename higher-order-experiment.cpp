#include <iostream>
#include <boost/chrono.hpp>
#include "higher-order.hpp"
#include "QPBO.h"
#include "gen-random.hpp"
#include "submodular-flow.hpp"

int main(int argc, char **argv) {
    typedef int64_t REAL;
    typedef HigherOrderEnergy<REAL, 4> HOE;
    typedef typename HOE::NodeId NodeId;
    typedef boost::chrono::system_clock::time_point TimePt;
    typedef boost::chrono::duration<double> Duration;

    const NodeId n = 100;
    const size_t m = 100;
    const size_t k = 4;

    TimePt ho_start = boost::chrono::system_clock::now();
    HOE ho;

    GenRandom(ho, n, k, m, (REAL)100, (REAL)800, (REAL)1600, 0);


    QPBO<REAL> qr(n, 0);
    ho.ToQuadratic(qr);
    qr.Solve();

    Duration ho_time = boost::chrono::system_clock::now() - ho_start;

    TimePt sf_start = boost::chrono::system_clock::now();

    SubmodularFlow sf;
    GenRandom(sf, n, k, m, (REAL)100, (REAL)800, (REAL)1600, 0);
    sf.PushRelabel();
    sf.ComputeMinCut();

    Duration sf_time = boost::chrono::system_clock::now() - sf_start;

    size_t labeled = 0;
    size_t ones = 0;
    for (NodeId i = 0; i < n; ++i) {
        int label = sf.GetLabel(i);
        if (label != qr.GetLabel(i))
            std::cout << "**WARNING: Different labels at pixel " << i << "**\n";
        if (label >= 0)
            labeled++;
        if (label == 1)
            ones++;
    }
    ASSERT(qr.ComputeTwiceEnergy() == sf.ComputeEnergy()*2);

    std::cout << "Labeled: " << labeled << "\n";
    std::cout << "Ones:    " << ones << "\n";
    std::cout << "Energy:  " << sf.ComputeEnergy() << "\n";
    std::cout << "HO time: " << ho_time.count() << " seconds\n";
    std::cout << "SF time: " << sf_time.count() << " seconds\n";


    return 0;
}

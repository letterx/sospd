#include <iostream>
#include <chrono>
#include "energy-common.hpp"
#include "higher-order-energy.hpp"
#include "qpbo.hpp"
#include "gen-random.hpp"
#include "submodular-ibfs.hpp"

int main(int argc, char **argv) {
    typedef HigherOrderEnergy<REAL, 4> HOE;
    typedef typename HOE::NodeId NodeId;
    typedef std::chrono::system_clock Clock;
    typedef Clock::time_point TimePt;
    typedef std::chrono::duration<double> Duration;

    const NodeId n = 160000;
    const size_t m = 160000;
    const size_t k = 4;

    // Run IBFS on random instance
    TimePt ibfsStart = Clock::now();

    SubmodularIBFS ibfs;
    GenRandom(ibfs, n, k, m, (REAL)100, (REAL)800, (REAL)1600, 0);
    ibfs.IBFS();
    ibfs.ComputeMinCut();

    Duration ibfsTime = Clock::now() - ibfsStart;


    // Run FGBZ reduction on same random instance
    TimePt hoStart = Clock::now();
    HOE ho;

    GenRandom(ho, n, k, m, (REAL)100, (REAL)800, (REAL)1600, 0);

    QPBO<REAL> qr(n, 0);
    ho.ToQuadratic(qr);
    qr.Solve();

    Duration hoTime = Clock::now() - hoStart;


    size_t labeled = 0;
    size_t ones = 0;
    for (NodeId i = 0; i < n; ++i) {
        int label = ibfs.GetLabel(i);
        /*
         *if (label != qr.GetLabel(i)) {
         *    std::cout << "WARNING: Different labels at pixel " << i << "**";
         *    std::cout << "\t" << qr.GetLabel(i) 
         *        << "\t" << label << "\t" << "\n";
         *}
         */
        if (label >= 0)
            labeled++;
        if (label == 1)
            ones++;
    }
    std::cout << "Labeled:     " << labeled << "\n";
    std::cout << "Ones:        " << ones << "\n";
    std::cout << "QR Energy:   " << qr.ComputeTwiceEnergy() << "\n";
    std::cout << "IBFS Energy: " << ibfs.ComputeEnergy() << "\n";
    std::cout << "HO time:     " << hoTime.count() << " seconds\n";
    std::cout << "IBFS time:   " << ibfsTime.count() << " seconds\n";
    
    ASSERT(qr.ComputeTwiceEnergy() == ibfs.ComputeEnergy()*2);

    return 0;
}

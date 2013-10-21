#include <iostream>
#include <chrono>
#include "higher-order-energy.hpp"
#include "QPBO.h"
#include "gen-random.hpp"
#include "submodular-flow.hpp"
#include "submodular-ibfs.hpp"
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <fstream>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: recover-crash crashfile\n";
        exit(-1);
    }

    SubmodularIBFS ibfs;
    std::ifstream ifs(argv[1]);
    boost::archive::binary_iarchive ar(ifs);
    ar & ibfs;
    ibfs.IBFS();
    ibfs.ComputeMinCut();
    std::cout << "IBFS Energy: " << ibfs.ComputeEnergy() << "\n";
    

    return 0;
}

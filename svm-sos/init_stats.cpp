#include "stats.hpp"
#include <iostream>
#include <fstream>
#include <boost/archive/text_oarchive.hpp>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: init_stats statsfile\n";
        exit(-1);
    }
    std::vector<TestStats> stats_list;
    std::ofstream of(argv[1], std::ios_base::trunc);
    boost::archive::text_oarchive oa(of);
    oa & stats_list;
    return 0;
}

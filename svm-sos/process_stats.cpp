#include "stats.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: process_stats statsfile\n";
        exit(-1);
    }
    std::vector<TestStats> stats_list = TestStats::ReadStats(argv[1]);
    std::cout << "Number of stat entries: " << stats_list.size() << "\n";
    return 0;
}

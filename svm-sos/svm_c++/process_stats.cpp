#include "stats.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: process_stats statsfile\n";
        exit(-1);
    }
    std::vector<TestStats> stats_list = TestStats::ReadStats(argv[1]);
    std::cout << "Number of stat entries: " << stats_list.size() << "\n";
    for (const auto& run : stats_list) {
        std::cout << "Model: " << run.m_model_file << "\n";
        std::cout << "\tAverage Loss:   " << run.AverageLoss() << "\n";
        std::cout << "\tAverage Time:   " << run.AverageClassifyTime() << "\n";
        std::cout << "\tTrain Time:     " << run.m_train_time << "\n";
        std::cout << "\tTrain Iters:    " << run.m_train_iters << "\n";
        std::cout << "\tNum Inferences: " << run.m_num_inferences << "\n";
    }
    return 0;
}

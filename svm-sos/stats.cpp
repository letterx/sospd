#include "stats.hpp"
#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/interprocess/sync/file_lock.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

TestStats::TestStats(const std::string& data_file, const std::string& model_file, STRUCT_LEARN_PARM* sparm)
    : m_data_file(data_file),
    m_model_file(model_file),
    m_image_stats(),
    m_num_examples(0),
    m_train_time(0),
    m_train_iters(0),
    m_num_inferences(0),
    m_maxdiff(0),
    m_epsilon(0),
    m_modellength(0),
    m_slacksum(0)
{ }

void TestStats::Write(const std::string& fname) const {
    boost::interprocess::file_lock flock(fname.c_str());
    {
        boost::interprocess::scoped_lock<boost::interprocess::file_lock> l(flock);

        std::vector<TestStats> stats_list;
        try {
            std::ifstream ifs(fname);
            boost::archive::text_iarchive iar(ifs);
            iar & stats_list;
            ifs.close();
        } catch (std::exception& e) { stats_list = std::vector<TestStats>(); }

        stats_list.push_back(*this);

        std::ofstream ofs(fname, std::ios_base::trunc | std::ios_base::out);
        boost::archive::text_oarchive oar(ofs);
        oar & stats_list;
    }
}

std::vector<TestStats> TestStats::ReadStats(const std::string& fname) {
    std::ifstream ifs(fname);
    TestStats stats;
    std::vector<TestStats> stats_list;

    boost::archive::text_iarchive ar(ifs);
    ar & stats_list;

    return stats_list;
}


#include "stats.hpp"
#include <fstream>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/interprocess/sync/file_lock.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

TestStats::TestStats(const std::string& data_file, const std::string& model_file, STRUCT_LEARN_PARM* sparm)
    : m_data_file(data_file),
    m_model_file(model_file)
{ }

template <typename Archive>
void TestStats::ImageStats::serialize(Archive& ar, const unsigned int version) {
    ar & loss;
    ar & classify_time;
}

template <typename Archive>
void TestStats::serialize(Archive& ar, const unsigned int version) {
    ar & m_data_file;
    ar & m_model_file;
    ar & m_image_stats;
}

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
        } catch (boost::archive::archive_exception& e) { }

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


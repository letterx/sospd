#ifndef _STATS_HPP_
#define _STATS_HPP_

extern "C" {
#include "svm_light/svm_common.h"
#include "svm_struct/svm_struct_common.h"
}
#include <string>
#include <vector>
#include <chrono>
#include <boost/serialization/access.hpp>

class TestStats {
    public:
        typedef std::chrono::duration<double> Duration;
        typedef std::chrono::system_clock::time_point TimePt;

        TestStats() = default;
        TestStats(const std::string& data_file, const std::string& model_file, STRUCT_LEARN_PARM* sparm);

        struct ImageStats {
            double loss;
            double classify_time;

            ImageStats() = default;
            ImageStats(double _loss, Duration _classify_time) : loss(_loss), classify_time(_classify_time.count()) { }

            private:
            friend class boost::serialization::access;
            template <typename Archive>
            void serialize(Archive& ar, const unsigned int version);
        };

        void Add(const ImageStats& s) { m_image_stats.push_back(s); }
        void ResetTimer() { m_timer_start = std::chrono::system_clock::now(); }
        void StopTimer() { m_last_time = std::chrono::system_clock::now() - m_timer_start; }
        Duration LastTime() { return m_last_time; }

        void Write(const std::string& fname) const;
        static std::vector<TestStats> ReadStats(const std::string& fname);

    private:
        std::string m_data_file;
        std::string m_model_file;
        std::vector<ImageStats> m_image_stats;
        TimePt m_timer_start;
        Duration m_last_time;

        friend class boost::serialization::access;
        template <typename Archive>
        void serialize(Archive& ar, const unsigned int version);
};


#endif

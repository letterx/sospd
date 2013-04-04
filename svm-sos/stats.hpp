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
#include <boost/serialization/vector.hpp>

class TestStats {
    public:
        typedef std::chrono::duration<double> Duration;
        typedef std::chrono::system_clock::time_point TimePt;

        TestStats();

        struct ImageStats {
            std::string name;
            double loss;
            double classify_time;

            ImageStats() = default;
            ImageStats(const std::string& _name, double _loss, Duration _classify_time) 
                : name(_name), loss(_loss), classify_time(_classify_time.count()) { }

            private:
            friend class boost::serialization::access;
            template <typename Archive>
            void serialize(Archive& ar, const unsigned int version);
        };

        void Add(const ImageStats& s) { m_image_stats.push_back(s); }
        void ResetTimer() { m_timer_start = std::chrono::system_clock::now(); }
        void StopTimer() { m_last_time = std::chrono::system_clock::now() - m_timer_start; }
        Duration LastTime() { return m_last_time; }

        // Aggregation functions
        double AverageLoss() const;
        double AverageClassifyTime() const;
        double TotalClassifyTime() const;

        void Write(const std::string& fname) const;
        static std::vector<TestStats> ReadStats(const std::string& fname);

        // Training stats
        size_t m_num_examples;
        double m_train_time;
        size_t m_train_iters;
        size_t m_num_inferences;
        double m_maxdiff;
        double m_epsilon;
        double m_modellength;
        double m_slacksum;

        std::string m_model_file;
    private:
        std::vector<ImageStats> m_image_stats;
        TimePt m_timer_start;
        Duration m_last_time;

        friend class boost::serialization::access;
        template <typename Archive>
        void serialize(Archive& ar, const unsigned int version);
};

template <typename Archive>
void TestStats::ImageStats::serialize(Archive& ar, const unsigned int version) {
    ar & name;
    ar & loss;
    ar & classify_time;
}

template <typename Archive>
void TestStats::serialize(Archive& ar, const unsigned int version) {
    ar & m_model_file;
    ar & m_image_stats;
    // Serialize training stats
    ar & m_num_examples;
    ar & m_train_time;
    ar & m_train_iters;
    ar & m_num_inferences;
    ar & m_maxdiff;
    ar & m_epsilon;
    ar & m_modellength;
    ar & m_slacksum;
}


#endif

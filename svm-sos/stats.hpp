#ifndef _STATS_HPP_
#define _STATS_HPP_

#include <chrono>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>

class TestStats {
    public:
        typedef std::chrono::duration<double> Duration;
        typedef std::chrono::system_clock::time_point TimePt;
        TestStats() { }

        struct ImageStats {
            double loss;
            Duration classify_time;

            ImageStats(double _loss, Duration _classify_time) : loss(_loss), classify_time(_classify_time) { }
            private:
            friend class boost::serialization::access;
            template <typename Archive>
            void serialize(Archive& ar, const unsigned int version) {
                ar & loss;
                ar & classify_time;
            }
        };

        void Add(const ImageStats& s) { m_image_stats.push_back(s); }
        void ResetTimer() { m_timer_start = std::chrono::system_clock::now(); }
        void StopTimer() { m_last_time = std::chrono::system_clock::now() - m_timer_start; }
        Duration LastTime() { return m_last_time; }

    private:
        std::vector<ImageStats> m_image_stats;
        TimePt m_timer_start;
        Duration m_last_time;

        friend class boost::serialization::access;
        template <typename Archive>
        void serialize(Archive& ar, const unsigned int version) {
            ar & m_image_stats;
        }
};


#endif

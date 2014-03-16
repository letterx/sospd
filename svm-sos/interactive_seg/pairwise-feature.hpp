#ifndef _PAIRWISE_FEATURE_HPP_
#define _PAIRWISE_FEATURE_HPP_

#include "feature.hpp"
#include "interactive_seg.hpp"
#include <map>


class PairwiseFeature : public FeatureGroup {
    public:
    typedef FeatureGroup::Constr Constr;
    double m_scale;

    PairwiseFeature() : m_scale(1.0) { }
    explicit PairwiseFeature(double scale) : m_scale(scale) { }

    virtual size_t NumFeatures() const { return 1; }
    virtual std::vector<FVAL> Psi(const PatternData& p, const LabelData& l) const {
        std::vector<FVAL> psi = {0.0};
        auto constPairwise = [&](const unsigned char& l1, const unsigned char& l2) {
            psi[0] += m_scale*LabelDiff(l1, l2);
        };
        ImageIterate(l.m_gt, cv::Point(1.0, 0.0), constPairwise);
        ImageIterate(l.m_gt, cv::Point(0.0, 1.0), constPairwise);

        for (auto& v : psi)
            v = -v;
        return psi;
    }
    virtual void AddToOptimizer(Optimizer& crf, const PatternData& p, const double* w) const override {
        auto constPairwise = [&](long l1, long l2) {
            crf.AddPairwiseTerm(l1, l2, 0, doubleToREAL(m_scale*w[0]), doubleToREAL(m_scale*w[0]), 0);
        };
        ImageIteri(p.m_image, cv::Point(1.0, 0.0), constPairwise);
        ImageIteri(p.m_image, cv::Point(0.0, 1.0), constPairwise);
    }
    virtual Constr CollectConstrs(size_t feature_base, double constraint_scale) const {
        Constr ret;
        std::pair<std::vector<std::pair<size_t, double>>, double> c = {{{feature_base, constraint_scale}}, 0.0};
        ret.push_back(c);
        return ret;
    }
    private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) { 
        //std::cout << "Serializing PairwiseFeature\n";
        ar & boost::serialization::base_object<FeatureGroup>(*this);
        ar & m_scale;
    }
};

BOOST_CLASS_EXPORT_GUID(PairwiseFeature, "PairwiseFeature")

class ContrastPairwiseFeature : public FeatureGroup {
    public:
    typedef FeatureGroup::Constr Constr;
    double m_scale;

    ContrastPairwiseFeature() : m_scale(1.0) { }
    explicit ContrastPairwiseFeature(double scale) : m_scale(scale) { }

    virtual size_t NumFeatures() const { return 1; }
    virtual std::vector<FVAL> Psi(const PatternData& p, const LabelData& l) const {
        const cv::Mat& downW = m_downW[p.Name()];
        const cv::Mat& rightW = m_rightW[p.Name()];
        std::vector<FVAL> psi = {0.0};
        std::function<void(const double&, const double&, const unsigned char&, const unsigned char&)>
            gradientPairwise = [&](const double& d1, const double& d2, const unsigned char& l1, const unsigned char& l2)
            {
                psi[0] += m_scale*d1*LabelDiff(l1, l2);
            };
        ImageIterate(downW, l.m_gt, cv::Point(0.0, 1.0), gradientPairwise);
        ImageIterate(rightW, l.m_gt, cv::Point(1.0, 0.0), gradientPairwise);

        for (auto& v : psi)
            v = -v;
        return psi;
    }
    virtual void AddToOptimizer(Optimizer& crf, const PatternData& p, const double* w) const override {
        cv::Mat gradWeight;
        const cv::Mat& downW = m_downW[p.Name()];
        const cv::Mat& rightW = m_rightW[p.Name()];
        std::function<void(const cv::Point&, const cv::Point&)> gradientPairwise = 
            [&](const cv::Point& p1, const cv::Point& p2)
            {
                REAL weight = doubleToREAL(m_scale*w[0]*gradWeight.at<double>(p1));
                CRF::NodeId i1 = p1.y*p.m_image.cols + p1.x;
                CRF::NodeId i2 = p2.y*p.m_image.cols + p2.x;
                crf.AddPairwiseTerm(i1, i2, 0, weight, weight, 0);
            };
        gradWeight = downW;
        ImageIterp(p.m_image, cv::Point(0.0, 1.0), gradientPairwise);
        gradWeight = rightW;
        ImageIterp(p.m_image, cv::Point(1.0, 0.0), gradientPairwise);
    }
    virtual Constr CollectConstrs(size_t feature_base, double constraint_scale) const {
        Constr ret;
        std::pair<std::vector<std::pair<size_t, double>>, double> c = {{{feature_base, constraint_scale}}, 0.0};
        ret.push_back(c);
        return ret;
    }
    virtual void Evaluate(const PatternVec& patterns) {
        std::cout << "Evaluating Pairwise Features...";
        std::cout.flush();
        for (const auto& xp : patterns) {
            const PatternData& x = *xp;
            cv::Mat& downW = m_downW[x.Name()];
            cv::Mat& rightW = m_rightW[x.Name()];
            double beta = calcBeta(x.m_image);
            downW.create(x.m_image.rows, x.m_image.cols, CV_64FC1);
            rightW.create(x.m_image.rows, x.m_image.cols, CV_64FC1);
            std::function<void(const cv::Vec3b&, const cv::Vec3b&, double&, double&)> calcExpDiff = 
                [&](const cv::Vec3b& color1, const cv::Vec3b& color2, double& d1, double& d2) {
                    cv::Vec3d c1 = color1;
                    cv::Vec3d c2 = color2;
                    cv::Vec3d diff = c1-c2;
                    d1 = exp(-beta*diff.dot(diff));
                    //d1 = abs(diff[0]) + abs(diff[1]) + abs(diff[2]);
            };
            ImageIterate(x.m_image, downW, cv::Point(0.0, 1.0), calcExpDiff);
            ImageIterate(x.m_image, rightW, cv::Point(1.0, 0.0), calcExpDiff);
        }
        std::cout << "Done!\n";
    }
    virtual void SaveEvaluation(const std::string& output_dir) const {
        std::string outfile = output_dir + "/pairwise-feature.dat";
        std::ofstream os(outfile, std::ios_base::binary);
        boost::archive::binary_oarchive ar(os);
        ar & m_rightW;
        ar & m_downW;
    }
    virtual void LoadEvaluation(const std::string& output_dir) {
        std::string infile = output_dir + "/pairwise-feature.dat";
        std::ifstream is(infile, std::ios_base::binary);
        boost::archive::binary_iarchive ar(is);
        ar & m_rightW;
        ar & m_downW;
    }
    private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) { 
        //std::cout << "Serializing ContrastPairwiseFeature\n";
        ar & boost::serialization::base_object<FeatureGroup>(*this);
        ar & m_scale;
    }

    mutable std::map<std::string, cv::Mat> m_downW;
    mutable std::map<std::string, cv::Mat> m_rightW;
};

BOOST_CLASS_EXPORT_GUID(ContrastPairwiseFeature, "ContrastPairwiseFeature")

#endif

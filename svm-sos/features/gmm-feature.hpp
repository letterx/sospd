#ifndef _GMM_FEATURE_HPP_
#define _GMM_FEATURE_HPP_

#include "feature.hpp"
#include "interactive_seg_app.hpp"
#include "gmm.hpp"
#include <map>
#include <fstream>
#include <boost/serialization/map.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

class GMMFeature : public AppTraits<InteractiveSegApp>::FG {
    public:
    double m_scale;
    int grabcut_iters;
    GMMFeature() : m_scale(1.0) { }
    explicit GMMFeature(double scale, int iters) : m_scale(0.1*scale), grabcut_iters(iters) { }

    virtual size_t NumFeatures() const { return 3; }
    virtual std::vector<FVAL> Psi(const IS_PatternData& p, const IS_LabelData& l) const {
        const cv::Mat& bgdUnaries = m_bgdUnaries[p.Name()];
        const cv::Mat& fgdUnaries = m_fgdUnaries[p.Name()];

        std::vector<FVAL> psi = {0.0, 0.0, 0.0};
        cv::Point pt;
        for (pt.y = 0; pt.y < p.m_image.rows; ++pt.y) {
            for (pt.x = 0; pt.x < p.m_image.cols; ++pt.x) {
                unsigned char label = l.m_gt.at<unsigned char>(pt);
                psi[0] += bgdUnaries.at<double>(pt)*m_scale*LabelDiff(label, cv::GC_FGD);
                psi[0] += fgdUnaries.at<double>(pt)*m_scale*LabelDiff(label, cv::GC_BGD);
                ASSERT(!std::isnan(psi[0]));
                ASSERT(std::isfinite(psi[0]));
            }
        }
        ImageCIterate(p.m_tri, l.m_gt,
            [&](const unsigned char& tri_label, const unsigned char& label) {
                if (tri_label == cv::GC_BGD)
                    psi[1] += m_scale*LabelDiff(label, cv::GC_BGD);
                if (tri_label == cv::GC_FGD)
                    psi[2] += m_scale*LabelDiff(label, cv::GC_FGD);
            });
        for (auto& v : psi)
            v = -v;
        return psi;
    }
    virtual void AddToCRF(CRF& crf, const IS_PatternData& p, double* w) const {
        const cv::Mat& bgdUnaries = m_bgdUnaries[p.Name()];
        const cv::Mat& fgdUnaries = m_fgdUnaries[p.Name()];

        cv::Point pt;
        for (pt.y = 0; pt.y < p.m_image.rows; ++pt.y) {
            for (pt.x = 0; pt.x < p.m_image.cols; ++pt.x) {
                CRF::NodeId id = pt.y * p.m_image.cols + pt.x;
                double E0 = w[0]*bgdUnaries.at<double>(pt)*m_scale;
                double E1 = w[0]*fgdUnaries.at<double>(pt)*m_scale;
                if (p.m_tri.at<unsigned char>(pt) == cv::GC_BGD) E1 += w[1]*m_scale;
                if (p.m_tri.at<unsigned char>(pt) == cv::GC_FGD) E0 += w[2]*m_scale;
                crf.AddUnaryTerm(id, doubleToREAL(E0), doubleToREAL(E1));
            }
        }
    }
    virtual void Evaluate(const std::vector<IS_PatternData*>& patterns) {
        std::cout << "Evaluating GMM Features...";
        std::cout.flush();
        static constexpr double prob_epsilon = 0.00001;
        for (const IS_PatternData* xp : patterns) {
            const IS_PatternData& x = *xp;
            cv::Mat& bgdUnaries = m_bgdUnaries[x.Name()];
            cv::Mat& fgdUnaries = m_fgdUnaries[x.Name()];
            cv::Mat bgdModel, fgdModel;
            cv::Mat tmp;
            x.m_tri.copyTo(tmp);
            cv::grabCut(x.m_image, tmp, cv::Rect(), bgdModel, fgdModel, grabcut_iters, cv::GC_INIT_WITH_MASK);
            GMM bgdGMM(bgdModel), fgdGMM(fgdModel);

            bgdUnaries.create(x.m_image.rows, x.m_image.cols, CV_64FC1);
            fgdUnaries.create(x.m_image.rows, x.m_image.cols, CV_64FC1);

            cv::Point pt;
            for (pt.y = 0; pt.y < x.m_image.rows; ++pt.y) {
                for (pt.x = 0; pt.x < x.m_image.cols; ++pt.x) {
                    const cv::Vec3b& color = x.m_image.at<cv::Vec3b>(pt);
                    double bgd_prob = bgdGMM(color);
                    double fgd_prob = fgdGMM(color);
                    if (bgd_prob < prob_epsilon) bgd_prob = prob_epsilon;
                    if (fgd_prob < prob_epsilon) fgd_prob = prob_epsilon;
                    bgdUnaries.at<double>(pt) = -log(bgd_prob);
                    fgdUnaries.at<double>(pt) = -log(fgd_prob);
                }
            }
        }
        std::cout << "Done!\n";
    }
    virtual void SaveEvaluation(const std::string& output_dir) const {
        std::string outfile = output_dir + "/gmm-feature.dat";
        std::ofstream os(outfile, std::ios_base::binary);
        boost::archive::binary_oarchive ar(os);
        ar & m_bgdUnaries;
        ar & m_fgdUnaries;
    }
    virtual void LoadEvaluation(const std::string& output_dir) {
        std::string infile = output_dir + "/gmm-feature.dat";
        std::ifstream is(infile, std::ios_base::binary);
        boost::archive::binary_iarchive ar(is);
        ar & m_bgdUnaries;
        ar & m_fgdUnaries;
    }

    private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        //std::cout << "Serializing GMMFeature\n";
        ar & boost::serialization::base_object<FeatureGroup>(*this);
        ar & m_scale;
        ar & grabcut_iters;
    }

    typedef std::map<std::string, cv::Mat> UnaryMap;
    mutable UnaryMap m_bgdUnaries;
    mutable UnaryMap m_fgdUnaries;
};

BOOST_CLASS_EXPORT_GUID(GMMFeature, "GMMFeature")

#endif

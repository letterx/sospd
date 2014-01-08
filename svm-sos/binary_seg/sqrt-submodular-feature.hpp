#ifndef _SQRT_SUBMODULAR_FEATURE_HPP_
#define _SQRT_SUBMODULAR_FEATURE_HPP_

#include "interactive_seg_app.hpp"
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

class SqrtSubmodularFeature : public BinarySegApp::FG {
    public: 
    typedef std::function<void(const std::vector<unsigned char>&)> PatchFn;
    typedef uint32_t Assgn;

    static constexpr Assgn clique_size = 4;
    double m_scale;

    SqrtSubmodularFeature() : m_scale(1.0) { }
    explicit SqrtSubmodularFeature(double scale) : m_scale(100.0*scale) { }

    virtual size_t NumFeatures() const override { return 1; }
    virtual std::vector<FVAL> Psi(const BS_PatternData& p, const BS_LabelData& l) const override {
        std::vector<FVAL> psi(NumFeatures(), 0);
        PatchFn f = [&](const std::vector<unsigned char>& labels) 
        {
            int num_edges = 0;
            for (size_t i = 0; i < clique_size; ++i) {
                for (size_t j = i+1; j < clique_size; ++j) {
                    if (labels[i] != labels[j])
                        num_edges++;
                }
            }
            psi[0] += m_scale*sqrt((double)num_edges);
        };
        ImageIteratePatch(l.m_gt, cv::Point(1.0, 1.0), f);

        for (auto& v : psi)
            v = -v;
        return psi;
    }
    virtual void AddToCRF(CRF& crf, const BS_PatternData& p, double* w) const override {
        std::vector<REAL> costTable(1 << clique_size, 0);
        for (Assgn a = 0; a < (1 << clique_size); ++a) {
            int num_edges = 0;
            for (size_t i = 0; i < clique_size; ++i) {
                for (size_t j = i+1; j < clique_size; ++j) {
                    bool label_i = (a & (1 << i)) != 0;
                    bool label_j = (a & (1 << j)) != 0;
                    if (label_i != label_j) num_edges++;
                }
            }
            costTable[a] += doubleToREAL(m_scale*w[0]*sqrt((double)num_edges));
        }
        typedef std::function<void(const std::vector<CRF::NodeId>&)> Fn;
        Fn f = [&](const std::vector<CRF::NodeId>& vars)
        {
            ASSERT(vars.size() == clique_size);
            crf.AddClique(vars, costTable);
        };
        ImageIteriPatch(p.m_image, cv::Point(1.0, 1.0), f);
    }
    private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) { 
        //std::cout << "Serializing SqrtSubmodularFeature\n";
        ar & boost::serialization::base_object<FeatureGroup>(*this);
        ar & m_scale;
    }
};

BOOST_CLASS_EXPORT_GUID(SqrtSubmodularFeature, "SqrtSubmodularFeature")

#endif

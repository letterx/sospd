#ifndef _BINARY_SUBMODULAR_FEATURE_HPP_
#define _BINARY_SUBMODULAR_FEATURE_HPP_

#include "interactive_seg_app.hpp"
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

class BinarySubmodularFeature : public BinarySegApp::FG {
    public: 
    typedef BinarySegApp::FG::Constr Constr;
    typedef std::function<void(const std::vector<unsigned char>&)> PatchFn;
    typedef uint32_t Assgn;

    static constexpr Assgn clique_size = 4;
    double m_scale;

    BinarySubmodularFeature() : m_scale(1.0) { }
    explicit BinarySubmodularFeature(double scale) : m_scale(100.0*scale) { }

    virtual size_t NumFeatures() const override { return (1 << clique_size); }
    virtual std::vector<FVAL> Psi(const BS_PatternData& p, const BS_LabelData& l) const override {
        std::vector<FVAL> psi(NumFeatures(), 0);
        PatchFn f = [&](const std::vector<unsigned char>& labels) 
        {
            Assgn a = 0;
            for (size_t i = 0; i < clique_size; ++i) {
                if (labels[i] == BinarySegApp::FGD)
                    a |= 1 << i;
            }
            psi[a] += m_scale;
        };
        ImageIteratePatch(l.m_gt, cv::Point(1.0, 1.0), f);

        for (auto& v : psi)
            v = -v;
        return psi;
    }
    virtual void AddToCRF(CRF& crf, const BS_PatternData& p, double* w) const override {
        std::vector<REAL> costTable(NumFeatures(), 0);
        for (size_t i = 0; i < NumFeatures(); ++i) {
            costTable[i] = doubleToREAL(m_scale*w[i]);
        }
        typedef std::function<void(const std::vector<CRF::NodeId>&)> Fn;
        Fn f = [&](const std::vector<CRF::NodeId>& vars)
        {
            ASSERT(vars.size() == clique_size);
            crf.AddClique(vars, costTable);
        };
        ImageIteriPatch(p.m_image, cv::Point(1.0, 1.0), f);
    }
    virtual Constr CollectConstrs(size_t feature_base, double constraint_scale) const override {
        typedef std::vector<std::pair<size_t, double>> LHS;
        typedef double RHS;
        Constr ret;
        Assgn max_assgn = (1 << clique_size) - 1;
        for (Assgn s = 0; s < max_assgn; ++s) {
            for (size_t i = 0; i < clique_size; ++i) {
                Assgn si = s | (1 << i);
                if (si != s) {
                    for (size_t j = i+1; j < clique_size; ++j) {
                        Assgn t = s | (1 << j);
                        if (t != s && j != i) {
                            Assgn ti = t | (1 << i);
                            // Decreasing marginal costs, so we require
                            // f(ti) - f(t) <= f(si) - f(s)
                            // i.e. f(si) - f(s) - f(ti) + f(t) >= 0
                            LHS lhs = {{feature_base+si, constraint_scale}, {feature_base+t, constraint_scale}};
                            lhs.push_back(std::make_pair(feature_base+s, -constraint_scale));
                            lhs.push_back(std::make_pair(feature_base+ti, -constraint_scale));
                            RHS rhs = 0;
                            ret.push_back(std::make_pair(lhs, rhs));
                        }
                    }
                }
            }
        }
        return ret;
    }
    virtual double Violation(size_t base, double* w) const override {
        size_t num_constraints = 0;
        double total_violation = 0;
        Assgn max_assgn = NumFeatures();
        for (Assgn s = 0; s < max_assgn; ++s) {
            for (size_t i = 0; i < clique_size; ++i) {
                Assgn si = s | (1 << i);
                if (si != s) {
                    for (size_t j = i+1; j < clique_size; ++j) {
                        Assgn t = s | (1 << j);
                        if (t != s && j != i) {
                            Assgn ti = t | (1 << i);
                            num_constraints++;
                            // Decreasing marginal costs, so we require
                            // f(ti) - f(t) <= f(si) - f(s)
                            // i.e. f(si) - f(s) - f(ti) + f(t) >= 0
                            double violation = -w[base+si] - w[base+t];
                            violation += w[base+s];
                            violation += w[base+ti];
                            if (violation > 0) total_violation += violation;
                        }
                    }
                }
            }
        }
        //std::cout << "Num constraints: " << num_constraints <<"\n";
        //std::cout << "Min violation: " << min_violation << "\n";
        return total_violation;
    }
    private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) { 
        //std::cout << "Serializing BinarySubmodularFeature\n";
        ar & boost::serialization::base_object<FeatureGroup>(*this);
        ar & m_scale;
    }
};

BOOST_CLASS_EXPORT_GUID(BinarySubmodularFeature, "BinarySubmodularFeature")

#endif

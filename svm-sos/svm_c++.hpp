#ifndef _SVM_CXX_HPP_
#define _SVM_CXX_HPP_

extern "C" {
#include "svm_light/svm_common.h"
#include "svm_struct/svm_struct_common.h"
}
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "submodular-flow.hpp"
#include "higher-order-energy.hpp"
#include "QPBO.h"
#include "gmm.hpp"

inline REAL doubleToREAL(double d) { return (REAL)(d * 500000.0); }

class PatternData {
    public:
        PatternData(const std::string& name, const cv::Mat& im, const cv::Mat& tri, STRUCT_LEARN_PARM* sparm);
        std::string m_name;
        cv::Mat m_image;
        cv::Mat m_tri;
        cv::Mat m_bgdModel;
        GMM m_bgdGMM;
        cv::Mat m_fgdModel;
        GMM m_fgdGMM;
        double m_beta;
        cv::Mat m_fgdUnaries;
        cv::Mat m_bgdUnaries;
        cv::Mat m_downW;
        cv::Mat m_rightW;
        cv::Mat m_fgdDist;
        cv::Mat m_bgdDist;
        cv::Mat m_closerMask;
};


class LabelData {
    public:
        LabelData() = default;
        LabelData(const std::string& name, const cv::Mat& gt);
        bool operator==(const LabelData& l) const;
        double Loss(const LabelData& l, double scale) const;

        std::string m_name;
        cv::Mat m_gt;

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, const unsigned int version) {
            ar & m_gt;
        }
};

class CRF {
    public:
        typedef int NodeId;
        CRF() { }
        std::function<void(const std::vector<NodeId>& nodes, const std::vector<REAL>& costTable)> AddClique;
        std::function<void(NodeId, NodeId, REAL, REAL, REAL, REAL)> AddPairwiseTerm;
        std::function<void(NodeId, REAL, REAL)> AddUnaryTerm;
        std::function<NodeId(int)> AddNode;
        std::function<int(NodeId)> GetLabel;
        std::function<void(void)> Solve;

        // Sorry, future me. But the following is like making CRF an abstract base class,
        // which you can subclass to anything fitting the interface. Please bear with me.
        template <typename T>
        void Wrap(T* crf) {
            using namespace std::placeholders;
            AddClique = std::bind(static_cast<void(T::*)(const std::vector<NodeId>&, const std::vector<REAL>&)>(&T::AddClique), crf, _1, _2);
            AddPairwiseTerm = std::bind(static_cast<void(T::*)(NodeId, NodeId, REAL, REAL, REAL, REAL)>(&T::AddPairwiseTerm), crf, _1, _2, _3, _4, _5, _6);
            AddUnaryTerm = std::bind(static_cast<void(T::*)(NodeId, REAL, REAL)>(&T::AddUnaryTerm), crf, _1, _2, _3);
            AddNode = std::bind(static_cast<NodeId(T::*)(int)>(&T::AddNode), crf, _1);
            GetLabel = std::bind(static_cast<int(T::*)(NodeId)const>(&T::GetLabel), crf, _1);
            Solve = std::bind(static_cast<void(T::*)(void)>(&T::Solve), crf);
        }
};

class HigherOrderWrapper {
    public:
        typedef HigherOrderEnergy<REAL, 4> HO;
        typedef QPBO<REAL> QR;
        typedef typename HO::NodeId NodeId;
        HigherOrderWrapper() : ho(), qr(0, 0) { }
        void AddClique(const std::vector<NodeId>& nodes, const std::vector<REAL>& costTable) {
            ho.AddClique(nodes, costTable);
        }
        void AddPairwiseTerm(NodeId i, NodeId j, REAL E00, REAL E01, REAL E10, REAL E11) {
            qr.AddPairwiseTerm(i, j, E00, E01, E10, E11);
        }
        void AddUnaryTerm(NodeId i, REAL E0, REAL E1) {
            ho.AddUnaryTerm(i, E0, E1);
        }
        NodeId AddNode(int n) { qr.AddNode(n); return ho.AddNode(n); }
        int GetLabel(NodeId i) const { return qr.GetLabel(i); }
        void Solve() {
            //std::cout << "Solving with HigherOrderEnergy\n";
            ho.ToQuadratic(qr);
            qr.MergeParallelEdges();
            qr.Solve();
        }
    private:
        HO ho;
        QR qr;
};


class FeatureGroup;

class ModelData {
    public:
        typedef QPBO<REAL> QR;
        ModelData();

        void InitFeatures(STRUCT_LEARN_PARM* sparm);
        long NumFeatures() const;
        void InitializeCRF(CRF& crf, const PatternData& p) const;
        void AddLossToCRF(CRF& crf, const PatternData& p, const LabelData& l, double scale) const;
        LabelData* ExtractLabel(const CRF& crf, const PatternData& x) const;
        LabelData* Classify(const PatternData& x, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const;
        LabelData* FindMostViolatedConstraint(const PatternData& x, const LabelData& y, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const;
        std::vector<boost::shared_ptr<FeatureGroup>> m_features;
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, const unsigned int version) {
            ar & m_features;
        }
};

inline double LabelDiff(unsigned char l1, unsigned char l2) {
    if (l1 == cv::GC_BGD || l1 == cv::GC_PR_BGD) {
        if (l2 == cv::GC_FGD || l2 == cv::GC_PR_FGD)
            return 1.0;
        else if (l2 == (unsigned char)-1)
            return 0.5;
        else 
            return 0.0;
    } else if (l1 == cv::GC_FGD || l1 == cv::GC_PR_FGD) {
        if (l2 == cv::GC_BGD || l2 == cv::GC_PR_BGD)
            return 1.0;
        else if (l2 == (unsigned char)-1)
            return 0.5;
        else 
            return 0.0;
    } else if (l1 == l2)
        return 0.0;
    else
        return 0.5;
}



#endif


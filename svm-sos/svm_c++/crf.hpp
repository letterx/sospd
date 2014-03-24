#ifndef _CRF_HPP_
#define _CRF_HPP_

#include "submodular-flow.hpp"
#include "higher-order-energy.hpp"
#include "submodular-ibfs.hpp"
#include "qpbo.hpp"

inline REAL doubleToREAL(double d) { return (REAL)(d * 500000.0); }

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
        int GetLabel(NodeId i) const { return const_cast<QR&>(qr).GetLabel(i); }
        void Solve() {
            //std::cout << "Solving with HigherOrderEnergy\n";
            ho.ToQuadratic(qr);
            qr.MergeParallelEdges();
            qr.Solve();
            qr.ComputeWeakPersistencies();
        }
    private:
        HO ho;
        QR qr;
};

#endif

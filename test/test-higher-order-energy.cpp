#include <boost/test/unit_test.hpp>
#include <iostream>
#include "higher-order-energy.hpp"
#include "QPBO.h"

BOOST_AUTO_TEST_SUITE(basicHigherOrder)

BOOST_AUTO_TEST_CASE(sanityCheck) {
    typedef int REAL;
    typedef HigherOrderEnergy<REAL, 4> HOE;
    typedef typename HOE::NodeId NodeId;

    HOE ho;
    ho.AddNode(4);

    std::vector<NodeId> nodes = {0, 1, 2, 3};
    const size_t numAssignments = 1 << 4;
    std::vector<REAL> energyTable(numAssignments, 0);
    energyTable[0xf] = -1;
    ho.AddClique(nodes, energyTable);

    QPBO<int> qr(4,0);
    ho.ToQuadratic(qr);
    qr.Solve();

    BOOST_CHECK_EQUAL(qr.ComputeTwiceEnergy(), -2);
    for (NodeId i = 0; i < 4; ++i) {
        BOOST_CHECK_EQUAL(qr.GetLabel(i), 1);
    }
}

BOOST_AUTO_TEST_SUITE_END()


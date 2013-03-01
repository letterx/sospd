#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SubmodularFlow
#include <boost/test/unit_test.hpp>
#include <iostream>
#include "submodular-flow.hpp"

BOOST_AUTO_TEST_SUITE(basicSubmodularFlow)

BOOST_AUTO_TEST_CASE(sanityCheck) {
    typedef SubmodularFlow::NodeId NodeId;

    SubmodularFlow sf;
    sf.AddNode(4);

    std::vector<NodeId> nodes = {0, 1, 2, 3};
    const size_t numAssignments = 1 << 4;
    std::vector<REAL> energyTable(numAssignments, 0);
    energyTable[0xf] = -1;
    sf.AddClique(nodes, energyTable);

    sf.PushRelabel();
    sf.ComputeMinCut();

    BOOST_CHECK_EQUAL(sf.ComputeEnergy(), -1);
    for (NodeId i = 0; i < 4; ++i) {
        BOOST_CHECK_EQUAL(sf.GetLabel(i), 1);
    }
}

BOOST_AUTO_TEST_SUITE_END()


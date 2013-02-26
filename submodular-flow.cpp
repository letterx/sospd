#include "submodular-flow.hpp"

SubmodularFlow::SubmodularFlow() {
    // Implement me (Alex)
}

SubmodularFlow::NodeId SubmodularFlow::AddNode(int n) {
    // Implement me (Alex)
}

int SubmodularFlow::GetLabel(NodeId n) const {
    // Implement me (Alex)
}

void SubmodularFlow::AddUnaryTerm(NodeId n, REAL E0, REAL E1) {
    // Implement me (Alex)
}

void SubmodularFlow::AddClique(const CliquePtr& cp) {
    // Implement me (Alex)
}

void SubmodularFlow::PushRelabel() {
    // Implement me (Sam)
}

void SubmodularFlow::ComputeMinCut() { 
    // Implement me (Sam)
}

REAL EnergyTableClique::ComputeEnergy(const std::vector<int>& labels) const {
    // Implement me (Alex)
}

REAL EnergyTableClique::ExchangeCapacity(NodeId u, NodeId v) const {
    // Implement me (Alex)
}

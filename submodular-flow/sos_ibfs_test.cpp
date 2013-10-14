#include <cstdio>
#include "submodular-ibfs.hpp"
#include "submodular-flow.hpp"

typedef boost::shared_ptr<IBFSEnergyTableClique> CliquePtr;
typedef std::vector<CliquePtr> CliqueVec;
typedef int NodeId;

int main(){
	SubmodularIBFS crf;
	crf.AddNode(3);
	crf.AddUnaryTerm(0, 12, 6);
	crf.AddUnaryTerm(1, 8, 8);
	crf.AddUnaryTerm(2, 6, 12);
	NodeId node_array[3] = {0, 1, 2};
	std::vector<NodeId> node(node_array, node_array + 3);
    std::vector<REAL> energy = {0, 3, 1, 2, 0, 2, 0, 0};
	crf.AddClique(node, energy);
	crf.Solve();
	std::cout << "S-T Cut: ";
    for (int i = 0; i < 3; ++i) std::cout << crf.GetLabel(i) << " ";
    std::cout << std::endl;
    std::vector<REAL> C_si = crf.GetC_si();
    std::cout << "C_si: ";
    for (int i = 0; i < 3; ++i) std::cout << C_si[i] << " ";
    std::cout << std::endl;
    std::vector<REAL> phi_si = crf.GetPhi_si();
    std::cout << "Phi_si: ";
    for (int i = 0; i < 3; ++i) std::cout << phi_si[i] << " ";
    std::cout << std::endl;
    std::vector<REAL> C_it = crf.GetC_it();
    std::cout << "C_it: ";
    for (int i = 0; i < 3; ++i) std::cout << C_it[i] << " ";
    std::cout << std::endl;
    std::vector<REAL> phi_it = crf.GetPhi_it();
    std::cout << "Phi_it: ";
    for (int i = 0; i < 3; ++i) std::cout << phi_it[i] << " ";
    std::cout << std::endl;
    CliqueVec clique = crf.GetCliques();
    CliquePtr c = clique[0];
    std::vector<REAL> phiCi = c->AlphaCi();
    std::cout << "Phi_Ci: ";
    for (int i = 0; i < 3; ++i) std::cout << phiCi[i] << " ";
    std::cout << std::endl;
    std::cout << "Energy: " << crf.ComputeEnergy() << std::endl;
    std::vector<int> label;
    label.push_back(1);
    label.push_back(1);
    label.push_back(0);
    std::cout << crf.ComputeEnergy(label) << std::endl;
	return 0;
}

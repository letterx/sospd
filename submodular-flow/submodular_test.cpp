#include <iostream>
#include <vector>
#include <string>

#include "submodular-functions.hpp"

void PrintEnergy(const std::vector<REAL>& energy) {
    std::cout << "{";
    for (REAL e : energy)
        std::cout << e << ", ";
    std::cout << "}";
}

int main(int argc, char **argv) {
    int n = 3;
    std::vector<REAL> energy = {4, 2, 0, 1, 3, 6, 4, 7};
    std::cout << "Energy:          ";
    PrintEnergy(energy);
    std::cout << "\n";

    std::vector<REAL> submodular = SubmodularLowerBound(n, energy);
    std::cout << "Lower Bound:     ";
    PrintEnergy(submodular);
    std::cout << "\n";

    std::cout << "Submodular:      " << CheckSubmodular(n, submodular) << "\n";

    std::vector<REAL> upper = SubmodularUpperBound(n, energy);
    std::cout << "Upper bound:     ";
    PrintEnergy(upper);
    std::cout << "\n";

    std::cout << "Submodular:      " << CheckSubmodular(n, upper) << "\n";
    std::cout << "Invariants hold: " << CheckUpperBoundInvariants(n, energy, upper) << "\n";

    return 0;
}

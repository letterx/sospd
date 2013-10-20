#include "submodular-functions.hpp"

#include <cassert>
#include <limits>
#include <iostream>


static inline Assgn NextPerm(Assgn v) {
    Assgn t = v | (v - 1); // t gets v's least significant 0 bits set to 1
    // Next set to 1 the most significant bit to change,
    // set to 0 the least significant ones, and add the necessary 1 bits.
    return (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctz(v) + 1));
}

std::vector<REAL> SubmodularUpperBound(int n, const std::vector<REAL>& energyTable) {
    assert(n < 32);
    REAL max_diff = 0;
    REAL max_energy = 0;

    std::vector<REAL> result = SubmodularLowerBound(n, energyTable);

    for (std::size_t i = 0; i < energyTable.size(); ++i) {
        max_diff = std::max(max_diff, energyTable[i] - result[i]);
        max_energy = std::max(max_energy, energyTable[i]);
    }

    for (REAL& e : result) {
        e += max_diff;
    }
    result[0] = energyTable[0];
    result.back() = energyTable.back();

    // Truncate singelton sets to max_energy, and then find a new lower-bound
    // (guaranteed to still upper-bound original function)
    for (int k = 0; k < n; ++k)
        result[1<<k] = std::min(result[1<<k], max_energy);
    result = SubmodularLowerBound(n, result);

    //assert(CheckSubmodular(n, result));
    return result;
}

std::vector<REAL> SubmodularLowerBound(int n, const std::vector<REAL>& energyTable) {
    assert(n < 32);
    Assgn max_assgn = 1 << n;
    assert(energyTable.size() == max_assgn);
    std::vector<REAL> result = energyTable;

    // Need to iterate over all k bit subsets in increasing k
    for (int k = 0; k < n; ++k) {
        Assgn bound;
        if (k == 0) bound = 0;
        else bound = max_assgn - 1;
        Assgn s = (1 << k) - 1;
        do {
            for (int i = 0; i < n; ++i) {
                Assgn s_i = s | (1 << i);
                if (s_i == s) continue;
                for (int j = i+1; j < n; ++j) {
                    Assgn s_j = s | (1 << j);
                    if (s_j == s) continue;
                    Assgn s_ij = s_i | s_j;
                    REAL submodularity = result[s] + result[s_ij]
                        - result[s_i] - result[s_j];
                    if (submodularity > 0) {
                        result[s_ij] -= submodularity;
                    }
                }
            }
            s = NextPerm(s);
        } while (s < bound);
    }
    return result;
}

std::vector<REAL> ZeroMarginalSet(int n, const std::vector<REAL>& energyTable, Assgn s) {
    std::vector<REAL> result(energyTable.size());
    Assgn base_set = (1 << n) - 1;
    Assgn not_s = base_set & (~s);
    for (Assgn t = 0; t <= base_set; ++t)
        result[t] = energyTable[t & not_s];
    return result;
}

void AddLinear(int n, std::vector<REAL>& energyTable, const std::vector<REAL>& psi) {
    Assgn max_assgn = 1 << n;
    assert(max_assgn == energyTable.size());
    assert(n == int(psi.size()));
    for (Assgn a = 0; a < max_assgn; ++a) {
        for (int i = 0; i < n; ++i) {
            if (a & (1 << i))
                energyTable[a] += psi[i];
        }
    }
}

void SubtractLinear(int n, std::vector<REAL>& energyTable, 
        const std::vector<REAL>& psi1, const std::vector<REAL>& psi2) {
    Assgn max_assgn = 1 << n;
    assert(max_assgn == energyTable.size());
    assert(n == int(psi1.size()));
    assert(n == int(psi2.size()));
    for (Assgn a = 0; a < max_assgn; ++a) {
        for (int i = 0; i < n; ++i) {
            if (a & (1 << i))
                energyTable[a] -= psi1[i];
            else
                energyTable[a] -= psi2[i];
        }
    }
}

void Normalize(int n, std::vector<REAL>& energyTable, std::vector<REAL>& psi) {
    Assgn max_assgn = 1 << n;
    assert(max_assgn == energyTable.size());
    assert(n == int(psi.size()));
    assert(energyTable[0] == 0);
    Assgn last_assgn = 0;
    Assgn this_assgn = 0;
    for (int i = 0; i < n; ++i) {
        this_assgn |= (1 << i);
        psi[i] = energyTable[last_assgn] - energyTable[this_assgn];
        last_assgn = this_assgn;
    }
    AddLinear(n, energyTable, psi);

    for (REAL e : energyTable)
        assert(e >= 0);
    assert(energyTable[0] == 0);
    assert(energyTable[max_assgn-1] == 0);
}

bool CheckSubmodular(int n, const std::vector<REAL>& energyTable) {
    assert(n < 32);
    Assgn max_assgn = 1 << n;
    assert(energyTable.size() == max_assgn);

    for (Assgn s = 0; s < max_assgn; ++s) {
        for (int i = 0; i < n; ++i) {
            Assgn s_i = s | (1 << i);
            if (s_i == s) continue;
            for (int j = i+1; j < n; ++j) {
                Assgn s_j = s | (1 << j);
                if (s_j == s) continue;
                Assgn s_ij = s_i | s_j;

                REAL submodularity = energyTable[s] + energyTable[s_ij]
                    - energyTable[s_i] - energyTable[s_j];
                if (submodularity > 0) {
                    std::cout << "Nonsubmodular: (" << s << ", " << i << ", " << j << "): ";
                    std::cout << energyTable[s] << " " << energyTable[s_i] << " "
                        << energyTable[s_j] << " " << energyTable[s_ij] << "\n";
                }
            }
        }
    }
    return true;
}

bool CheckUpperBoundInvariants(int n, const std::vector<REAL>& energyTable,
        const std::vector<REAL>& upperBound) {
    int energy_len = energyTable.size();
    assert(energy_len == int(upperBound.size()));
    REAL max_energy = std::numeric_limits<REAL>::min();
    for (int i = 0; i < energy_len; ++i) {
        if (energyTable[i] > upperBound[i])
            return false;
        max_energy = std::max(energyTable[i], max_energy);
    }
    for (int i = 0; i < n; ++i) {
        if (upperBound[1 << i] > max_energy)
            return false;
    }
    return CheckSubmodular(n, upperBound);
}

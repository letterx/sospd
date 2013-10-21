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

void SubmodularUpperBound(int n, std::vector<REAL>& energyTable) {
    assert(n < 32);
    REAL max_energy = 0;
    REAL zero_energy = energyTable[0];
    REAL last_energy = energyTable.back();
    for (std::size_t i = 0; i < energyTable.size(); ++i) {
        max_energy = std::max(max_energy, energyTable[i]);
    }

    REAL max_diff = SubmodularLowerBound(n, energyTable);
    //assert(CheckSubmodular(n, energyTable));

    for (REAL& e : energyTable) {
        e += max_diff;
    }
    energyTable[0] = zero_energy;
    energyTable.back() = last_energy;

    // Truncate singelton sets to max_energy, and then find a new lower-bound
    // (guaranteed to still upper-bound original function)
    for (int k = 0; k < n; ++k)
        energyTable[1<<k] = std::min(energyTable[1<<k], max_energy);
    SubmodularLowerBound(n, energyTable);

    //assert(CheckSubmodular(n, energyTable));
}

REAL SubmodularLowerBound(int n, std::vector<REAL>& energyTable, bool early_finish) {
    assert(n < 32);
    Assgn max_assgn = 1 << n;
    assert(energyTable.size() == max_assgn);
    REAL max_diff = 0;

    // Need to iterate over all k bit subsets in increasing k
    for (int k = 1; k <= n; ++k) {
        bool changed = false;
        Assgn bound;
        if (k == 0) bound = 0;
        else bound = max_assgn - 1;
        Assgn s = (1 << k) - 1;
        do {
            REAL subtract_from_s = 0;
            for (int i = 0; i < n; ++i) {
                Assgn s_i = s ^ (1 << i); // Set s - i
                if (s_i >= s) continue;
                for (int j = i+1; j < n; ++j) {
                    Assgn s_j = s ^ (1 << j); // Set s - j
                    if (s_j >= s) continue;
                    Assgn s_ij = s_i & s_j;
                    REAL submodularity = energyTable[s] + energyTable[s_ij]
                        - energyTable[s_i] - energyTable[s_j];
                    if (submodularity > subtract_from_s) {
                        subtract_from_s = submodularity;
                    }
                }
            }
            energyTable[s] -= subtract_from_s;
            changed |= (subtract_from_s > 0);
            max_diff = std::max(max_diff, subtract_from_s);
            s = NextPerm(s);
        } while (s < bound);
        if (early_finish && !changed)
            break;
    }
    return max_diff;
}

void ZeroMarginalSet(int n, std::vector<REAL>& energyTable, Assgn s) {
    Assgn base_set = (1 << n) - 1;
    Assgn not_s = base_set & (~s);
    for (Assgn t = 0; t <= base_set; ++t)
        energyTable[t] = energyTable[t & not_s];
}

void AddLinear(int n, std::vector<REAL>& energyTable, const std::vector<REAL>& psi) {
    Assgn max_assgn = 1 << n;
    assert(max_assgn == energyTable.size());
    assert(n == int(psi.size()));
    REAL sum = 0;
    Assgn last_gray = 0;
    for (Assgn a = 1; a < max_assgn; ++a) {
        Assgn gray = a ^ (a >> 1);
        Assgn diff = gray ^ last_gray;
        int changed_bit = __builtin_ctz(diff);
        if (gray & diff)
            sum += psi[changed_bit];
        else
            sum -= psi[changed_bit];
        energyTable[gray] += sum;
        last_gray = gray;
    }
}

void SubtractLinear(int n, std::vector<REAL>& energyTable, 
        const std::vector<REAL>& psi1, const std::vector<REAL>& psi2) {
    Assgn max_assgn = 1 << n;
    assert(max_assgn == energyTable.size());
    assert(n == int(psi1.size()));
    assert(n == int(psi2.size()));
    REAL sum = 0;
    for (int i = 0; i < n; ++i)
        sum += psi2[i];
    energyTable[0] -= sum;
    Assgn last_gray = 0;
    for (Assgn a = 1; a < max_assgn; ++a) {
        Assgn gray = a ^ (a >> 1);
        Assgn diff = gray ^ last_gray;
        int changed_idx = __builtin_ctz(diff);
        if (gray & diff)
            sum += psi1[changed_idx] - psi2[changed_idx];
        else
            sum += psi2[changed_idx] - psi1[changed_idx];
        energyTable[gray] -= sum;
        last_gray = gray;
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
                    return false;
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

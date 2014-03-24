/*
 * foe-cliques.hpp
 * 
 * Copyright 2012 Alexander Fix
 * See LICENSE.txt for license information
 *
 * Implementation of the Field of Experts cliques
 */
#include "foe-cliques.hpp"

double alpha[3] = {0.586612685392731, 1.157638405566669, 0.846059486257292};
double expert[3][4] = {
	{-0.0582774013402734, 0.0339010363051084,
        -0.0501593018104054, 0.0745568557931712},
	{0.0492112815304123, -0.0307820846538285,
        -0.123247230948424, 0.104812330861557},
	{0.0562633568728865, 0.0152832583489560,
        -0.0576215592718086, -0.0139673758425540}
};


REAL FoEEnergy::energy(const Label* buf) const {
    double energy = 0.0;
    for (int i = 0; i < 3; ++i) {
        double dot = 0.0;
        for (int j = 0; j < 4; ++j) {
            dot += expert[i][j] * buf[j];
        }
        energy += alpha[i] * log(1 + 0.5 * dot * dot);
    }
    return energy*DoubleToREAL;
}

REAL FoEUnaryEnergy(unsigned char orig, unsigned char label, double sigma) {
        double dist = (double)orig - (double)label;
        double e = dist*dist / (sigma*sigma * 2);
        return DoubleToREAL * e;
}

void AddFoEGrad(const Clique& clique, const std::vector<Label>& current,
        std::vector<double>& grad) {
    Label values[4];
    for (int i = 0; i < 4; ++i)
        values[i] = current[clique.nodes()[i]];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            double dot = 0.0;
            for (int k = 0; k < 4; ++k)
                dot += expert[j][k] * values[k];
            grad[clique.nodes()[i]] += alpha[j] * expert[j][i] 
                * dot / (1 + 0.5 * dot * dot);
        }
    }
}

double FoEUnaryGrad(Label orig, Label current, double sigma) {
    return (double(current) - double(orig)) / (sigma*sigma);
}

#!/usr/bin/env python

import sys

if len(sys.argv) != 2:
    print "Usage: process-results.py results-dir"
    sys.exit(-1)

resultsDir = sys.argv[1]
files = [l.strip() for l in open(resultsDir + "/files.txt")]
methods = [l.strip() for l in open(resultsDir + "/methods.txt")]

times = {}
energy = {}
for f in files:
    times[f] = {}
    energy[f] = {}
    for m in methods:
        resultFile = open(resultsDir + "/" + f + "_" + m + ".txt")
        header = resultFile.readline()
        while header != "":
            line = resultFile.readline()
            if header == "%times\n":
                times[f][m] = [float(entry) for entry in line.strip()[:-1].split(';')]
                times[f][m] = times[f][m][1:]
            elif header == "%values\n":
                energy[f][m] = [float(entry) for entry in line.strip()[:-1].split(';')]
                energy[f][m] = energy[f][m][1:]
            elif header == "%states\n":
                break
            header = resultFile.readline()


avgEnergy = {}
avgTime = {}

for m in methods:
    methodEnergy = 0
    methodTime = 0
    for f in files:
        methodEnergy += energy[f][m][-1]
        methodTime += times[f][m][-1]
    avgEnergy[m] = methodEnergy/len(files)
    avgTime[m] = methodTime/len(files)

best = {}
for m in methods:
    best[m] = 0
for f in files:
    bestEnergy = float("inf")
    for m in methods:
        bestEnergy = min(bestEnergy, energy[f][m][-1])
    for m in methods:
        if energy[f][m][-1] == bestEnergy:
            best[m] += 1
for m in methods:
    best[m] = 100*float(best[m])/float(len(files))

methodToName = { 
        "sospd_pairwise":"SoSPD Pair",
        "sospd_chen":"SoSPD OldCVPR",
        "sospd_cardinality":"SoSPD Card",
        "sospd_gurobiL1":"SoSPD L1",
        "sospd_gurobiL2":"SoSPD L2",
        "sospd_gurobiLInfty":"SoSPD LInfty",
        "sospd_cvpr14":"SoSPD CVPR14",
        "sospd_triple":"SoSPD L1 (3 Cliques)",
        "sospd_tripleInfty":"SoSPD LInfty (3 Cliques)",
        "trws":"OGM-TRWS",
        "icm":"OGM-ICM",
        "bp":"OGM-BP",
        "trbp":"OGM-TRBP",
        "lazy_flipper":"OGM-LF",
        "inf_and_flip":"OGM-Inf-and-flip",
        "ad3":"OGM-AD3",
        "reduction_fusion":"OGM-Alpha-Fusion",
        "sos_pair_fusion":"SoS-Fusion Pair",
        "sos_cvpr_fusion":"SoS-Fusion CVPR14",
        "sos_chen_fusion":"SoS-Fusion OldCVPR",
        "sos_cardinality_fusion":"SoS-Fusion Card",
        "sos_triple_fusion":"SoS-Fusion L1 (3 Cliques)",
        "sos_tripleInfty_fusion":"SoS-Fusion LInfty (3 Cliques)",
        "sospd_pairwise_grad":"SoSPD-Grad Pair",
        "sospd_chen_grad":"SoSPD-Grad OldCVPR",
        "sospd_cardinality_grad":"SoSPD-Grad Card",
        "sospd_gurobiL1_grad":"SoSPD-Grad L1",
        "sospd_gurobiL2_grad":"SoSPD-Grad L2",
        "sospd_gurobiLInfty_grad":"SoSPD-Grad LInfty",
        "sospd_cvpr14_grad":"SoSPD-Grad CVPR14",
        "sospd_latPair_grad":"SoSPD-Grad LatticePair",
        "sospd_latCard_grad":"SoSPD-Grad LatticeCard",
        "reduction_fusion_grad":"OGM-Grad-Fusion",
        "sos_pair_fusion_grad":"SoS-Grad-Fusion Pair",
        "sos_cvpr_fusion_grad":"SoS-Grad-Fusion CVPR14",
        "sos_chen_fusion_grad":"SoS-Grad-Fusion OldCVPR",
        "sos_cardinality_fusion_grad":"SoS-Grad-Fusion Card"
    }

for m in methods:
    if m not in methodToName:
        methodToName[m] = m

for m in methods:
    print "{:30} & {:11.2f} & {:9.3f} & {:9.2f}\%\\\\".format(methodToName[m], avgEnergy[m], avgTime[m], best[m])

# Write graph output
for f in files:
    for m in methods:
        outputFile = open(resultsDir + "/" + f + "_" + m + "-graph.txt",
                "w")
        for i in xrange(0, len(energy[f][m])):
            line = "{:4}\t{:11.4f}\t{:9.4f}\n".format(i, energy[f][m][i], times[f][m][i])
            outputFile.write(line)
        outputFile.close()

gnuplotString = """\
set terminal pdfcairo
set out "{0}-energy.pdf"
set xlabel "Time (seconds)"
set ylabel "Energy"

plot \\
"""
#methodString
    #"results-ub/REPLACE_FILE-reduction-1.stats" using 3:5 title "FGBZ-Fusion" with linespoints,
    #"results-ub/REPLACE_FILE-hocr-1.stats" using 3:5 title "HOCR-Fusion" with linespoints,
    #"results-ub/REPLACE_FILE-spd-alpha-0.stats" using 3:5 title "SoSPD-Fusion" with linespoints,
    #"results-ub/REPLACE_FILE-spd-alpha-height-0.stats" using 3:5 title "SoSPD-Best-Fusion" with linespoints
for f in files:
    gnuplotOut = open(resultsDir + "/" + f + ".gp", "w")
    gnuplotOut.write(gnuplotString.format(resultsDir + "/" + f))
    for m in methods[:-1]:
        line = """\t\t"{0}" using 3:2 title "{1}" with linespoints,\\\n""".format(resultsDir + "/" + f + "_" + m + "-graph.txt", methodToName[m])
        gnuplotOut.write(line)
    line = """\t\t"{0}" using 3:2 title "{1}" with linespoints,\n""".format(resultsDir + "/" + f + "_" + methods[-1] + "-graph.txt", methodToName[methods[-1]])
    gnuplotOut.write(line)
    gnuplotOut.close()


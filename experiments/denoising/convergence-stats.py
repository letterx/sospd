#!/usr/bin/env python

import sys

total_energy = 0
total_time = 0
total_iters = 0
num_files = 0
for fname in sys.argv[1:]:
    num_files += 1
    f = open(fname, "r")
    final_energy = 0
    final_time = 0
    total_lines = 0
    for line in f:
        total_lines += 1
        fields = line.split()
        final_energy = float(fields[4])
        final_time = float(fields[2])
    total_energy += final_energy
    total_time += final_time
    total_iters += total_lines
total_energy /= num_files
total_time /= num_files
total_iters /= float(num_files)

print "Average energy: ", total_energy
print "Average time: ", total_time
print "Average iters: ", total_iters

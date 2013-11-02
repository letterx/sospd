set terminal pdfcairo
set out "REPLACE_FILE-energy.pdf"
set xlabel "Time (seconds)"
set ylabel "Energy"
set xrange [0:60]

plot \
    "./results-with-ub/REPLACE_FILE-reduction-grad-1.stats" using 3:5 title "FGBZ-Gradient" with linespoints, \
    "./results-with-ub/REPLACE_FILE-hocr-grad-1.stats" using 3:5 title "HOCR-Gradient" with linespoints, \
    "./results-with-ub/REPLACE_FILE-spd-alpha-0.stats" using 3:5 title "SoSPD-Alpha" with linespoints, \
    "./results-with-ub/REPLACE_FILE-spd-grad-0.stats" using 3:5 title "SoSPD-Gradient" with linespoints


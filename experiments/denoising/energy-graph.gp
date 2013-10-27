set terminal pdfcairo
set out "REPLACE_FILE-energy.pdf"
set xlabel "Time (seconds)"
set ylabel "Energy"
set xrange [0:60]

plot "REPLACE_FILE-reduction.stats" using 3:5 title "FGBZ" with linespoints, \
    "REPLACE_FILE-reduction-grad.stats" using 3:5 title "FGBZ Grad" with linespoints, \
    "REPLACE_FILE-spd-alpha.stats" using 3:5 title "SPD3 Alpha" with linespoints, \
    "REPLACE_FILE-spd-alpha-height.stats" using 3:5 title "SPD3 Best Alpha" with linespoints, \
    "REPLACE_FILE-spd-blur-random.stats" using 3:5 title "SPD3 Blur Random" with linespoints, \
    "REPLACE_FILE-spd-grad.stats" using 3:5 title "SPD3 Grad" with linespoints

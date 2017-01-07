#!/usr/bin/gnuplot

set title "Timing gcd(a=10^n-3, b=10^n)"
set ylabel "Seconds"
set xlabel "Number of digits"
unset log
set yrange [0:.5]
set terminal png size 512,384 enhanced font "Helvetica,10"
set output 'OUT0.gcd.png'
plot [1:2**20] "OUT.gcd.theirs" with linespoints title "a.gcd(b)", "OUT.gcd.mine" with linespoints title "myGcd(a,b)"
set output 'OUT1.gcd.png'
unset yrange; replot
set output 'OUT2.gcd.png'
set log; replot

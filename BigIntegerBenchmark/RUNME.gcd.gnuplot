#!/usr/bin/gnuplot

set ylabel "Seconds"
set xlabel "Number of digits"
unset log
set yrange [0:.5]
#set terminal png size 640,480 enhanced
set terminal png size 512,384 enhanced
set output 'OUT0.png'
plot [1:2**20] "OUT.gcd.theirs" with linespoints title "a.gcd(b)", "OUT.gcd.mine" with linespoints title "myGcd(a,b)"
set output 'OUT1.png'
unset yrange; replot
set output 'OUT2.png'
set log; replot

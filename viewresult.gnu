set term wxt persist enhanced size 3000,1000
set palette rgb 33,13,10
set multiplot layout 1,3
set size square
unset xtics
unset ytics
#set xrange [-0.5:999.5]
#set yrange [-0.5:999.5]
plot "A.dat" matrix w image
plot "B.dat" matrix w image
plot "C.dat" matrix w image

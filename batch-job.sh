#!/bin/sh
parallel --bar --retry-failed --joblog logs/log -j1 \
'python run.py -n={1} -k={2} -d={3} -b={4} -r={5} -c={6} --ema -s={7} --device=0' ::: \
nltcs kdd audio netflix jester ::: \
50 70 100 ::: \
10 30 50 ::: \
64 128 ::: \
0.005 0.001 ::: \
0.1 0.2 0.5 ::: \
17 23
###ad accidents audio bbc netflix book 20ng cr52 webkb dna jester kdd kosarek msnbc msweb nltcs plants pumsb_star tmovie tretail ::: \
#retail webkb dna kosarek  ::: \
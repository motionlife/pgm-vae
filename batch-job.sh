#!/bin/sh
#parallel --bar --retry-failed --joblog logs/log_nltcs_kdd -j8 \
#'python run.py --name={1} -k={2} --dim={3} --batch={4} --epoch={5} --rate={6} --cost={7} --seed={8} --device=-1 --ema' ::: \
#nltcs kdd ::: \
#100 50 ::: \
#10 15 30 ::: \
#128 ::: \
#400 ::: \
#0.0005 0.001 ::: \
#0.1 0.25 ::: \
#1 2 3
###ad accidents audio bbc netflix book 20ng cr52 webkb dna jester kdd kosarek msnbc msweb nltcs plants pumsb_star tmovie tretail ::: \
#webkb(839) kosarek(190) retail(135) audio(100) netflix(100) jester(100) kdd(64) nltcs(16)
#parallel --bar --retry-failed --joblog logs/log_ranj -j2 \
#'python run.py --name={1} -k={2} --dim={3} --batch={4} --epoch={5} --rate={6} --cost={7} --seed={8} --device=0 --ema' ::: \
#retail audio netflix jester ::: \
#100 50 ::: \
#20 50 70 ::: \
#128 ::: \
#400 ::: \
#0.0005 0.001 ::: \
#0.1 0.25 0.5 ::: \
#1
parallel --bar --retry-failed --joblog logs/log_webko -j1 \
'python run.py --name={1} -k={2} --dim={3} --batch={4} --epoch={5} --rate={6} --cost={7} --seed={8} --device=1 --ema' ::: \
webkb kosarek ::: \
100 50 ::: \
50 70 100 ::: \
128 ::: \
400 ::: \
0.0005 0.001 ::: \
0.1 0.25 0.5 ::: \
7
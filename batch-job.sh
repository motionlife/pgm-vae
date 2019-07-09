#!/bin/sh

#parallel --bar --retry-failed --joblog logs/log_nltcs_kdd -j4 \
#'python run.py --name={1} -k={2} --dim={3} --batch={4} --epoch={5} --rate={6} --cost={7} --seed={8} --device=-1 --ema' ::: \
#nltcs kdd ::: \
#100 ::: \
#30 50 70 ::: \
#128 ::: \
#250 ::: \
#0.0005 0.001 ::: \
#0.25 0.5 1 ::: \
#11

#-------------------------------------------------------------------------------------
###ad accidents audio bbc netflix book 20ng cr52 webkb dna jester kdd kosarek msnbc msweb nltcs plants pumsb_star tmovie tretail ::: \
#webkb(839) kosarek(190) retail(135) audio(100) netflix(100) jester(100) kdd(64) nltcs(16)
#-------------------------------------------------------------------------------------

#parallel --bar --retry-failed --joblog logs/log_ranj -j2 \
#'python run.py --name={1} -k={2} --dim={3} --batch={4} --epoch={5} --rate={6} --cost={7} --seed={8} --device=0 --ema' ::: \
#kosarek retail audio netflix jester ::: \
#100 150 ::: \
#50 100 ::: \
#128 ::: \
#250 ::: \
#0.0005 0.001 ::: \
#1 0.25 0.5 ::: \
#12

#-------------------------------------------------------------------------------------

#parallel --bar --retry-failed --joblog logs/log_webko -j1 \
#'python run.py --name={1} -k={2} --dim={3} --batch={4} --epoch={5} --rate={6} --cost={7} --seed={8} --device=1 --ema' ::: \
#webkb ::: \
#200 ::: \
#50 100 ::: \
#128 ::: \
#250 ::: \
#0.0005 0.001 ::: \
#0.25 0.5 1 ::: \
#17

parallel --bar --retry-failed --joblog logs/log_stu -j2 \
'python run.py --name={1} -k={2} --dim={3} --batch={4} --epoch={5} --rate={6} --cost={7} --seed={8} --device=-1 --ema' ::: \
50-17-8 students_03_02-0000 ::: \
200 ::: \
70 100 ::: \
128 ::: \
250 ::: \
0.0005 0.001 ::: \
0.25 0.5 1 ::: \
7
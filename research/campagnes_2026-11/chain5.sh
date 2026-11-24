#!/bin/bash
# Mid scene WITH random fast movements (jrk), no hard zone, 5 freqs x 2 reps: ref / clip10 / cfix12 (paired with wkM1/wkM2).
S=/tmp/claude-0/-root/28c21aa4-61e4-4f8c-b7ee-3d09b2142a5f/scratchpad; R=/root/worspace/cosmik_mppi_sim
cd $R; export ROS_LOCALHOST_ONLY=1 DEMO_LOCALHOST_ONLY=1 DEMO_DOMAIN_ID=93
M=x100+alt+cattab+b75+mid+jrk
echo "start wkM3 $(TZ=Europe/Paris date +%H:%M)"
python3 -u $S/wk.py repeat-wkM3 20261172 2 0.1,0.2,0.3,0.4,0.5 midj=$M+pmax15 midj_clip10=$M+pmax15+clip10 midj_cfix12=$M+pmax15+cfix12 >> $S/wkM3.log 2>&1
echo "chain5 done $(TZ=Europe/Paris date +%H:%M)"

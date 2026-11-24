#!/bin/bash
# Mid scene, more reps: 6 configs (with / without jrk) x 5 freqs, reps 3-6 (reps 1-2 linked from wkM1/wkM2/wkM3, skipped).
S=/tmp/claude-0/-root/28c21aa4-61e4-4f8c-b7ee-3d09b2142a5f/scratchpad; R=/root/worspace/cosmik_mppi_sim
cd $R; export ROS_LOCALHOST_ONLY=1 DEMO_LOCALHOST_ONLY=1 DEMO_DOMAIN_ID=93
M=x100+alt+cattab+b75+mid
echo "start wkM4 $(TZ=Europe/Paris date +%H:%M)"
python3 -u $S/wk.py repeat-wkM4 20261173 6 0.1,0.2,0.3,0.4,0.5 \
  mid=$M+pmax15 midj=$M+jrk+pmax15 mid_clip10=$M+pmax15+clip10 midj_clip10=$M+jrk+pmax15+clip10 \
  mid_cfix12=$M+pmax15+cfix12 midj_cfix12=$M+jrk+pmax15+cfix12 >> $S/wkM4.log 2>&1
echo "chain6 done $(TZ=Europe/Paris date +%H:%M)"

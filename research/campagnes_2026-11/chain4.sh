#!/bin/bash
# No hard zone, 5 frequencies x 2 reps, existing YAML params only.
# wkM2: mid scene with clip10 / pmax20 / cfix12 (each 3/10 at x -1.00 vs ref 5/10). wkB6: reactive arm with clip10 / pmax20.
S=/tmp/claude-0/-root/28c21aa4-61e4-4f8c-b7ee-3d09b2142a5f/scratchpad; R=/root/worspace/cosmik_mppi_sim
cd $R; export ROS_LOCALHOST_ONLY=1 DEMO_LOCALHOST_ONLY=1 DEMO_DOMAIN_ID=93
M=x100+alt+cattab+b75+mid; K=x100+y70+cattab+b75+blk+blkR+react+jrk
echo "start wkM2 $(TZ=Europe/Paris date +%H:%M)"
python3 -u $S/wk.py repeat-wkM2 20261170 2 0.1,0.2,0.3,0.4,0.5 mid_clip10=$M+pmax15+clip10 mid_pmax20=$M+pmax20 mid_cfix12=$M+pmax15+cfix12 >> $S/wkM2.log 2>&1
echo "start wkB6 $(TZ=Europe/Paris date +%H:%M)"
python3 -u $S/wk.py repeat-wkB6 20261171 2 0.1,0.2,0.3,0.4,0.5 react_clip10=$K+pmax15+clip10 react_pmax20=$K+pmax20 >> $S/wkB6.log 2>&1
echo "chain4 done $(TZ=Europe/Paris date +%H:%M)"

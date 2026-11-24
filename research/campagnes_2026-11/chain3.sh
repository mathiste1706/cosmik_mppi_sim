#!/bin/bash
# No hard zone, 5 frequencies x 2 reps: right arm on the path midpoint (wkM1), then right arm reaching onto B
# when the robot approaches, reach timing tied to the frequency (wkB5). x -1.00.
S=/tmp/claude-0/-root/28c21aa4-61e4-4f8c-b7ee-3d09b2142a5f/scratchpad; R=/root/worspace/cosmik_mppi_sim
cd $R; export ROS_LOCALHOST_ONLY=1 DEMO_LOCALHOST_ONLY=1 DEMO_DOMAIN_ID=93
echo "start wkM1 $(TZ=Europe/Paris date +%H:%M)"
python3 -u $S/wk.py repeat-wkM1 20261164 2 0.1,0.2,0.3,0.4,0.5 mid=x100+alt+cattab+b75+pmax15+mid > $S/wkM1.log 2>&1
echo "start wkB5 $(TZ=Europe/Paris date +%H:%M)"
python3 -u $S/wk.py repeat-wkB5 20261168 2 0.1,0.2,0.3,0.4,0.5 react=x100+y70+cattab+b75+pmax15+blk+blkR+react+jrk > $S/wkB5.log 2>&1
echo "chain3 done $(TZ=Europe/Paris date +%H:%M)"

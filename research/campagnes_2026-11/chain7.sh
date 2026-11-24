#!/bin/bash
# Reactive arm blocking B, WITH jrk, more reps: ref / clip10 x 5 freqs, reps 3-6 (reps 1-2 linked from wkB5/wkB6, skipped).
# Starts once chain6 (wkM4) has finished.
S=/tmp/claude-0/-root/28c21aa4-61e4-4f8c-b7ee-3d09b2142a5f/scratchpad; R=/root/worspace/cosmik_mppi_sim
while pgrep -f "next/chain6.sh" >/dev/null; do sleep 60; done
cd $R; export ROS_LOCALHOST_ONLY=1 DEMO_LOCALHOST_ONLY=1 DEMO_DOMAIN_ID=93
K=x100+y70+cattab+b75+blk+blkR+react+jrk
echo "start wkB7 $(TZ=Europe/Paris date +%H:%M)"
python3 -u $S/wk.py repeat-wkB7 20261174 6 0.1,0.2,0.3,0.4,0.5 \
  react=$K+pmax15 react_clip10=$K+pmax15+clip10 >> $S/wkB7.log 2>&1
# pmax20 removed on request (2026-11-23): react_pmax20=$K+pmax20
echo "chain7 done $(TZ=Europe/Paris date +%H:%M)"

#!/usr/bin/env python3
"""Same conditions for every config: shuffled blocks, RViz at 10 fps, random start kept. Resumable; restores the config on exit."""
import json, os, random, re, shutil, signal, statistics, subprocess, sys, time
from pathlib import Path
S = Path('/tmp/claude-0/-root/28c21aa4-61e4-4f8c-b7ee-3d09b2142a5f/scratchpad')
ROOT = Path('/root/worspace/cosmik_mppi_sim'); Y = ROOT/'storm/content/configs/mpc/franka_reacher_cat_real.yml'
BACKUP = S/'backup/franka_reacher_cat_real_hp.yml'; OUT = ROOT/'.runtime'/sys.argv[1]
U = ROOT/'ros2_ws/src/storm_pick_place_demo/scripts/urdf_human'; U_BACKUP = S/'backup/urdf_human'
# Task manager (blocked goal, flag stop); restored from its backup on exit like the two files above.
TM = ROOT/'ros2_ws/src/storm_pick_place_demo/storm_pick_place_demo/pick_place_task_manager_node.py'; TM_BACKUP = S/'backup/pick_place_task_manager_node.py'
U = ROOT/'ros2_ws/src/storm_pick_place_demo/scripts/urdf_human'; U_BACKUP = S/'backup/urdf_human'
ELBOW_B = (-1.8, -1.0, -2.0, -1.7, -0.5, 2.4, 0.3)
# Weekend campaigns (2026-11-13). Flags:
#  fix  : table_collision ignores panda_link0/link1 and weight 1000 -> 100 (CaT baseline fix)
#  h75  : horizon 75 (fine steps, ~3 s) + fixed 3.2 s forecast
#  b75  : beta 25 -> 75          q2m: state_bound margin on q2 0.15     high: shoulder 1.5 rad
#  ori3 : goal orientation weight 1 -> 3      vel: state_filter velocity coefficient 0 -> 0.05
#  nocat: all cat_* disabled, primitive_collision threshold 0.10 (10 cm margin kept), self-collision cost 100
#  xNNN / yNN: human base x = -NNN/100, y = NN/100 (default x -0.85, y 0.20)   lag02/lag05: forecast content 0.2/0.5 s stale   nopred: forecast = current pose
#  amp08/amp12: forecast amplitude x0.8/x1.2   m15: CaT human margin 0.10 -> 0.15   pmax10: CaT collision p_max 0.05 -> 0.10
#  n2000: 2000 particles   cov05: init_cov 0.3 -> 0.5
#  blk: left hand reaches over B periodically (urdf_human block_*, left arm added to the collision geometry)
#  jrk: random fast bumps on the right shoulder (urdf_human jerk_*, seeded per rep/frequency, unknown to the forecast)
#  mid: right shoulder centred at 0.9 rad (hand on the A-B path midpoint for x -1.00)   blkB: left hand exactly over B
#  react: the blocking reach starts when the robot end effector approaches B (no command to the robot)
#  blkR: right arm reaches onto B (forward-backward) instead of the sine, with y70 (human shifted to y 0.70)
#  b2s: (with react) the reach is a 2 s out-and-back (1 s out, no dwell over B, 1 s back), whatever the frequency label
#  trgNN: (with react) the reach starts when the robot is at most NN cm from B (block_trigger_far = NN/100)
#  stop: the task manager holds the robot while its goal is within 0.12 m of a human cylinder (goal_block_distance)
#  w100 / w300: primitive_collision weight 1000 -> 100 / 300 (with nocat)
CFG = {kv.split('=')[0]: tuple(f for f in kv.split('=')[1].split('+') if f) for kv in sys.argv[5:]}
FREQS = tuple(sys.argv[4].split(',')); REPEATS = int(sys.argv[3])
ENV = dict(os.environ, ROS_LOCALHOST_ONLY='1', DEMO_LOCALHOST_ONLY='1', DEMO_DOMAIN_ID='93', ROS_DOMAIN_ID='93', DISPLAY=':99')
ENV.pop('XAUTHORITY', None)

def alive(name):
    out = subprocess.run(['ps', '-C', name, '-o', 'stat='], capture_output=True, text=True).stdout
    return any(s.strip() and not s.strip().startswith('Z') for s in out.splitlines())

def ensure_display():
    quiet = dict(env=ENV, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not alive('Xvfb'):
        for p in ('/tmp/.X99-lock', '/tmp/.X11-unix/X99'):
            if os.path.exists(p): os.remove(p)
        subprocess.Popen(['setsid', 'nohup', 'Xvfb', ':99', '-screen', '0', '1920x1080x24'], **quiet)
        time.sleep(2); print('started Xvfb', flush=True)
    if not alive('rviz2'):
        subprocess.Popen(['setsid', 'nohup', 'bash', '-c', f'source {ROOT}/scripts/env.sh >/dev/null 2>&1; exec rviz2 -d {S}/record10.rviz --ros-args -p use_sim_time:=true'], **quiet)
        time.sleep(8); print('started rviz2', flush=True)

def sub(s, old, new):
    assert s.count(old) == 1, old; return s.replace(old, new)

def config_text(c):
    f = CFG[c]; s = BACKUP.read_text()
    if 'fix' in f:
        s = sub(s, '  table_collision:\n    weight: 1000.0\n', '  table_collision:\n    weight: 100.0  # was 1000.0\n    ignore_links: [panda_link0, panda_link1]\n')
    if 'ign300' in f:   # base ignored, intermediate table weight
        s = sub(s, '  table_collision:\n    weight: 1000.0\n', '  table_collision:\n    weight: 300.0  # was 1000.0\n    ignore_links: [panda_link0, panda_link1]\n')
    if 'hp20' in f: s = sub(s, '  human_proximity:\n    weight: 0.0\n', '  human_proximity:\n    weight: 20.0\n')
    if 'pmax30' in f: s = sub(s, '    p_max: 0.05  #', '    p_max: 0.30  #')
    if 'cattab' in f:   # table as a CaT constraint (new mppi.cat_table_collision), base links ignored, weight kept 1000
        s = sub(s, '  table_collision:\n    weight: 1000.0\n', '  table_collision:\n    weight: 1000.0\n    ignore_links: [panda_link0, panda_link1]\n')
        old_self = '  cat_self_collision:\n    enabled: true\n    p_max: 0.15\n    tau_c: 0.95\n'
        s = sub(s, old_self, old_self + '  cat_table_collision:\n    enabled: true\n    p_max: 0.15\n    tau_c: 0.95\n')
    if 'post' in f:   # posture hint per target, elbow clear of the whole arm sweep (IK, 10.7 cm clearance); q1 pulled too
        s = sub(s, '  retract_state:\n', '  retract_state_by_goal_y:  # A side / B side\n    negative: [-2.114, 1.269, 0.029, -1.011, -2.161, 0.8, 1.243]\n    positive: [2.134, 1.103, -0.011, -0.521, 0.775, 1.133, -0.622]\n  retract_state:\n')
        s = sub(s, '    vec_weight:\n    - 0.1\n    - 1.0\n    - 1.0\n    - 1.0\n    - 1.0\n    - 0.1\n    - 0.0\n',
                   '    vec_weight:  # q1 was 0.1\n    - 1.0\n    - 1.0\n    - 1.0\n    - 1.0\n    - 1.0\n    - 0.1\n    - 0.0\n')
    if 'pmax50' in f: s = sub(s, '    p_max: 0.05  #', '    p_max: 0.50  #')
    if 'pmax100' in f: s = sub(s, '    p_max: 0.05  #', '    p_max: 1.0  #')
    if 'h75' in f: s = sub(s, '  horizon: 50  #', '  horizon: 75  #')
    if 'b75' in f: s = sub(s, '  beta: 25.0  #', '  beta: 75.0  #')
    if 'q2m' in f:
        old = '    bound_thresh: 0.05  # was 0.25'; s = sub(s, old, '    bound_thresh_override: {1: 0.15}\n' + old)
    if 'ori3' in f: s = sub(s, '    weight:  # [orientation, position]\n    - 1.0\n', '    weight:  # [orientation, position]\n    - 3.0\n')
    if 'm15' in f: s = sub(s, '    distance_threshold: 0.1\n', '    distance_threshold: 0.15  # was 0.1 (human margin increased, never reduced)\n')
    if 'm12' in f: s = sub(s, '    distance_threshold: 0.1\n', '    distance_threshold: 0.12  # was 0.1 (human margin increased, never reduced)\n')
    if 'pmax10' in f: s = sub(s, '    p_max: 0.05  #', '    p_max: 0.10  #')
    if 'pmax15' in f: s = sub(s, '    p_max: 0.05  #', '    p_max: 0.15  #')
    # Hard zone / fixed normalization of the human constraint (new mppi.py options, off without these flags).
    if 'core3' in f: s = sub(s, '    violation_scale: 0.25\n  cat_state_bound:\n', '    violation_scale: 0.25\n    core_distance: 0.03  # hard zone: delta -> 1 at contact\n  cat_state_bound:\n')
    if 'cfix' in f: s = sub(s, '    violation_scale: 0.25\n', '    violation_scale: 0.25\n    cmax_fixed: 0.4  # one sphere at contact = 0.1 m / 0.25\n')
    # Parameter-only tuning of the 10 cm human constraint without the hard zone (existing YAML keys).
    for flag, v in (('cfix08', '0.8'), ('cfix12', '1.2')):
        if flag in f: s = sub(s, '    violation_scale: 0.25\n', f'    violation_scale: 0.25\n    cmax_fixed: {v}  # fixed normalization\n')
    if 'clip10' in f: s = sub(s, '    violation_clip_max: 0.2\n    violation_scale: 0.25\n', '    violation_clip_max: 0.1  # was 0.2\n    violation_scale: 0.25\n')
    if 'pmax20' in f: s = sub(s, '    p_max: 0.05  #', '    p_max: 0.20  #')
    if 'n2000' in f: s = sub(s, '  num_particles: 1000  #', '  num_particles: 2000  #')
    if 'cov05' in f: s = sub(s, '  init_cov: 0.3  #', '  init_cov: 0.5  #')
    if 'vel' in f: s = sub(s, 'state_filter_coeff:\n  position: 0.1\n  velocity: 0.0\n', 'state_filter_coeff:\n  position: 0.1\n  velocity: 0.05\n')
    if 'nocat' in f:
        assert s.count('    enabled: true') == 4; s = s.replace('    enabled: true', '    enabled: false')
        w = '100.0' if 'w100' in f else '300.0' if 'w300' in f else '1000.0'
        s = sub(s, '    weight: 1000.0\n    distance_threshold: 0.05\n', f'    weight: {w}\n    distance_threshold: 0.10\n')
        s = sub(s, '  robot_self_collision:\n    weight: 0.0\n', '  robot_self_collision:\n    weight: 100.0\n')
    return s

def human_text(c, f=0.0, k=0):
    u = U_BACKUP.read_text()
    if 'h75' in CFG[c]: u = sub(u, "self.declare_parameter('forecast_horizon', 1.0)", "self.declare_parameter('forecast_horizon', 3.2)")
    if 'high' in CFG[c]: u = sub(u, "    'right_shoulder_Z': 1.2,", "    'right_shoulder_Z': 1.5,")
    if 'mid' in CFG[c]: u = sub(u, "    'right_shoulder_Z': 1.2,", "    'right_shoulder_Z': 0.9,")  # hand centred on the A-B path midpoint (x -1.00)
    # Human base position: flag xNNN -> x = -NNN/100 (e.g. x105 -> -1.05), yNN -> y = NN/100 (default -0.85, 0.20).
    hx = next((-int(f[1:]) / 100 for f in CFG[c] if f[0] == 'x' and f[1:].isdigit()), -0.85)
    hy = next((int(f[1:]) / 100 for f in CFG[c] if f[0] == 'y' and f[1:].isdigit()), 0.20)
    if (hx, hy) != (-0.85, 0.20):
        u = sub(u, "self.declare_parameter('xyz', [-0.85, 0.20, 0.15])", f"self.declare_parameter('xyz', [{hx}, {hy:.2f}, 0.15])")
    # Strings below follow urdf_human with the random amplitude alternation (backup/urdf_human.alt;
    # previous baseline: backup/urdf_human.pre_alt).
    for flag, lag in (('lag02', 0.2), ('lag05', 0.5)):   # stale prediction: forecast content lags the truth (stamps unchanged)
        if flag in CFG[c]:
            u = sub(u, "zip(self.segments, self.segment_poses(self.joint_positions(t_ns * 1e-9, amp_time, jerk_time)))):",
                       f"zip(self.segments, self.segment_poses(self.joint_positions(t_ns * 1e-9 - ({lag} if ns_suffix else 0.0), amp_time, jerk_time)))):")
    for flag, amp in (('amp08', 0.8), ('amp12', 1.2)):   # wrong predicted amplitude (forecast only)
        if flag in CFG[c]:
            u = sub(u, "            q[self.sine_joint] += amp * math.sin(2.0 * math.pi * t / self.sine_period)\n",
                       f"            q[self.sine_joint] += amp * ({amp} if amp_time is not None else 1.0) * math.sin(2.0 * math.pi * t / self.sine_period)\n")
            u = sub(u, "amp_time=None if self.forecast_knows_switch else t,\n", "amp_time=t,\n")
    # Random alternation between a small and a large amplitude, one per arm period (flag alt).
    # Seed from (rep, frequency): every config sees the same amplitude sequence in a given rep/frequency.
    if 'alt' in CFG[c] or 'altk' in CFG[c]:
        u = sub(u, "self.declare_parameter('sine_amplitudes', [0.0])", "self.declare_parameter('sine_amplitudes', [0.15, 0.45])")
        u = sub(u, "self.declare_parameter('sine_switch_seed', 0)", f"self.declare_parameter('sine_switch_seed', {int(k) * 100 + round(float(f) * 10)})")
    if 'altk' in CFG[c]:   # forecast follows the true future, switches included
        u = sub(u, "self.declare_parameter('forecast_knows_switch', False)", "self.declare_parameter('forecast_knows_switch', True)")
    if 'blk' in CFG[c]:
        u = sub(u, "self.declare_parameter('block_enabled', False)", "self.declare_parameter('block_enabled', True)")
    if 'blkR' in CFG[c]:   # RIGHT arm does the forward-backward reach onto B instead of the sine (use with y70, x100)
        u = sub(u, "self.declare_parameter('motion', 'sine')", "self.declare_parameter('motion', 'static')")
        u = sub(u, "self.declare_parameter('block_joints', ['left_shoulder_Z', 'left_elbow_Z'])", "self.declare_parameter('block_joints', ['right_shoulder_Z', 'right_elbow_Z'])")
        u = sub(u, "self.declare_parameter('block_retracted', [0.0, 1.8])", "self.declare_parameter('block_retracted', [0.0, 1.725])")
        u = sub(u, "self.declare_parameter('block_extended', [1.2, 0.0])", "self.declare_parameter('block_extended', [0.80, 0.65])")
        u = sub(u, "self.declare_parameter('block_segments', ['upperarm_L', 'lowerarm_L'])", "self.declare_parameter('block_segments', ['upperarm_R'])")
    if 'react' in CFG[c]:   # the reach starts when the robot approaches B (urdf_human block_mode reactive)
        u = sub(u, "self.declare_parameter('block_mode', 'periodic')", "self.declare_parameter('block_mode', 'reactive')")
        # Reach / retract follow the campaign frequency (half-sine of 1/(2f) s each, 4 s over B in between);
        # slower reaches start farther from B so the hand lands on B about when the robot gets there.
        m = 1.0 / (2.0 * float(f)); far = min(0.95, 0.15 + 0.25 * m)
        if 'b2s' in CFG[c]:   # 2 s out-and-back, no dwell over B (requested 2026-11-24)
            m = 1.0; far = min(0.95, 0.15 + 0.25 * m)
            u = sub(u, "self.declare_parameter('block_dwell', 4.0)", "self.declare_parameter('block_dwell', 0.0)")
        for fl in CFG[c]:
            if fl.startswith('trg') and fl[3:].isdigit():
                far = int(fl[3:]) / 100.0
        u = sub(u, "self.declare_parameter('block_move', 1.0)", f"self.declare_parameter('block_move', {m:.3f})")
        u = sub(u, "self.declare_parameter('block_period', 12.0)", f"self.declare_parameter('block_period', {2.0 * m + 5.0:.3f})")
        u = sub(u, "self.declare_parameter('block_trigger_far', 0.45)", f"self.declare_parameter('block_trigger_far', {far:.3f})")
    if 'blkB' in CFG[c]:   # hand exactly over B (x -1.00): lateral left_shoulder_X added to the reach
        u = sub(u, "self.declare_parameter('block_joints', ['left_shoulder_Z', 'left_elbow_Z'])", "self.declare_parameter('block_joints', ['left_shoulder_X', 'left_shoulder_Z', 'left_elbow_Z'])")
        u = sub(u, "self.declare_parameter('block_retracted', [0.0, 1.8])", "self.declare_parameter('block_retracted', [0.40, 0.0, 1.65])")
        u = sub(u, "self.declare_parameter('block_extended', [1.2, 0.0])", "self.declare_parameter('block_extended', [0.18, 0.85, 0.55])")
    if 'jrk' in CFG[c]:
        u = sub(u, "self.declare_parameter('jerk_enabled', False)", "self.declare_parameter('jerk_enabled', True)")
        u = sub(u, "self.declare_parameter('jerk_seed', 0)", f"self.declare_parameter('jerk_seed', {int(k) * 100 + round(float(f) * 10)})")
    if 'nopred' in CFG[c]:   # no prediction: forecast = current pose only (held over the horizon)
        u = sub(u, "self.declare_parameter('forecast_horizon', 1.0)", "self.declare_parameter('forecast_horizon', 0.0)")
        u = sub(u, "self.declare_parameter('forecast_horizon_max', 2.0)", "self.declare_parameter('forecast_horizon_max', 0.0)")
    return u

def tm_text(c):
    s = TM_BACKUP.read_text()
    if 'stop' in CFG[c]:
        s = sub(s, 'self.declare_parameter("goal_block_distance", 0.0)', 'self.declare_parameter("goal_block_distance", 0.12)')
    return s

def rtf(d):
    v = [float(x) for x in re.findall(r'real-time factor ([\d.]+)', (d/'launch.log').read_text(errors='ignore'))]
    return statistics.median(v) if v else None

def run(c, f, k):
    d = OUT/c/f'f{f}'/f'rep{k}'; rd = d/f'freq_{f}Hz'
    if (rd/'result.json').exists() and (rd/'video.mp4').exists(): return
    d.mkdir(parents=True, exist_ok=True)
    for attempt in (1, 2):
        if shutil.disk_usage(ROOT).free < 5e9: sys.exit('less than 5 GB free: stopping')
        ensure_display(); Y.write_text(config_text(c)); U.write_text(human_text(c, f, k)); TM.write_text(tm_text(c))
        subprocess.run(['bash', '-c', f'cd {ROOT}; export HUMAN_BENCHMARK_TIMINGS={d}/timings.jsonl; source scripts/env.sh >/dev/null 2>&1; exec python3 scripts/arm_frequency_sweep.py --output {d} --frequencies {f} --repeats 1 '
                        '--mppi-multimodal false --human-prediction true --min-timeout 110 --record-display :99'],
                       env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if (rd/'result.json').exists(): break
        print(f'{c} {f} rep{k}: no result, attempt {attempt}', flush=True)
    else:
        return
    r = json.loads((rd/'result.json').read_text())
    print(time.strftime('%H:%M'), c, f, f'rep{k}', 'B', r['spatial_arrivals'].get('B', 0), 'samples', r['penetration_samples'], 'to', r['timeout_events'], 'rtf', rtf(rd), flush=True)

def jobs():
    rng = random.Random(int(sys.argv[2])); out = []
    for k in range(1, REPEATS+1):
        block = [(c, f, k) for c in CFG for f in FREQS]; rng.shuffle(block); out += block
    return out

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: sys.exit('stopped'))
    try:
        for job in jobs(): run(*job)
    finally:
        shutil.copy2(BACKUP, Y); shutil.copy2(U_BACKUP, U); shutil.copy2(TM_BACKUP, TM); print('config restored', flush=True)

"""wkM1 (mid) + wkB5 (react, freq-linked reach): per-run metrics; for react, reconstruct reaches from the EE trace."""
import json, glob, numpy as np, re
S = '/tmp/claude-0/-root/28c21aa4-61e4-4f8c-b7ee-3d09b2142a5f/scratchpad'; R = '/root/worspace/cosmik_mppi_sim/.runtime'
CC = [('repeat-wkM1','mid'),('repeat-wkB5','react')] + [('repeat-wkM2',c) for c in ('mid_clip10','mid_pmax20','mid_cfix12')] + [('repeat-wkB6',c) for c in ('react_clip10','react_pmax20')]
B = np.array([-0.5, 0.5, 0.3])
runs = []
for camp, cfg in CC:
    for p in sorted(glob.glob(f'{R}/{camp}/{cfg}/f*/rep*/freq_*/result.json')):
        d = p.rsplit('/', 1)[0]; r = json.load(open(p))
        f = float(p.split('/')[-4][1:]); rep = int(p.split('/')[-3][3:])
        bad = r['messages'].get('joint_states', 0) / max(r['simulated_seconds_after_first_goal'], 1) > 32
        tr = np.load(d + '/trace.npy'); t = tr[:, 0]; ee = tr[:, 1:4]; clr = tr[:, -1]
        x = dict(cfg=cfg, f=f, rep=rep, dir=d, bad=bad, B=r['spatial_arrivals'].get('B', 0), pen=r['penetration_samples'],
                 dmin=round(100 * r['min_human_clearance_m'], 1), to=r['timeout_events'],
                 seg={k.split('/')[-1]: round(100 * v, 1) for k, v in r['min_clearance_by_human_segment_m'].items()})
        ic = int(np.argmin(clr)); x['tmin'] = round(float(t[ic]), 1); x['ee_at_min'] = [round(float(v), 2) for v in ee[ic]]
        x['dB_at_min'] = round(100 * float(np.linalg.norm(ee[ic] - B)), 1)
        if cfg.startswith('react'):
            m = 1 / (2 * f); far = min(0.95, 0.15 + 0.25 * m); dwell, refr = 4.0, 3.0
            dB = np.linalg.norm(ee - B, axis=1); t0 = None; prev = None; reaches = []
            for ti, di in zip(t, dB):
                if t0 is not None and ti - t0 < 2 * m + dwell + refr: prev = di; continue
                if prev is not None and di < prev - 1e-4 and 0.20 <= di <= far: t0 = ti; reaches.append(ti)
                prev = di
            nlog = len(re.findall(r'block reach starts', open(d + '/launch.log').read()))
            ev = []
            for t0 in reaches:
                on = (t >= t0 + m) & (t < t0 + m + dwell)        # hand fully on B
                blk = (t >= t0) & (t < t0 + 2 * m + dwell)      # hand moving or on B
                after = np.where((t >= t0 + 2 * m + dwell) & (dB < 0.03))[0]
                arr = np.where(dB < 0.03)[0]; arr_blk = bool(np.any(blk & (dB < 0.03)))
                if not on.any(): continue
                # speed while hand on B
                ii = np.where(on)[0]; v = np.linalg.norm(np.diff(ee[ii], axis=0), axis=1).sum() / max(t[ii[-1]] - t[ii[0]], 1e-3)
                ev.append(dict(t0=round(float(t0), 1), dB0=round(100 * float(dB[np.searchsorted(t, t0)]), 1),
                               dB_on_min=round(100 * float(dB[on].min()), 1), dB_on_end=round(100 * float(dB[ii[-1]]), 1),
                               v_on=round(100 * v, 1), clr_min=round(100 * float(clr[blk].min()), 1), arrived_while_blocked=arr_blk,
                               delay_after=None if not len(after) else round(float(t[after[0]] - (t0 + 2 * m + dwell)), 2)))
            x['nreach_log'] = nlog; x['nreach_rec'] = len(reaches); x['reaches'] = ev
        runs.append(x)
json.dump(runs, open(f'{S}/m2b6_runs.json', 'w'), indent=1)
for x in runs:
    s = f"{x['cfg']:5s} {x['f']} r{x['rep']} {'BAD ' if x['bad'] else ''}B={x['B']:2d} pen={x['pen']:3d} dmin={x['dmin']:5.1f} to={x['to']} seg={x['seg']} tmin={x['tmin']} ee={x['ee_at_min']} dB={x['dB_at_min']}"
    if x['cfg'].startswith('react'):
        e = x['reaches']; s += f"\n      reaches log {x['nreach_log']} rec {x['nreach_rec']}; dB on-min {[v['dB_on_min'] for v in e]}; arrived-while-blocked {sum(v['arrived_while_blocked'] for v in e)}; delays {[v['delay_after'] for v in e]}; v_on {[v['v_on'] for v in e]}; clr {[v['clr_min'] for v in e]}"
    print(s)

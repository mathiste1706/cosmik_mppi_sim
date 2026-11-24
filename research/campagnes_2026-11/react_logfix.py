"""Reach start times from the urdf_human log (wall clock) mapped to sim time with a linear fit on the matched reconstructed reaches."""
import json, re, numpy as np
S='/tmp/claude-0/-root/28c21aa4-61e4-4f8c-b7ee-3d09b2142a5f/scratchpad'
R=json.load(open(f'{S}/m2b6_runs.json')); Bp=np.array([-0.5,0.5,0.3])
for x in R:
    if not x['cfg'].startswith('react'): continue
    W=[float(w) for w in re.findall(r'\[(\d+\.\d+)\] \[urdf_human\]: \[human-event\] block reach starts', open(x['dir']+'/launch.log').read())]
    rec=[e['t0'] for e in x['reaches']]
    # anchor on the first reach, rtf from the run, then refine with matched pairs
    a, b = rec[0] - 0.98*W[0], 0.98
    for _ in range(3):
        pr=[(w, min(rec, key=lambda r: abs(r-(a+b*w)))) for w in W]; pr=[(w,r) for w,r in pr if abs(r-(a+b*w))<1.5]
        b, a = np.polyfit([p[0] for p in pr],[p[1] for p in pr],1)
    T0=[a+b*w for w in W]
    tr=np.load(x['dir']+'/trace.npy'); t=tr[:,0]; ee=tr[:,1:4]; clr=tr[:,-1]; dB=np.linalg.norm(ee-Bp,axis=1); m=1/(2*x['f'])
    ev=[]
    for t0 in T0:
        end=t0+2*m+4; on=(t>=t0+m)&(t<t0+m+4); blk=(t>=t0)&(t<end); ii=np.where(on)[0]
        if not len(ii): continue
        after=np.where((t>=end)&(dB<0.04))[0]; i0=np.searchsorted(t,t0)
        v=np.linalg.norm(np.diff(ee[ii],axis=0),axis=1).sum()/max(t[ii[-1]]-t[ii[0]],1e-3)
        ev.append(dict(t0=round(float(t0),1), dB0=round(100*float(dB[min(i0,len(t)-1)]),1), dB_on_min=round(100*float(dB[on].min()),1),
                       v_on=round(100*v,1), clr_min=round(100*float(clr[blk].min()),1), B_during=bool(np.any(blk&(dB<0.04))),
                       delay=None if not len(after) else round(float(t[after[0]]-end),2), x_on=round(float(ee[ii,0].mean()),2)))
    x['reaches']=ev; x['fit']=[round(float(a),2), round(float(b),4), len(pr), len(W)]
json.dump(R,open(f'{S}/m2b6_runs.json','w'),indent=1)
for c in ('react','react_clip10','react_pmax20'):
    print('==',c)
    for f in (0.1,0.2,0.3,0.4,0.5):
        g=[x for x in R if x['cfg']==c and x['f']==f]; e=[v for x in g for v in x['reaches']]; d=[v['delay'] for v in e if v['delay'] is not None]
        print(f"  {f}: fit {[x['fit'][2:] for x in g]} reaches {len(e)} Bdur {sum(v['B_during'] for v in e)} on {np.median([v['dB_on_min'] for v in e]):.1f}/{min(v['dB_on_min'] for v in e):.1f} v {np.median([v['v_on'] for v in e]):.1f} clr {min(v['clr_min'] for v in e):.1f} delay {np.median(d):.1f} [{min(d):.1f}-{max(d):.1f}] dB0 {np.median([v['dB0'] for v in e]):.0f} late {sum(v['dB0']<30 for v in e)}")

"""wkM4 (mid, +/- jrk, reps 1-6) and wkB7 (reactive arm on B, jrk, reps 1-6): per-run metrics, jerk attribution, reactive reaches."""
import sys, os, json, glob, re, numpy as np
S='/tmp/claude-0/-root/28c21aa4-61e4-4f8c-b7ee-3d09b2142a5f/scratchpad'; RT='/root/worspace/cosmik_mppi_sim/.runtime'
src=open(S+'/next/test_next.py').read().split('# 1) defaults')[0]
cwd=os.getcwd(); os.chdir('/root/worspace/cosmik_mppi_sim/ros2_ws/src/storm_pick_place_demo/scripts'); exec(src); os.chdir(cwd)
Bp=np.array([-0.5,0.5,0.3])
CC=[('repeat-wkM4',c) for c in ('mid','midj','mid_clip10','midj_clip10','mid_cfix12','midj_cfix12')]+[('repeat-wkB7',c) for c in ('react','react_clip10')]
runs=[]
for camp,cfg in CC:
  for p in sorted(glob.glob(f'{RT}/{camp}/{cfg}/f*/rep*/freq_*/result.json')):
    d=p.rsplit('/',1)[0]; r=json.load(open(p)); f=float(p.split('/')[-4][1:]); rep=int(p.split('/')[-3][3:])
    js=r['messages'].get('joint_states',0)/max(r['simulated_seconds_after_first_goal'],1)
    tr=np.load(d+'/trace.npy'); t=tr[:,0]; ee=tr[:,1:4]; clr=tr[:,-1]; dB=np.linalg.norm(ee-Bp,axis=1)
    x=dict(cfg=cfg,f=f,rep=rep,dir=d,js=round(js,1),bad=js>32,B=r['spatial_arrivals'].get('B',0),pen=r['penetration_samples'],
           dmin=round(100*r['min_human_clearance_m'],1),to=r['timeout_events'])
    jrk='j' in cfg.split('_')[0] or cfg.startswith('react')
    h=make(jerk_enabled=True,jerk_seed=rep*100+round(f*10),sine_amplitudes=[0.15,0.45])
    ii=np.where(clr<0)[0]; eps=[]
    for i in ii:
        if eps and t[i]-t[eps[-1][-1]]<0.5: eps[-1].append(i)
        else: eps.append([i])
    x['eps']=[]
    for e in eps:
        t0=t[e[0]]; s=[a for a,_ in h.jerks if a<=t0+1e-6]; sb=t0-s[-1] if s else 99
        x['eps'].append(dict(t0=round(float(t0),1),dur=round(float(t[e[-1]]-t0),2),n=len(e),depth=round(100*float(clr[e].min()),1),since=round(sb,2),inbump=bool(jrk and sb<0.5),ee=[round(float(v),2) for v in ee[e[0]]]))
    if cfg.startswith('react'):
        m=1/(2*f); far=min(0.95,0.15+0.25*m); rec=[]; t0=None; prev=None
        for ti,di in zip(t,dB):
            if t0 is not None and ti-t0<2*m+4+3: prev=di; continue
            if prev is not None and di<prev-1e-4 and 0.20<=di<=far: t0=ti; rec.append(ti)
            prev=di
        W=[float(w) for w in re.findall(r'\[(\d+\.\d+)\] \[urdf_human\]: \[human-event\] block reach starts', open(d+'/launch.log').read())]
        if not rec or not W:
            print('NO REACH', cfg, f, rep, len(rec), len(W), 'B', x['B']); x['reaches']=[]; x['fit']=[0,len(W)]; runs.append(x); continue
        a,b=rec[0]-0.98*W[0],0.98; pr=[]
        for _ in range(3):
            pr=[(w,min(rec,key=lambda q:abs(q-(a+b*w)))) for w in W]; pr=[(w,q) for w,q in pr if abs(q-(a+b*w))<1.5]
            if len(pr)>=2: b,a=np.polyfit([p[0] for p in pr],[p[1] for p in pr],1)
        ev=[]
        for t0 in [a+b*w for w in W]:
            end=t0+2*m+4; on=(t>=t0+m)&(t<t0+m+4); blk=(t>=t0)&(t<end); jj=np.where(on)[0]
            if not len(jj): continue
            after=np.where((t>=end)&(dB<0.04))[0]
            ev.append(dict(t0=round(float(t0),1),dB_on_min=round(100*float(dB[on].min()),1),clr_min=round(100*float(clr[blk].min()),1),
                           B_during=bool(np.any(blk&(dB<0.04))),delay=None if not len(after) else round(float(t[after[0]]-end),2),x_on=round(float(ee[jj,0].mean()),2)))
        x['reaches']=ev; x['fit']=[len(pr),len(W)]
    runs.append(x)
json.dump(runs,open(S+'/m4_runs.json','w'),indent=1)
print('bad runs',[ (x['cfg'],x['f'],x['rep'],x['js']) for x in runs if x['bad']], 'n',len(runs))
for c in [c for _,c in CC]:
    X=[x for x in runs if x['cfg']==c]; ct=[x for x in X if x['pen']>0]; nc=[x['dmin'] for x in X if x['pen']==0]
    E=[e for x in X for e in x['eps']]; Eb=[e for e in E if e['inbump']]
    nb=sum(1 for x in X if any(not e['inbump'] for e in x['eps']))
    byf={f:sum(1 for x in X if x['f']==f and x['pen']>0) for f in (0.1,0.2,0.3,0.4,0.5)}
    print(f"{c:13s} n={len(X)} contact {len(ct)}/{len(X)} byf {list(byf.values())} worst {min(x['dmin'] for x in X)} samples {sum(x['pen'] for x in X)} med_nc {np.median(nc):.1f} B {np.mean([x['B'] for x in X]):.2f} to {sum(x['to'] for x in X)} eps {len(E)} inbump {len(Eb)} depth_b {sorted(e['depth'] for e in Eb)[:3]} nonbump_runs {nb} B0runs {sum(1 for x in X if x['B']<=2)}")
    if c.startswith('react'):
        R_=[e for x in X for e in x['reaches']]; dl=[e['delay'] for e in R_ if e['delay'] is not None]
        print(f"      reaches {len(R_)} B_during {sum(e['B_during'] for e in R_)} dB_on med {np.median([e['dB_on_min'] for e in R_]):.1f} min {min(e['dB_on_min'] for e in R_):.1f} x_on {np.median([e['x_on'] for e in R_]):.2f} delay med {np.median(dl):.1f} fit {[x['fit'] for x in X if x['fit'][0]<x['fit'][1]]}")
        for x in ct: print('      contact', x['f'], x['rep'], x['pen'], x['dmin'], [(e['t0'],e['depth'],e['since'],e['ee']) for e in x['eps']])

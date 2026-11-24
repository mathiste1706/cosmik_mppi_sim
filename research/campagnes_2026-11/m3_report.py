import json, glob, numpy as np, re
S='/tmp/claude-0/-root/28c21aa4-61e4-4f8c-b7ee-3d09b2142a5f/scratchpad'; R='/root/worspace/cosmik_mppi_sim/.runtime'
CC=[('repeat-wkM1','mid'),('repeat-wkM2','mid_clip10'),('repeat-wkM2','mid_cfix12')]+[('repeat-wkM3',c) for c in ('midj','midj_clip10','midj_cfix12')]
runs=[]
for camp,cfg in CC:
  for p in sorted(glob.glob(f'{R}/{camp}/{cfg}/f*/rep*/freq_*/result.json')):
    d=p.rsplit('/',1)[0]; r=json.load(open(p)); f=float(p.split('/')[-4][1:]); rep=int(p.split('/')[-3][3:])
    js=r['messages'].get('joint_states',0)/max(r['simulated_seconds_after_first_goal'],1)
    tr=np.load(d+'/trace.npy'); t=tr[:,0]; ee=tr[:,1:4]; clr=tr[:,-1]; ic=int(np.argmin(clr))
    lg=open(d+'/launch.log').read(); nj=len(re.findall(r'jerk',lg,re.I))
    seg={k.split('/')[-1]:round(100*v,1) for k,v in r['min_clearance_by_human_segment_m'].items()}
    runs.append(dict(camp=camp,cfg=cfg,f=f,rep=rep,dir=d,js=round(js,1),bad=js>32,B=r['spatial_arrivals'].get('B',0),pen=r['penetration_samples'],
      dmin=round(100*r['min_human_clearance_m'],1),to=r['timeout_events'],seg=seg,tmin=round(float(t[ic]),1),ee=[round(float(v),2) for v in ee[ic]],nj=nj))
json.dump(runs,open(f'{S}/m3_runs.json','w'),indent=1)
for x in runs: print(f"{x['cfg']:12s} {x['f']} r{x['rep']} js={x['js']} {'BAD' if x['bad'] else ''} B={x['B']:2d} pen={x['pen']:3d} dmin={x['dmin']:6.1f} to={x['to']} tmin={x['tmin']} ee={x['ee']} nj={x['nj']} seg={ {k:v for k,v in x['seg'].items() if v<10} }")

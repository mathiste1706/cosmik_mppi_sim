import sys,json,numpy as np,os
src=open('test_next.py').read().split('# 1) defaults')[0]
os.chdir('/root/worspace/cosmik_mppi_sim/ros2_ws/src/storm_pick_place_demo/scripts'); exec(src)
S='/tmp/claude-0/-root/28c21aa4-61e4-4f8c-b7ee-3d09b2142a5f/scratchpad'
R=json.load(open(S+'/m3_runs.json')); EP=[]
for r in R:
    seed=r['rep']*100+round(r['f']*10); h=make(jerk_enabled=True,jerk_seed=seed,sine_amplitudes=[0.15,0.45])
    tr=np.load(r['dir']+'/trace.npy'); t=tr[:,0]; clr=tr[:,-1]; ee=tr[:,1:4]
    r['njerk']=sum(1 for a,_ in h.jerks if t[0]<=a<=t[-1])
    ii=np.where(clr<0)[0]; eps=[]
    for i in ii:
        if eps and t[i]-t[eps[-1][-1]]<0.5: eps[-1].append(i)
        else: eps.append([i])
    r['eps']=[]
    for e in eps:
        t0=t[e[0]]; s=[a for a,_ in h.jerks if a<=t0+1e-6]; sb=t0-s[-1] if s else 99
        d=dict(t0=round(float(t0),1),dur=round(float(t[e[-1]]-t0),2),n=len(e),depth=round(100*float(clr[e].min()),1),since=round(sb,2),inbump=bool(sb<0.5),ee=[round(float(v),2) for v in ee[e[0]]])
        r['eps'].append(d)
json.dump(R,open(S+'/m3_runs.json','w'),indent=1)
for c in ['mid','midj','mid_clip10','midj_clip10','mid_cfix12','midj_cfix12']:
    x=[r for r in R if r['cfg']==c]; e=[d for r in x for d in r['eps']]; eb=[d for d in e if d['inbump']]
    runs_nb=sum(1 for r in x if any(not d['inbump'] for d in r['eps']))
    print(f"{c:12s} eps {len(e)} inbump {len(eb)} (samples {sum(d['n'] for d in eb)}/{sum(d['n'] for d in e)}, depth bump {sorted(d['depth'] for d in eb)}) ; runs with non-bump contact {runs_nb}/10; jerks/run {np.mean([r['njerk'] for r in x]):.1f}")
    print('    nonbump', [(r['f'],r['rep'],d['t0'],d['dur'],d['depth'],d['since'],d['ee']) for r in x for d in r['eps'] if not d['inbump']])

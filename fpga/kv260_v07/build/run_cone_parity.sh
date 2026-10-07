#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/../../.." && pwd)"
work="${CONE_PARITY_WORK_DIR:-$script_dir/work/cone_parity}"
mkdir -p "$work"
HLS_INCLUDE="${HLS_INCLUDE:-/tools/Xilinx/Vitis_HLS/2022.1/include}"
g++ -O2 -std=c++17 -Wno-unknown-pragmas -I"$HLS_INCLUDE" -I"$script_dir/../src" \
  "$script_dir/../src/snn_qp_v07_kernel.cpp" "$script_dir/../src/native_conic_v07.cpp" \
  -o "$work/native_conic" 2>"$work/native_compile.stderr"
PYTHONPATH="$root_dir" \
  "${PYTHON:-$root_dir/.venv/bin/python}" - "$work" <<'PY'
import json, struct, subprocess, sys
from pathlib import Path
import numpy as np
from fpga.kv260_v07.src.kernel_model import Cone, solve_v07, solve_native_reset, friction_objective, k0_for
from fpga.kv260_v07.tools import cone_reference as ref

work = Path(sys.argv[1]); native_exe = work / "native_conic"

def write_case(path, A, C, G, cns, scale, b, d, x0, *, k0, ctol, iters, projmax,
               has_lower=False, lower=0., has_upper=False, upper=0., cones=()):
    A=np.asarray(A,float); C=np.asarray(C,float).reshape((-1,A.shape[0])); G=np.asarray(G,float).reshape((C.shape[0],C.shape[0]))
    b=np.asarray(b,float).reshape(-1); d=np.asarray(d,float).reshape(-1); x0=np.asarray(x0,float).reshape(-1)
    cns=np.asarray(cns,float).reshape(-1); scale=np.asarray(scale,float).reshape(-1)
    with open(path,'wb') as f:
        f.write(struct.pack('<4I',0x56303743,A.shape[0],C.shape[0],len(cones)))
        f.write(struct.pack('<2diiidid',float(k0),float(ctol),int(iters),int(projmax),int(has_lower),float(lower),int(has_upper),float(upper)))
        for v in (A.ravel(),C.ravel(),G.ravel(),cns,scale,b,d,x0): f.write(np.asarray(v,dtype='<f8').tobytes())
        for field in ('kind','offset','length'):
            f.write(np.asarray([0 if c.kind=='ball' else 1 for c in cones] if field=='kind' else [getattr(c,field) for c in cones],dtype='<u4').tobytes())
        for field in ('radius','mu','center'):
            f.write(np.asarray([getattr(c,field) for c in cones],dtype='<f8').tobytes())

def run_native(case, payload):
    inp=work/(case+'.bin'); out=work/(case+'.native.json'); write_case(inp,**payload)
    subprocess.run([str(native_exe),str(inp),str(out)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    return json.loads(out.read_text())

def raw(x): return np.rint(np.asarray(x)*2**24).astype(np.int64).tolist()
def objective(A,b,x): return float(.5*np.asarray(x)@np.asarray(A)@np.asarray(x)+np.asarray(b)@np.asarray(x))
rows=[]
# Euclidean ball anchor, n=8.
n=8; A=np.eye(n); b=np.zeros(n); x0=np.r_[2.,np.zeros(n-1)]; ball=(Cone('ball',0,n,radius=1.),)
m=solve_v07(A,b,x0=x0,cones=ball,n_iters=3,projection_cap=32)
payload=dict(A=A,C=np.zeros((1,n)),G=np.zeros((1,1)),cns=np.array([1.]),scale=np.array([1.]),b=b,d=np.array([0.]),x0=x0,k0=.03,ctol=1e-6,iters=3,projmax=32,cones=ball)
r=run_native('ball8',payload); rows.append({'case':'ball-n8','model_raw':raw(m['x']),'native_raw':r['raw'],'raw_equal':raw(m['x'])==r['raw'],'model_events':len(m['events']),'native_events':r['events'],'native_status':r['status']})
# Mixed row + upper bound + ball + scaled SOC anchor.
n=6; A=np.eye(n); b=np.zeros(n); C=np.array([[1.,0,0,0,0,0]]); d=np.array([-.25]); x0=np.array([2.,0,0,-1,2,0.]); cones=(Cone('ball',0,3,radius=1.),Cone('scaled_soc',3,3,mu=.4))
m=solve_v07(A,b,C=C,d=d,x0=x0,cones=cones,upper=.75,n_iters=3,projection_cap=32)
payload=dict(A=A,C=C,G=C@C.T,cns=np.array([1.]),scale=np.array([1.]),b=b,d=d,x0=x0,k0=.03,ctol=1e-6,iters=3,projmax=32,has_upper=True,upper=.75,cones=cones)
r=run_native('mixed',payload); rows.append({'case':'mixed-row-bound-ball-soc','model_raw':raw(m['x']),'native_raw':r['raw'],'raw_equal':raw(m['x'])==r['raw'],'model_events':len(m['events']),'native_events':r['events'],'native_status':r['status']})
# q=1 and q=4 friction anchors for each recorded seed.
for q in (1,4):
  for seed in (4903,4914,4925):
    A,b,target=friction_objective(seed,q); cones=tuple(Cone('scaled_soc',3*i,3,mu=.4) for i in range(q)); iters=1024; cap=64
    fixed=solve_native_reset(A,b,np.zeros(3*q),mu=.4,contacts=q,k0=k0_for(A),n_iters=iters,projection_cap=cap)
    double=ref.double_reference(A,b,q,.4,iters,cap); clar=ref.clarabel_reference(A,b,q,.4)
    agree,first=ref.event_agreement(fixed.events,double['events'])
    payload=dict(A=A,C=np.zeros((1,3*q)),G=np.zeros((1,1)),cns=np.array([1.]),scale=np.array([1.]),b=b,d=np.array([0.]),x0=np.zeros(3*q),k0=k0_for(A),ctol=1e-6,iters=iters,projmax=cap,cones=cones)
    r=run_native(f'f{q}_{seed}',payload); mr=raw(fixed.x)
    rows.append({'case':f'friction-q{q}-mu0.4-seed{seed}','q':q,'seed':seed,'iterations':iters,'model_raw':mr,'native_raw':r['raw'],'raw_equal':mr==r['raw'],'model_events':len(fixed.events),'native_events':r['events'],'event_agreement_vs_binary64':agree,'first_divergence_step':first,'state_gap_vs_binary64':float(np.max(np.abs(fixed.x-double['final']))),'model_objective':objective(A,b,fixed.x),'binary64_objective':double['objective'],'clarabel_status':clar.get('status'),'clarabel_objective':clar.get('objective'),'model_clarabel_relative_gap':float(abs(objective(A,b,fixed.x)-clar.get('objective'))/max(1.,abs(clar.get('objective')))) if np.isfinite(clar.get('objective',np.nan)) else None,'native_status':r['status']})
# Randomized mixed battery: arbitrary Hessians and nonzero affine data, with
# disjoint ball/SOC blocks, rows, and bounds.  This exercises winner ordering
# and register arithmetic outside the hand-picked anchors.
rng=np.random.default_rng(20460924)
random_rows=[]
for case_id in range(50):
  n=int(rng.integers(3,9)); k=int(rng.integers(1,4)); cursor=0; cones=[]
  for j in range(k):
    remaining=n-cursor; left=k-j-1
    if remaining < (1 if j == k-1 else 1)+3*left: break
    if j == k-1:
      length=int(rng.integers(1,remaining+1))
    else:
      max_len=remaining-3*left
      length=int(rng.integers(1,max_len+1))
    if length >= 3 and (j % 2 == 1 or rng.random() < .5):
      kind='scaled_soc'; length=max(3,length); mu=float(rng.uniform(.08,2.5))
      cones.append(Cone(kind,cursor,length,mu=mu))
    else:
      kind='ball'; radius=float(rng.uniform(.2,2.0)); center=float(rng.uniform(-1,1))
      cones.append(Cone(kind,cursor,length,radius=radius,center=center))
    cursor += length
  if not cones: cones=(Cone('ball',0,1,radius=1.),)
  cones=tuple(cones)
  A=rng.normal(size=(n,n)); A=A.T@A+.5*np.eye(n); b=rng.normal(size=n)
  x0=rng.normal(size=n); mrows=int(rng.integers(1,3)); C=rng.normal(size=(mrows,n)); d=rng.normal(size=mrows)
  lower=float(rng.uniform(-2,-.2)) if case_id%2 else None; upper=float(rng.uniform(.2,2)) if case_id%3 else None
  k0=float(k0_for(A)); iters=2; cap=4
  model=solve_v07(A,b,C=C,d=d,x0=x0,cones=cones,k0=k0,n_iters=iters,projection_cap=cap,lower=lower,upper=upper)
  payload=dict(A=A,C=C,G=C@C.T,cns=np.sum(C*C,axis=1),scale=np.ones(mrows),b=b,d=d,x0=x0,k0=k0,ctol=1e-6,iters=iters,projmax=cap,has_lower=lower is not None,lower=0 if lower is None else lower,has_upper=upper is not None,upper=0 if upper is None else upper,cones=cones)
  native=run_native(f'random_{case_id:02d}',payload)
  equal=raw(model['x'])==native['raw']
  random_rows.append({'case':case_id,'n':n,'cone_count':len(cones),'raw_equal':equal,'model_events':len(model['events']),'native_events':native['events'],'model_status':model['status'],'native_status':native['status']})
  if not equal: print('RANDOM_MISMATCH',json.dumps(random_rows[-1],sort_keys=True))
rows.append({'case':'randomized-mixed-battery','count':len(random_rows),'raw_equal':all(r['raw_equal'] for r in random_rows),'rows':random_rows})
# Review regressions, retained as executable parity rows.
review_rows=[]
def review_case(name, A, b, x0, cones, **kw):
  C=kw.pop('C',np.zeros((1,len(b)))); d=kw.pop('d',np.zeros(len(C)))
  k0=kw.pop('k0',.03); ctol=kw.pop('ctol',1e-6); iters=kw.pop('iters',1); cap=kw.pop('projmax',1)
  lower=kw.pop('lower',None); upper=kw.pop('upper',None)
  model=solve_v07(A,b,C=C,d=d,x0=x0,cones=cones,k0=k0,ctol=ctol,n_iters=iters,projection_cap=cap,lower=lower,upper=upper)
  payload=dict(A=A,C=C,G=C@C.T,cns=np.sum(C*C,axis=1),scale=np.ones(len(C)),b=b,d=d,x0=x0,k0=k0,ctol=ctol,iters=iters,projmax=cap,has_lower=lower is not None,lower=0 if lower is None else lower,has_upper=upper is not None,upper=0 if upper is None else upper,cones=cones)
  native=run_native('review_'+name,payload)
  review_rows.append({'case':'review-'+name,'raw_equal':raw(model['x'])==native['raw'],'model_raw':raw(model['x']),'native_raw':native['raw'],'model_events':len(model['events']),'native_events':native['events'],'model_status':model['status'],'native_status':native['status']})
# Crossed SOC rankings and SOC-vs-row ranking.
n=6; A=np.zeros((n,n)); b=np.zeros(n); x0=np.array([0.,1.,0.,0.,1.5,0.]); cones=(Cone('scaled_soc',0,3,mu=.1),Cone('scaled_soc',3,3,mu=2.)); review_case('cross_rank_soc',A,b,x0,cones,k0=0.,iters=1,projmax=1)
n=3; A=np.zeros((n,n)); b=np.zeros(n); x0=np.array([0.,1.2,0.]); C=np.array([[0.,1.,0.]]); d=np.array([-.2]); review_case('row_between_soc',A,b,x0,(Cone('scaled_soc',0,3,mu=1.),),C=C,d=d,k0=0.,iters=1,projmax=1)
# Apex-vs-radial SOC winner.
n=6; x0=np.array([-3.,.1,0.,0.,1.5,0.]); review_case('apex_vs_radial',np.zeros((n,n)),np.zeros(n),x0,(Cone('scaled_soc',0,3,mu=.4),Cone('scaled_soc',3,3,mu=.4)),k0=.03,iters=1,projmax=1)
# Interior ball and nonidentity Hessian.
n=4; A=np.eye(n); b=np.array([4.,-2.5,.125,3.]); x0=np.array([.5,-.25,.125,-1.]); review_case('interior_ball',A,b,x0,(Cone('ball',0,n,radius=50.),),k0=.03,iters=1,projmax=1)
rng_probe=np.random.default_rng(7); R=rng_probe.standard_normal((n,n)); A=R.T@R+2*np.eye(n); b=rng_probe.standard_normal(n); x0=rng_probe.standard_normal(n)*2.; review_case('nonidentity_ball',A,b,x0,(Cone('ball',0,n,radius=1.),),k0=k0_for(A),iters=16,projmax=32)
# Range-edge ball and tiny-norm SOC ratio.
n=2; review_case('range_edge_ball',np.zeros((n,n)),np.zeros(n),np.array([127.9,-127.]),(Cone('ball',0,2,radius=1.,center=-128.),),k0=.03,iters=1,projmax=4)
n=3; review_case('tiny_norm_soc',np.zeros((n,n)),np.zeros(n),np.array([1e-5,1.5e-5,0.]),(Cone('scaled_soc',0,3,mu=.4),),k0=.03,iters=1,projmax=1)
# Top-bit norm regressions: widened norm register and normalized mantissa.
for nn in (7,8):
  review_case(f'ball_topbit_n{nn}',np.zeros((nn,nn)),np.zeros(nn),np.full(nn,100.),(Cone('ball',0,nn,radius=1.),),k0=0.,iters=1,projmax=4)
rows.extend(review_rows)
output={'schema':'v07-resident-cone-parity-v2','resident_top':'snn_qp_v07','rows':rows}
(work/'anchors.json').write_text(json.dumps(output,indent=2,sort_keys=True))
for row in rows: print(json.dumps(row,sort_keys=True))
PY
printf 'NATIVE COMPILE: '; if [[ -s "$work/native_compile.stderr" ]]; then tail -1 "$work/native_compile.stderr"; else echo clean; fi
printf 'ANCHORS: '; python3 -c 'import json; x=json.load(open("'"$work"'/anchors.json")); print(json.dumps(x,sort_keys=True,separators=(",",":")))'

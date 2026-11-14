/* T4 KartPlex - fastest-lap AI (owner: 'the best line in terms of time, not the best mathematical one; it can use trail
   braking etc. how a pro would drive').
   1. Usable track: from the viewer's own surface map, scanning across the track every 0.75 m. Kerbs and asphalt
      past the paint count (as in the sim: a lap is only invalid with all four wheels off); grass / gravel and lanes beyond
      the painted edge (pit lane, link roads) do not.
   2. Line: lap-time minimisation. A quasi-steady lap simulation of the T4 Jr (same mass, power, drive force, drag, rolling
      resistance, rear-axle braking with load transfer and surface grip as the drive physics) with a friction ellipse, so
      braking, cornering and acceleration share the tyre: braking bleeds off into the turn-in (trail braking) and power comes
      in as the wheel unwinds. Gradient descent on the line offsets (finite differences, Adam), inside the usable track.
   3. Driver: an AI drives that line with the REAL kart physics (tyre slip, load transfer, solid-axle scrub, kerbs, grip
      patches) and learns, lap by lap, how much speed each part of the lap really takes: slower where it slid wide or
      off, faster where it was clean. The lap-time model is then re-calibrated from what the kart actually did and the
      line re-optimised; two rounds.
   Usage (local viewer): window.T4OPT.run().then(r=>...) ; saves data/bestline.json through the dev server. */
(()=>{
const g=9.81, DS=0.75, ON=new Set(['asphalt','grip','patch','kerb']);
const T=()=>window.__t4drive;
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const MC=new MessageChannel(); let MCr=null; MC.port1.onmessage=()=>{ const r=MCr; MCr=null; r&&r(); };
const yieldNow=()=>new Promise(r=>{ MCr=r; MC.port2.postMessage(0); });   // yields to the page without timer throttling (works with the pane hidden)
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
const log=(...a)=>{ console.log('[T4OPT]',...a); (window.T4OPT.logs=window.T4OPT.logs||[]).push(a.join(' ')); };

// ---------- geometry ----------
function resampleClosed(P,ds){
  const n=P.length, cum=[0]; for(let i=1;i<=n;i++){ const a=P[i-1],b=P[i%n]; cum.push(cum[i-1]+Math.hypot(b[0]-a[0],b[1]-a[1])); }
  const L=cum[n], N=Math.round(L/ds), out=[]; let k=0;
  for(let i=0;i<N;i++){ const s=i*L/N; while(cum[k+1]<s) k++; const a=P[k%n],b=P[(k+1)%n],t=(s-cum[k])/Math.max(cum[k+1]-cum[k],1e-9);
    out.push([a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t]); }
  return {pts:out,L};
}
function frames(P){ const N=P.length, H=new Float64Array(N), nx=new Float64Array(N), nz=new Float64Array(N);
  for(let i=0;i<N;i++){ const a=P[(i-1+N)%N], b=P[(i+1)%N]; const h=Math.atan2(b[0]-a[0],b[1]-a[1]); H[i]=h; nx[i]=Math.cos(h); nz[i]=-Math.sin(h); }  // forward (sin h,cos h); left (cos h,-sin h)
  return {H,nx,nz}; }

// ---------- usable track width ----------
async function trackBounds(P,F){
  // the painted lines are the track limits for the AI exactly as for the player (limits_build.py -> data/limits.json)
  try{ const d=await (await fetch('data/limits.json')).json(), L=T().LINE, M=L.length;
    if(d&&d.left&&d.left.length===M){ const N=P.length, bl=new Float64Array(N), br=new Float64Array(N); let j=0;
      for(let i=0;i<N;i++){ let best=1e18; if(i===0){ for(let k=0;k<M;k++){ const dd=(L[k][0]-P[i][0])**2+(L[k][1]-P[i][1])**2; if(dd<best){best=dd;j=k;} } }
        else { const j0=j; for(let k=-8;k<=8;k++){ const kk=(j0+k+M)%M, dd=(L[kk][0]-P[i][0])**2+(L[kk][1]-P[i][1])**2; if(dd<best){best=dd;j=kk;} } }
        const a=L[(j-1+M)%M], b=L[(j+1)%M], ln=Math.hypot(b[0]-a[0],b[1]-a[1])||1, fx=(b[0]-a[0])/ln, fz=(b[1]-a[1])/ln;
        const o=(P[i][0]-L[j][0])*fz+(P[i][1]-L[j][1])*(-fx);        // + = left of the reference line
        bl[i]=Math.max(0,d.left[j]-o); br[i]=Math.max(0,d.right[j]+o); }
      log('track limits from data/limits.json (painted lines)'); return {bl,br}; } }catch(e){ log('limits.json missing - scanning'); }
  const paint=await (await fetch('data/paint.json')).json(), grid=new Map();
  for(const r of paint.edge){ for(let i=1;i<r.length;i++){ const a=r[i-1],b=r[i],L=Math.hypot(b[0]-a[0],b[1]-a[1]),m=Math.max(1,Math.ceil(L/0.04));
      for(let k=0;k<=m;k++){ const x=a[0]+(b[0]-a[0])*k/m, z=-(a[1]+(b[1]-a[1])*k/m), key=Math.floor(x)+','+Math.floor(z); (grid.get(key)||grid.set(key,[]).get(key)).push(x,z); } } }
  const nearPaint=(x,z)=>{ for(let i=-1;i<=1;i++) for(let j=-1;j<=1;j++){ const L=grid.get((Math.floor(x)+i)+','+(Math.floor(z)+j)); if(!L) continue;
      for(let k=0;k<L.length;k+=2){ const dx=L[k]-x,dz=L[k+1]-z; if(dx*dx+dz*dz<0.0064) return true; } } return false; };
  const N=P.length, bl=new Float64Array(N), br=new Float64Array(N), sf=T().surfAt;
  for(let i=0;i<N;i++){
    for(const side of [1,-1]){ let d=0.1, crossed=-1, last=0;
      for(;d<=8.0;d+=0.1){ const x=P[i][0]+side*F.nx[i]*d, z=P[i][1]+side*F.nz[i]*d;
        if(!ON.has(sf(x,z))) break;
        if(crossed<0&&nearPaint(x,z)) crossed=d;
        if(crossed>=0&&d>crossed+1.15) break;                 // at most a kerb / run-off strip past the painted edge
        last=d; }
      (side>0?bl:br)[i]=last; }
    if(i%200===0) await yieldNow();
  }
  const mn=(A,w)=>{ const B=new Float64Array(N); for(let i=0;i<N;i++){ let m=1e9; for(let k=-w;k<=w;k++) m=Math.min(m,A[(i+k+N)%N]); B[i]=m; } return B; };
  return {bl:mn(bl,3),br:mn(br,3)};
}

// ---------- lap-time model (quasi-steady, friction ellipse) ----------
function makeModel(P,F,B,KM,SURFP){
  const N=P.length, sf=T().surfAt, O0=-6, OD=0.25, NO=49;                 // grip / rolling resistance across the track, every 0.25 m
  const MU=new Float32Array(N*NO), CR=new Float32Array(N*NO);
  for(let i=0;i<N;i++) for(let j=0;j<NO;j++){ const o=O0+j*OD, q=sf(P[i][0]+F.nx[i]*o,P[i][1]+F.nz[i]*o), s=SURFP[q]||SURFP.grass; MU[i*NO+j]=s.mu*(q==='kerb'?0.95:1); CR[i*NO+j]=s.crr*(q==='kerb'?1.2:1); }   // kerb: flat, ridden hard (owner: use more kerb)
  const at=(A,i,o)=>{ const f=Math.min(NO-1.001,Math.max(0,(o-O0)/OD)), j=f|0, t=f-j; return A[i*NO+j]*(1-t)+A[i*NO+j+1]*t; };
  const L=KM.a+KM.b;
  const brakeMax=mu=>{ let Fx=-1.2*KM.m*g; for(let it=0;it<8;it++){ const Nr=Math.max(0,KM.m*g*KM.a/L+Fx*KM.hcg/L); Fx=-Math.min(1.2*KM.m*g,0.9*KM.mu0*mu*Nr); } return -Fx/KM.m; };
  const tracMax=mu=>{ let Fx=KM.Fdrive; for(let it=0;it<8;it++){ const Nr=KM.m*g*KM.a/L+Fx*KM.hcg/L; Fx=Math.min(KM.Fdrive,0.95*KM.mu0*mu*Nr); } return Fx; };   // traction cap; the engine force itself comes from KM.drive(v)
  const TB=new Float64Array(201), TT=new Float64Array(201); for(let k=0;k<=200;k++){ TB[k]=brakeMax(k/100); TT[k]=tracMax(k/100); }   // lookup by grip (0..2)
  const lut=(A,m)=>{ const f=Math.min(199.999,Math.max(0,m*100)), j=f|0, t=f-j; return A[j]*(1-t)+A[j+1]*t; };
  const qx=new Float64Array(N), qz=new Float64Array(N), ds=new Float64Array(N), kap=new Float64Array(N), mu=new Float64Array(N), cr=new Float64Array(N),
        vl=new Float64Array(N), vf=new Float64Array(N), vb=new Float64Array(N), ax=new Float64Array(N);
  const P0x=Float64Array.from(P,p=>p[0]), P0z=Float64Array.from(P,p=>p[1]);
  const prm={klat:0.92,kbrk:1.0,ktrac:1.0};
  function lap(off,keep){
    for(let i=0;i<N;i++){ qx[i]=P0x[i]+F.nx[i]*off[i]; qz[i]=P0z[i]+F.nz[i]*off[i];
      mu[i]=0.5*(at(MU,i,off[i]-0.5)+at(MU,i,off[i]+0.5)); cr[i]=0.5*(at(CR,i,off[i]-0.5)+at(CR,i,off[i]+0.5)); }
    for(let i=0;i<N;i++){ const j=(i+1)%N; ds[i]=Math.hypot(qx[j]-qx[i],qz[j]-qz[i]); }
    for(let i=0;i<N;i++){ const a=(i-2+N)%N, b=(i+2)%N;                                            // signed curvature over a 1.5 m chord (+ = left)
      const h1=Math.atan2(qx[i]-qx[a],qz[i]-qz[a]), h2=Math.atan2(qx[b]-qx[i],qz[b]-qz[i]);
      const s=Math.hypot(qx[i]-qx[a],qz[i]-qz[a])+Math.hypot(qx[b]-qx[i],qz[b]-qz[i]); kap[i]=2*wrap(h2-h1)/Math.max(s,1e-6); }
    for(let i=0;i<N;i++){ const ay=KM.mu0*mu[i]*g*prm.klat; vl[i]=Math.min(40,Math.sqrt(ay/Math.max(Math.abs(kap[i]),1e-4))); }
    const drag=v=>0.5*1.2*KM.CdA*v*v/KM.m;
    // forward (acceleration), twice round for the closed lap
    let v=vl[0]; for(let pass=0;pass<2;pass++) for(let i=0;i<N;i++){ v=Math.min(v,vl[i]); vf[i]=v;
      const ayM=KM.mu0*mu[i]*g*prm.klat, ay=v*v*Math.abs(kap[i]), rem=Math.sqrt(Math.max(0,1-(ay/ayM)**2));
      const a=rem*Math.min(KM.drive?KM.drive(v):KM.P/Math.max(v,2.5),lut(TT,mu[i])*prm.ktrac)*Math.min(1,Math.max(0,((KM.vmax||99)-v)/0.6))/KM.m-drag(v)-cr[i]*g; v=Math.sqrt(Math.max(0.5,v*v+2*a*ds[i])); }
    // backward (braking): the brake force the ellipse leaves while turning -> braking overlaps turn-in (trail braking)
    v=vf[N-1]; for(let pass=0;pass<2;pass++) for(let i=N-1;i>=0;i--){ v=Math.min(v,vf[i]); vb[i]=v;
      const ayM=KM.mu0*mu[i]*g*prm.klat, ay=v*v*Math.abs(kap[i]), rem=Math.sqrt(Math.max(0,1-(ay/ayM)**2));
      const a=rem*lut(TB,mu[i])*prm.kbrk+drag(v)+cr[i]*g; v=Math.sqrt(v*v+2*a*ds[(i-1+N)%N]); }
    let t=0; for(let i=0;i<N;i++){ const j=(i+1)%N; t+=2*ds[i]/(vb[i]+vb[j]); }
    if(keep){ for(let i=0;i<N;i++){ const j=(i+1)%N; ax[i]=(vb[j]*vb[j]-vb[i]*vb[i])/(2*ds[i]); }
      return {t,x:Array.from(qx),z:Array.from(qz),v:Array.from(vb),ax:Array.from(ax),kap:Array.from(kap),mu:Array.from(mu)}; }
    return t;
  }
  return {lap,prm,N};
}

// ---------- line optimisation (offsets at control points every 3 m, periodic Catmull-Rom) ----------
async function optimise(model,B,init,iters,tag){
  const N=model.N, M=4, K=Math.floor(N/M), off=new Float64Array(N);
  const lo=new Float64Array(K), hi=new Float64Array(K); const marg=-0.45;   // owner: two wheels inside is legal -> the centre may sit 0.45 m past the limit
  for(let k=0;k<K;k++){ let l=1e9,h=1e9; for(let j=-2;j<=2;j++){ const i=(k*M+j+N)%N; l=Math.min(l,B.br[i]); h=Math.min(h,B.bl[i]); } lo[k]=-(l-marg); hi[k]=h-marg; if(lo[k]>hi[k]){ const m=(lo[k]+hi[k])/2; lo[k]=hi[k]=m; } }
  const a=new Float64Array(K); for(let k=0;k<K;k++) a[k]=Math.min(hi[k],Math.max(lo[k],init?init[k*M]:0));
  const expand=(c)=>{ for(let k=0;k<K;k++){ const p0=c[(k-1+K)%K],p1=c[k],p2=c[(k+1)%K],p3=c[(k+2)%K];
      for(let j=0;j<M;j++){ const t=j/M,t2=t*t,t3=t2*t; off[k*M+j]=0.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t2+(-p0+3*p1-3*p2+p3)*t3); } }
    for(let i=K*M;i<N;i++) off[i]=c[K-1]; return off; };
  const Tof=c=>model.lap(expand(c));
  let best=Tof(a), bestA=Float64Array.from(a), mA=new Float64Array(K), vA=new Float64Array(K), gr=new Float64Array(K); const eps=0.04;
  log(tag,'start',best.toFixed(3),'s');
  for(let it=1;it<=iters;it++){
    const base=Tof(a);
    for(let k=0;k<K;k++){ const s=a[k]; a[k]=Math.min(hi[k],s+eps); const tp=Tof(a); a[k]=Math.max(lo[k],s-eps); const tm=Tof(a); a[k]=s; gr[k]=(tp-tm)/(Math.max(1e-6,Math.min(hi[k],s+eps)-Math.max(lo[k],s-eps))); }
    const lr=0.06*Math.pow(0.996,it);
    for(let k=0;k<K;k++){ mA[k]=0.9*mA[k]+0.1*gr[k]; vA[k]=0.999*vA[k]+0.001*gr[k]*gr[k];
      const mh=mA[k]/(1-Math.pow(0.9,it)), vh=vA[k]/(1-Math.pow(0.999,it)); a[k]=Math.min(hi[k],Math.max(lo[k],a[k]-lr*mh/(Math.sqrt(vh)+1e-9))); }
    if(it%10===0){ const t=Tof(a); if(t<best){ best=t; bestA.set(a); } if(it%40===0) log(tag,'iter',it,t.toFixed(3),'s (best',best.toFixed(3)+')'); await yieldNow(); }
  }
  a.set(bestA); return {off:Float64Array.from(expand(a)),t:Tof(a)};
}

// ---------- AI driver on the real physics ----------
function makeDriver(path,KM,PADC,REF){
  const N=path.x.length, sc=new Float64Array(N).fill(0.97), limit=new Uint8Array(N);
  const H=new Float64Array(N), kp=Float64Array.from(path.kap);
  for(let i=0;i<N;i++){ const a=(i-1+N)%N,b=(i+1)%N; H[i]=Math.atan2(path.x[b]-path.x[a],path.z[b]-path.z[a]); }
  function lapRun(learn,startIdx,laps){ const g=lapRunG(learn,startIdx,laps,false); let r; do{ r=g.next(); }while(!r.done); return r.value; }
  function* lapRunG(learn,startIdx,laps,chunked){
    const D=T().DRV, phys=T().physics, h=1/300, Lw=KM.a+KM.b;
    const save=JSON.stringify(D), MR=Math.random; let sd=12345; const RND=()=>{ sd=(sd*1664525+1013904223)>>>0; return sd/4294967296; }; Math.random=RND;   // deterministic kerb jolts: fair comparisons
    let i=startIdx, x=path.x[i], z=path.z[i];
    Object.assign(D,{x,z,h:H[i],vx:Math.sin(H[i])*path.v[i]*0.9,vz:Math.cos(H[i])*path.v[i]*0.9,rr:path.v[i]*kp[i]*0.9,steer:0,thr:0,brk:0,kj:0,whP:null,rev:0,lapOn:false,valid:true,tyreT:65});   // AI laps on warm tyres
    const bad=new Uint8Array(N), errMax=new Float64Array(N), times=[], rec={thr:new Float64Array(N),brk:new Float64Array(N),v:new Float64Array(N),steer:new Float64Array(N),lat:new Float64Array(N)};
    let t=0, lapStart=null, crossed=0, prevI=i, maxSteps=Math.round((laps+1.3)*80/h), traj=[], trajBest=null, tBest=1e9, nextRec=0;
    for(let step=0;step<maxSteps;step++){
      if(chunked&&step&&step%250===0){ Math.random=MR; yield; Math.random=RND; }                 // live learning: give the frame back to the player
      // nearest path point (local search)
      let bi=i, bd=1e9; for(let k=-15;k<=40;k++){ const j=(i+k+N)%N, dx=D.x-path.x[j], dz=D.z-path.z[j], d=dx*dx+dz*dz; if(d<bd){ bd=d; bi=j; } } i=bi;
      const u=D.vx*Math.sin(D.h)+D.vz*Math.cos(D.h);
      const nx=Math.cos(H[i]), nz=-Math.sin(H[i]), e=(D.x-path.x[i])*nx+(D.z-path.z[i])*nz, he=wrap(H[i]-D.h);
      // steering: path curvature feed-forward (a little ahead) + heading / lateral feedback + yaw-rate damping
      const ja=(i+Math.round(Math.max(1,u*0.12)/0.75))%N, kff=kp[ja];
      let delta=Math.atan(Lw*kff)+1.15*he-Math.atan(1.6*e/(Math.abs(u)+2))+0.13*(kff*u-D.rr);   // stronger yaw-rate correction = catches the rear like a pro (countersteer)
      const dmax=0.34/(1+u*u/(PADC.wheel?1600:420)); D.steer=Math.max(-1,Math.min(1,delta/dmax));
      // speed: learned target along the line; brakes ease off as the corner tightens = trail braking
      const jl=(i+Math.round(Math.max(1,u*0.10)/0.75))%N, vt=Math.min(sc[i]*path.v[i],sc[jl]*path.v[jl]), err=vt-u;
      const sliding=(D.slip||0)>0.14;                                                          // rear stepping out: feed the throttle in, don't stamp on it
      if(err>0.05){ D.thr=sliding?0.45:1; D.brk=0; } else if(err<-0.25){ D.thr=0; D.brk=Math.min(1,0.25-err*1.1); } else { D.thr=Math.max(0,0.5+err*2); D.brk=0; }   // pro: flat out or firm brake, trail off into the corner
      D.hitV=0; phys(h); t+=h; const hitNow=D.hitV||0;
      // lap timing at path index 0 (start / finish line)
      if(prevI>N-40&&i<40){ if(lapStart!==null){ const lt=t-lapStart; times.push(lt); if(lt<tBest){ tBest=lt; trajBest=traj; } } lapStart=t; traj=[]; nextRec=t; crossed++; if(crossed>laps) break; }
      if(lapStart!==null&&t>=nextRec){ traj.push([+D.x.toFixed(3),+D.z.toFixed(3),+D.h.toFixed(4),+D.steer.toFixed(3)]); nextRec+=0.05; }   // ghost: 20 Hz from the timing line
      prevI=i;
      if(crossed>=1){ rec.thr[i]=D.thr; rec.brk[i]=D.brk; rec.v[i]=u; rec.steer[i]=D.steer; rec.lat[i]=D.latG||0;
        errMax[i]=Math.max(errMax[i],Math.abs(e));
        let offT=false; if(REF){ const o=(D.x-REF.P[i][0])*REF.F.nx[i]+(D.z-REF.P[i][1])*REF.F.nz[i]; offT=o>REF.B.bl[i]+0.52||o<-(REF.B.br[i]+0.52); }   // three wheels past the limit = track limits (two inside is fine, same rule as the sim)
        if(offT||hitNow>0.8) bad[i]=1; else if(Math.abs(he)>0.6) bad[i]=2; else if(Math.abs(e)>1.6) bad[i]=3; }   // wall contact counts as a mistake too                                                // or a spin / big slide - not a kerb wobble
      if(Math.abs(e)>3.5||u<1){ bad[i]=1; break; }                                             // lost it: stop this run
    }
    const res={times,bad,errMax,rec,ok:times.length>0,traj:trajBest};
    Object.assign(D,JSON.parse(save)); Math.random=MR;
    if(learn){ const dec=new Uint8Array(N);
      for(let k=0;k<N;k++) if(bad[k]){ limit[k]=1; for(let j=-45;j<=4;j++) dec[(k+j+N)%N]=1; }                 // slid wide / off: back off into that corner
      for(let k=0;k<N;k++) if(dec[k]) sc[k]=Math.max(0.6,sc[k]*0.985);
      if(res.ok) for(let k=0;k<N;k++){ if(dec[k]) continue; let clean=true; for(let j=-10;j<=10;j++) if(errMax[(k+j+N)%N]>0.9){ clean=false; break; }
        if(clean) sc[k]=Math.min(1.25,sc[k]*1.01); } }                                                         // clean: push a bit harder next lap
    return res;
  }
  return {lapRun,lapRunG,sc,H,limit};
}

async function learnLaps(drv,start,rounds,tag,path){
  let best=null; const hits=new Float64Array(drv.sc.length), why={};
  for(let r=0;r<rounds;r++){
    const scBefore=Float64Array.from(drv.sc), res=drv.lapRun(true,start,2);
    const clean=res.ok&&!res.bad.some(b=>b); for(let k=0;k<hits.length;k++){ hits[k]+=res.bad[k]?1:0; if(res.bad[k]) (why[k]=why[k]||[0,0,0,0])[res.bad[k]]++; }
    if(res.times.length) log(tag,'lap',r,res.times.map(t=>t.toFixed(3)).join(' / '),clean?'clean':'(slid wide)');
    if(clean){ const t=Math.min(...res.times); if(!best||t<best.t) best={t,rec:res.rec,sc:scBefore,traj:res.traj}; }
    await yieldNow();
  }
  if(path){ const N=hits.length, cl=[]; for(let k=0;k<N;k++){ if(hits[k]<3) continue; const c=cl.find(q=>Math.abs(q.k-k)<20||Math.abs(q.k-k)>N-20); if(c){ c.n+=hits[k]; } else cl.push({k,n:hits[k]}); }
    cl.sort((a,b)=>b.n-a.n); for(const c of cl.slice(0,6)) log(tag,'trouble spot at e',path.x[c.k].toFixed(1),'n',(-path.z[c.k]).toFixed(1),'idx',c.k,'hits',c.n,'curv',path.kap[c.k].toFixed(3),'v',path.v[c.k].toFixed(1),'why off/spin/wide',(()=>{ const w=[0,0,0,0]; for(let j=-20;j<=20;j++){ const q=why[(c.k+j+N)%N]; if(q) for(let r=1;r<4;r++) w[r]+=q[r]; } return w.slice(1).join('/'); })()); }
  return best;
}

// ---------- final stage: the AI tries small line changes corner by corner on the REAL physics, keeps what is faster ----------
async function refine(model,B,REF,offBase,scBase,KM,PADC,rounds,opts_refT){
  const N=model.N, marg=-0.45, lo=Float64Array.from(B.br,v=>-(v-marg)), hi=Float64Array.from(B.bl,v=>v-marg);
  let off=Float64Array.from(offBase);
  const evalLap=o=>{ const p=model.lap(o,true), d=makeDriver(p,KM,PADC,REF); d.sc.set(scBase); const r=d.lapRun(false,0,2);
    return (r.ok&&!r.bad.some(b=>b))?r.times[r.times.length-1]:Infinity; };
  let bestT=evalLap(off); log('refine start',bestT.toFixed(3),'s'); if(!isFinite(bestT)) bestT=opts_refT||1e9;
  const kap=model.lap(off,true).kap, corners=[];
  for(let i=0;i<N;i++){ const a=Math.abs(kap[i]); if(a<0.035) continue; let pk=true; for(let j=-14;j<=14;j++) if(Math.abs(kap[(i+j+N)%N])>a){ pk=false; break; } if(pk) corners.push(i); }
  log('refine:',corners.length,'corners');
  for(let r=0;r<rounds;r++){ const step=0.32/(r+1); let gained=0;
    for(const c of corners) for(const sh of [-20,0,20]) for(const dir of [1,-1]){
      const tr=Float64Array.from(off), mid=c+sh;
      for(let j=-24;j<=24;j++){ const q=(mid+j+N)%N; tr[q]=Math.min(hi[q],Math.max(lo[q],tr[q]+dir*step*Math.exp(-(j*j)/(2*64)))); }
      const t=evalLap(tr);
      if(t<bestT-0.003){ gained+=bestT-t; bestT=t; off=tr; log('refine r'+r,'corner at e',model.lap(off,true).x[c].toFixed(1),(sh<0?'entry':sh>0?'exit':'apex'),(dir>0?'left':'right'),step.toFixed(2),'m ->',t.toFixed(3),'s'); }
      await yieldNow(); }
    log('refine round',r,'gained',gained.toFixed(3),'s'); if(gained<0.01) break; }
  return {off,t:bestT};
}

// ---------- speed refinement on the real physics: carry a bit more / less speed through each 20 m of the lap, keep what is faster and still clean ----------
async function refineSpeed(path,REF,sc0,KM,PADC,rounds,startT){
  const N=sc0.length; let sc=Float64Array.from(sc0);
  const evalSc=s=>{ const d=makeDriver(path,KM,PADC,REF); d.sc.set(s); const r=d.lapRun(false,0,2); return {t:(r.ok&&!r.bad.some(b=>b))?r.times[r.times.length-1]:Infinity,r}; };
  let cur=evalSc(sc), bestT=isFinite(cur.t)?cur.t:(startT||1e9), bestR=cur.r; log('speed refine start',bestT.toFixed(3),'s');
  for(let r=0;r<rounds;r++){ const st=0.018/(r+1); let gained=0;
    for(let seg=0;seg<N;seg+=26) for(const dir of [1,-1]){
      const tr=Float64Array.from(sc); for(let j=-20;j<=20;j++){ const q=(seg+j+N)%N; tr[q]=Math.min(1.35,Math.max(0.6,tr[q]*(1+dir*st*Math.exp(-(j*j)/(2*100))))); }
      const e=evalSc(tr); if(e.t<bestT-0.002){ gained+=bestT-e.t; bestT=e.t; sc=tr; bestR=e.r; }
      await yieldNow(); }
    log('speed refine round',r,'gained',gained.toFixed(3),'s ->',bestT.toFixed(3),'s'); if(gained<0.01) break; }
  return {sc,t:bestT,rec:bestR.rec,traj:bestR.traj};
}

async function run(opts={}){
  window.T4OPT.logs=[];
  const t4=T(); if(!t4.ready){ t4.driveStart(true); for(let i=0;i<60&&!t4.ready;i++) await sleep(500); }
  // (no camera switch: each AI run saves / restores the kart state itself, and rAF may be paused when the pane is hidden)     // back to free camera so the render loop leaves the kart alone (each AI run saves / restores the kart state anyway)
  const KM=t4.KM, SURFP=t4.SURFP, PADC=t4.PADC;
  const ref=resampleClosed(t4.LINE,DS); let P=ref.pts; const s0=P.reduce((b,p,i)=>Math.hypot(p[0]-13.8,p[1]+17.6)<Math.hypot(P[b][0]-13.8,P[b][1]+17.6)?i:b,0);
  P=P.slice(s0).concat(P.slice(0,s0));                                    // lap starts at the start / finish line
  const F=frames(P); log('reference line',P.length,'points,',ref.L.toFixed(1),'m');
  const B=await trackBounds(P,F); log('track width median',(()=>{ const w=Array.from(B.bl).map((v,i)=>v+B.br[i]).sort((a,b)=>a-b); return w[w.length>>1].toFixed(2); })(),'m');
  const REF={P,F,B}, model=makeModel(P,F,B,KM,SURFP);
  // 1) line by lap-time minimisation
  let opt=await optimise(model,B,null,opts.iters||300,'line');
  let path=model.lap(opt.off,true), offUsed=opt.off; log('model lap',path.t.toFixed(3),'s');
  // 2) AI learns the speeds on the real physics
  let drv=makeDriver(path,KM,PADC,REF), best=await learnLaps(drv,0,opts.laps||40,'AI',path);
  if(!best){ log('no clean lap - stopping'); return null; }
  // 3) real-physics line refinement, corner by corner, then a few more learning laps on the final line
  const rf=await refine(model,B,REF,opt.off,best.sc,KM,PADC,opts.rounds||3,best.t);
  if(rf.t<best.t){ path=model.lap(rf.off,true); offUsed=rf.off; drv=makeDriver(path,KM,PADC,REF); drv.sc.set(best.sc); const b2=await learnLaps(drv,0,opts.laps2||15,'AI final',path); if(b2&&b2.t<rf.t) best=b2; else { const rr=drv.lapRun(false,0,2); best={...best,t:rf.t,sc:best.sc,rec:rr.rec,traj:rr.traj}; } opt={off:rf.off}; }
  { const rs=await refineSpeed(path,REF,best.sc,KM,PADC,opts.srounds||3,best.t); if(rs.t<best.t&&rs.traj) best={...best,t:rs.t,sc:rs.sc,rec:rs.rec,traj:rs.traj}; }
  // reference: the same AI on the old line, same learning budget
  let bestRef=null; if(!opts.skipRef){ const refPath=model.lap(new Float64Array(P.length),true), drvRef=makeDriver(refPath,KM,PADC,REF); bestRef=await learnLaps(drvRef,0,opts.laps||40,'AI on old line'); }
  const out={note:'fastest-lap AI (lineopt.js): lap-time-minimised line, refined and driven by an AI on the real kart physics (trail braking via the friction ellipse)',
    lapTime:+best.t.toFixed(3), modelTime:+path.t.toFixed(3), oldLineTime:bestRef?+bestRef.t.toFixed(3):null,
    line:path.x.map((x,i)=>[+x.toFixed(3),+path.z[i].toFixed(3)]), speed:Array.from(best.rec.v,v=>+v.toFixed(2)),
    throttle:Array.from(best.rec.thr,v=>+v.toFixed(2)), brake:Array.from(best.rec.brk,v=>+v.toFixed(2)), steer:Array.from(best.rec.steer,v=>+v.toFixed(2)),
    scale:Array.from(best.sc,v=>+v.toFixed(3)), targetSpeed:path.v.map((v,i)=>+(v*best.sc[i]).toFixed(2)), klat:model.prm.klat,
    pathV:path.v.map(v=>+v.toFixed(3)), kap:path.kap.map(v=>+v.toFixed(5)), off:Array.from(offUsed,v=>+v.toFixed(3)),
    ghost:{dt:0.05,pts:best.traj||[]}, kart:{type:'Tillotson T4 Jr',m:KM.m,P:+KM.P.toFixed(0),Fdrive:+KM.Fdrive.toFixed(0),mu0:KM.mu0,vmax:+KM.vmax.toFixed(2),Tq:KM.Tq,gear:KM.gear}};
  await fetch('/savejson?f=bestline',{method:'POST',body:JSON.stringify(out)});
  log('saved data/bestline.json: AI lap',out.lapTime,'s (model',out.modelTime,'s; old line',out.oldLineTime,'s)');
  window.T4OPT.result=out; return out;
}
async function quick(over={},iters=90){
  const t4=T(); if(!t4.ready){ t4.driveStart(true); for(let i=0;i<60&&!t4.ready;i++) await sleep(500); }
  const KM=Object.assign({},t4.KM,over);
  if(!quick.cache){ const ref=resampleClosed(t4.LINE,DS); let P=ref.pts; const s0=P.reduce((b,p,i)=>Math.hypot(p[0]-13.8,p[1]+17.6)<Math.hypot(P[b][0]-13.8,P[b][1]+17.6)?i:b,0);
    P=P.slice(s0).concat(P.slice(0,s0)); const F=frames(P); const B=await trackBounds(P,F); quick.cache={P,F,B}; }
  const {P,F,B}=quick.cache, model=makeModel(P,F,B,KM,t4.SURFP); if(over.klat) model.prm.klat=over.klat;
  const o=await optimise(model,B,null,iters,'quick'); const p=model.lap(o.off,true); let vmax=0; for(const v of p.v) vmax=Math.max(vmax,v);
  return {t:+p.t.toFixed(3),vmax:+(vmax*3.6).toFixed(1)};
}
// ---------- live learning (owner: 'make the AI adaptive so it learns every lap') ----------
// While you drive, the AI keeps driving its own laps in the background on the same physics (a few ms per frame). Every AI
// lap is one experiment, always started from its best lap so far, and kept only if the lap is clean and faster:
//  - carry a little more / less speed through one stretch (where to brake, how much to carry)
//  - move its line a little at one corner (entry / apex / exit)
//  - your line: when you set a valid lap faster than its best, it drives your line and keeps it if that is faster for it too
// A faster lap becomes the ghost from your next lap. Progress is kept in this browser (localStorage).
const LIVE={on:false,st:null,gen:null,aiSnap:null,job:null};
const LKEY='t4_ai_live2';
function liveSig(d,KM){ return [d.lapTime,KM.m,KM.mu0,(KM.Tq||[]).map(q=>q[1]).join(',')].join('|'); }
const RUNUP=200;                                                              // ~150 m run-up, then one timed flying lap
async function liveStart(){
  if(LIVE.on) return; const t4=T();
  for(let k=0;k<120&&!(t4.LINE&&t4.LINE.length&&t4.GHOST);k++) await sleep(500);
  let d=null; try{ d=await (await fetch('data/bestline.json',{cache:'no-store'})).json(); }catch(e){ return; }
  if(!d||!d.pathV||!d.off||!d.scale){ log('live: bestline.json has no learning data - run T4OPT.run() first'); return; }
  while(!t4.ready) await sleep(500);                                        // walls + surfaces loaded (drive mode): judge laps exactly as they will be driven
  const KM=t4.KM, PADC=t4.PADC, ref=resampleClosed(t4.LINE,DS); let P=ref.pts;
  const s0=P.reduce((b,p,i)=>Math.hypot(p[0]-13.8,p[1]+17.6)<Math.hypot(P[b][0]-13.8,P[b][1]+17.6)?i:b,0); P=P.slice(s0).concat(P.slice(0,s0));
  if(P.length!==d.off.length){ log('live: line changed - re-run the optimiser'); return; }
  const F=frames(P), B=await trackBounds(P,F), REF={P,F,B}, model=makeModel(P,F,B,KM,t4.SURFP); model.prm.klat=d.klat||model.prm.klat;
  const N=P.length, marg=-0.45, lo=Float64Array.from(B.br,v=>-(v-marg)), hi=Float64Array.from(B.bl,v=>v-marg);
  const evalLap=(off,sc)=>{ const path=model.lap(Float64Array.from(off),true), drv=makeDriver(path,KM,PADC,REF); drv.sc.set(sc); const r=drv.lapRun(false,(N-RUNUP)%N,1);
    return {path,drv,r,ok:r.ok&&!r.bad.some(b=>b)&&r.times.length>0,t:r.times.length?r.times[r.times.length-1]:Infinity}; };
  const sig=liveSig(d,KM); let st=null; try{ st=JSON.parse(localStorage.getItem(LKEY)||'null'); }catch(e){}
  let cur=null;
  if(st&&st.sig===sig&&st.sc&&st.sc.length===N&&st.off&&st.off.length===N){ cur=evalLap(st.off,st.sc); if(!cur.ok) st=null; }
  if(!st||st.sig!==sig){ cur=evalLap(d.off,d.scale); st={sig,laps:0,kept:0,sc:d.scale.slice(),off:d.off.slice(),traj:null,from:'optimiser'}; }
  st.best=cur.ok?cur.t:(d.lapTime||cur.t); if(cur.ok&&cur.r.traj) st.traj=cur.r.traj;
  const kap=cur.path.kap, corners=[]; for(let i=0;i<N;i++){ const a=Math.abs(kap[i]); if(a<0.035) continue; let pk=true; for(let j=-14;j<=14;j++) if(Math.abs(kap[(i+j+N)%N])>a){ pk=false; break; } if(pk) corners.push(i); }
  LIVE.st={st,KM,PADC,REF,model,N,lo,hi,corners,ci:0,seenPlayer:null,path:cur.path};
  if(st.traj) setGhost(st.traj,st.best,false);
  LIVE.on=true; window.T4OPT.live=LIVE; log('live learning on: best',st.best.toFixed(3),'s after',st.laps,'AI laps ('+st.kept+' improvements kept)'); requestAnimationFrame(liveTick);
}
function setGhost(traj,lap,announce){ const G=T().GHOST; if(!G||!traj||!traj.length) return; G.pending={pts:traj,lap};
  if(announce&&T().flash) T().flash('AI learned a faster lap: '+lap.toFixed(3)); }
function liveSave(){ try{ const s=LIVE.st.st; localStorage.setItem(LKEY,JSON.stringify({...s,sc:Array.from(s.sc,v=>+(+v).toFixed(4)),off:Array.from(s.off,v=>+(+v).toFixed(3))})); }catch(e){} }
function nextJob(){ const L=LIVE.st, s=L.st, N=L.N, rnd=Math.random;
  const pl=T().PLAP&&T().PLAP.last;                                          // you beat it: drive your line
  if(pl&&pl!==L.seenPlayer&&pl.t<s.best){ L.seenPlayer=pl; const off=new Float64Array(N), P=L.REF.P, F=L.REF.F, pts=pl.pts, M=pts.length; let j=0;
    for(let i=0;i<N;i++){ let bd=1e18,bj=j; const k0=i===0?0:-30, k1=i===0?M:60; for(let k=k0;k<k1;k++){ const kk=((i===0?k:j+k)%M+M)%M, dx=pts[kk][0]-P[i][0], dz=pts[kk][1]-P[i][1], dd=dx*dx+dz*dz; if(dd<bd){ bd=dd; bj=kk; } } j=bj;
      off[i]=Math.min(L.hi[i],Math.max(L.lo[i],(pts[bj][0]-P[i][0])*F.nx[i]+(pts[bj][1]-P[i][1])*F.nz[i])); }
    const sm=new Float64Array(N); for(let i=0;i<N;i++){ let a=0; for(let k=-4;k<=4;k++) a+=off[(i+k+N)%N]; sm[i]=a/9; }
    return {kind:'your line',off:sm,sc:Float64Array.from(s.sc).map(v=>v*0.97),tries:4}; }  // start a bit gentler on a new line, then push
  if(rnd()<0.35&&L.corners.length){                                            // move the line at one corner
    const c=L.corners[L.ci%L.corners.length], ph=Math.floor(L.ci/L.corners.length)%6; L.ci++;
    const shift=[-32,0,22][ph%3], dir=ph<3?1:-1, step=(0.12+0.2*rnd())*dir*Math.sign(L.path.kap[c]||1), off=Float64Array.from(s.off);
    for(let k=-26;k<=26;k++){ const i=(c+shift+k+N)%N, w=0.5+0.5*Math.cos(Math.PI*k/26); off[i]=Math.min(L.hi[i],Math.max(L.lo[i],off[i]+step*w)); }
    return {kind:['entry','apex','exit'][ph%3]+' line at e '+L.path.x[c].toFixed(0),off,sc:Float64Array.from(s.sc),tries:1}; }
  const k=Math.floor(rnd()*N), wdt=12+Math.floor(rnd()*28), f=(rnd()<0.6?1:-1)*(0.006+0.022*rnd()), sc=Float64Array.from(s.sc);   // speed through one stretch
  for(let j=-wdt;j<=wdt;j++){ const i=(k+j+N)%N; sc[i]=Math.min(1.35,Math.max(0.6,sc[i]*(1+f*(0.5+0.5*Math.cos(Math.PI*j/wdt))))); }
  return {kind:(f>0?'more':'less')+' speed at e '+L.path.x[k].toFixed(0),off:null,sc,tries:1};
}
function startJob(J){ const L=LIVE.st, path=J.off?L.model.lap(J.off,true):L.path, drv=makeDriver(path,L.KM,L.PADC,L.REF); drv.sc.set(J.sc); J.path=path; J.drv=drv;
  LIVE.gen=drv.lapRunG(J.kind==='your line',(L.N-RUNUP)%L.N,1,true); LIVE.aiSnap=null; }
function liveTick(){
  if(!LIVE.on) return; const t4=T();
  if(t4.mode!=='drive'||!t4.GHOST.on||document.hidden||(window.T4RACE&&window.T4RACE.active)){ setTimeout(()=>requestAnimationFrame(liveTick),400); return; }   // paused while racing
  const L=LIVE.st, D=t4.DRV;
  if(!LIVE.gen){ LIVE.job=nextJob(); startJob(LIVE.job); }
  const user=JSON.stringify(D), sh=t4.shake; if(LIVE.aiSnap) Object.assign(D,JSON.parse(LIVE.aiSnap));
  const t0=performance.now(); let r={done:false};
  try{ while(performance.now()-t0<3.5){ r=LIVE.gen.next(); if(r.done) break; } }catch(e){ console.warn('[T4OPT] live',e); r={done:true,value:null}; }
  LIVE.aiSnap=r.done?null:JSON.stringify(D); Object.assign(D,JSON.parse(user)); t4.shake=sh;
  if(r.done){ LIVE.gen=null; const res=r.value, J=LIVE.job, s=L.st; s.laps++;
    const clean=res&&res.ok&&!res.bad.some(b=>b)&&res.times.length, t=clean?res.times[res.times.length-1]:Infinity;
    (LIVE.hist=LIVE.hist||[]).push([J.kind,res&&res.times.length?+res.times[res.times.length-1].toFixed(3):null,!!clean,res?res.bad.reduce((a,b)=>a+(b?1:0),0):-1]); if(LIVE.hist.length>40) LIVE.hist.shift();
    if(clean&&t<s.best-0.002&&res.traj&&res.traj.length){ const gain=s.best-t; s.best=t; s.traj=res.traj; s.from=J.kind; s.kept=(s.kept||0)+1;
      if(J.off){ s.off=Array.from(J.off); L.path=J.path; } s.sc=Array.from(J.drv.sc);
      setGhost(res.traj,t,true); log('live: AI lap',s.laps,J.kind,'->',t.toFixed(3),'s (-'+gain.toFixed(3)+')'); liveSave(); }
    else if(J.kind==='your line'&&--J.tries>0){ LIVE.gen=J.drv.lapRunG(true,(L.N-RUNUP)%L.N,1,true); LIVE.aiSnap=null; }   // a few learning laps on your line
    if(s.laps%20===0) liveSave(); }
  requestAnimationFrame(liveTick);
}
function liveProbe(n){ const L=LIVE.st, s=L.st, out=[]; for(let k=0;k<n;k++){ const J=nextJob(), path=J.off?L.model.lap(J.off,true):L.path, drv=makeDriver(path,L.KM,L.PADC,L.REF); drv.sc.set(J.sc);
  const r=drv.lapRun(false,(L.N-RUNUP)%L.N,1), kinds={}; r.bad.forEach(b=>{ if(b) kinds[b]=(kinds[b]||0)+1; }); out.push({kind:J.kind,t:r.times.length?+r.times[r.times.length-1].toFixed(3):null,kinds}); } return {best:+s.best.toFixed(3),out}; }   // debug: what the experiments produce
function liveCompare(){ const L=LIVE.st, s=L.st, mk=()=>{ const d=makeDriver(L.path,L.KM,L.PADC,L.REF); d.sc.set(s.sc); return d; };
  const a=mk().lapRun(false,(L.N-RUNUP)%L.N,1); const g=mk().lapRunG(false,(L.N-RUNUP)%L.N,1,true); let r; do{ r=g.next(); }while(!r.done);
  const D=T().DRV, g2=mk().lapRunG(false,(L.N-RUNUP)%L.N,1,true); let snap=null, r2; do{ const u=JSON.stringify(D); if(snap) Object.assign(D,JSON.parse(snap)); r2=g2.next(); snap=r2.done?null:JSON.stringify(D); Object.assign(D,JSON.parse(u)); }while(!r2.done);
  return {sync:a.times, chunked:r.value.times, swapped:r2.value.times}; }
window.T4OPT={run,quick,liveStart,live:LIVE,liveProbe,liveCompare};
})();

# EXPERIMENTAL / SUPERSEDED — not part of the morphompm package. Taichi autodiff
# segfaults on matrix-MPM adjoints (see docs/PIPELINE.md); superseded by the numpy
# package (morphompm.integrate/diff). Kept for the record; do NOT build on this.

import time, sys
import numpy as np
import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f64)

steps = 24
N, dx = 20, 0.05
inv_dx = 1.0/dx
E, nu = 1.0e4, 0.3
mu0 = E/(2*(1+nu)); lam0 = E*nu/((1+nu)*(1-2*nu))
dt, damping, density = 1e-3, 0.05, 1000.0
center, half, sp = 0.5, 0.05, 0.025
coords = np.arange(center-half, center+half+1e-9, sp)
pts = np.array([(a,b,c) for a in coords for b in coords for c in coords], dtype=np.float64)
n = len(pts); pmass = density*sp**3; vol0 = pmass/density

x=ti.Vector.field(3,ti.f64,(steps,n),needs_grad=True)
v=ti.Vector.field(3,ti.f64,(steps,n),needs_grad=True)
C=ti.Matrix.field(3,3,ti.f64,(steps,n),needs_grad=True)
F=ti.Matrix.field(3,3,ti.f64,(steps,n),needs_grad=True)
grid_v=ti.Vector.field(3,ti.f64,(steps,N,N,N),needs_grad=True)
grid_m=ti.field(ti.f64,(steps,N,N,N),needs_grad=True)
g=ti.field(ti.f64,(),needs_grad=True)
loss=ti.field(ti.f64,(),needs_grad=True)
x0=ti.Vector.field(3,ti.f64,n)

@ti.kernel
def set_initial():
    for p in range(n):
        x[0,p]=x0[p]; v[0,p]=ti.Vector([0.,0.,0.]); C[0,p]=ti.Matrix.zero(ti.f64,3,3); F[0,p]=ti.Matrix.identity(ti.f64,3)
@ti.kernel
def clear_grid():
    for f,i,j,k in grid_m:
        grid_m[f,i,j,k]=0.0; grid_v[f,i,j,k]=ti.Vector([0.,0.,0.])
@ti.kernel
def p2g(f:ti.i32):
    for p in range(n):
        gv=g[None]; volg=vol0*gv*gv*gv
        Xp=x[f,p]*inv_dx; base=ti.floor(Xp-0.5).cast(ti.i32); fx=Xp-base.cast(ti.f64)
        w=[0.5*(1.5-fx)**2,0.75-(fx-1.0)**2,0.5*(fx-0.5)**2]
        Fe=F[f,p]*(1.0/gv); J=Fe.determinant(); I3=ti.Matrix.identity(ti.f64,3)
        tau=mu0*(Fe@Fe.transpose()-I3)+lam0*ti.log(J)*I3; sk=tau*(-4.0*inv_dx*inv_dx*volg)
        for i,j,k in ti.static(ti.ndrange(3,3,3)):
            offs=ti.Vector([i,j,k]); dpos=(offs.cast(ti.f64)-fx)*dx; weight=w[i][0]*w[j][1]*w[k][2]; idx=base+offs
            grid_m[f,idx[0],idx[1],idx[2]]+=weight*pmass
            grid_v[f,idx[0],idx[1],idx[2]]+=weight*(pmass*(v[f,p]+C[f,p]@dpos)+(sk@dpos)*dt)
@ti.kernel
def grid_op(f:ti.i32):
    for i,j,k in ti.ndrange(N,N,N):
        m=grid_m[f,i,j,k]
        if m>1e-12: grid_v[f,i,j,k]=grid_v[f,i,j,k]*((1.0-damping)/m)
@ti.kernel
def g2p(f:ti.i32):
    for p in range(n):
        Xp=x[f,p]*inv_dx; base=ti.floor(Xp-0.5).cast(ti.i32); fx=Xp-base.cast(ti.f64)
        w=[0.5*(1.5-fx)**2,0.75-(fx-1.0)**2,0.5*(fx-0.5)**2]
        nv=ti.Vector.zero(ti.f64,3); nC=ti.Matrix.zero(ti.f64,3,3)
        for i,j,k in ti.static(ti.ndrange(3,3,3)):
            offs=ti.Vector([i,j,k]); dpos=(offs.cast(ti.f64)-fx)*dx; weight=w[i][0]*w[j][1]*w[k][2]; idx=base+offs
            gvel=grid_v[f,idx[0],idx[1],idx[2]]; nv+=weight*gvel; nC+=4.0*inv_dx*inv_dx*weight*gvel.outer_product(dpos)
        v[f+1,p]=nv; C[f+1,p]=nC; x[f+1,p]=x[f,p]+dt*nv; F[f+1,p]=(ti.Matrix.identity(ti.f64,3)+dt*nC)@F[f,p]
@ti.kernel
def compute_loss():
    for p in range(n):
        loss[None]+=(1.0/n)*F[steps-1,p].determinant()

def forward():
    clear_grid(); set_initial()
    for f in range(steps-1):
        p2g(f); grid_op(f); g2p(f)

x0.from_numpy(pts); g[None]=1.3
def t(msg, fn):
    a=time.time(); r=fn(); ti.sync(); print(f"{msg}: {time.time()-a:.2f}s", flush=True); return r
print(f"n={n}, steps={steps}", flush=True)
t("forward #1 (compile+run)", lambda: forward())
t("forward #2 (run only)", lambda: forward())
def tape_run():
    loss[None]=0.0
    with ti.ad.Tape(loss):
        forward(); compute_loss()
t("tape #1 (compile backward+run)", tape_run)
t("tape #2 (run only)", tape_run)
print("grad", g.grad[None], flush=True)

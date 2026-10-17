#!/usr/bin/env python3
"""3-D linear acoustics on a Cartesian staggered grid, GPU via PyTorch.
Representative local rectangular airgap; rigid x and y duct walls within channel, free air above/below,
absorbing sponge only on outer computational boundaries.
"""
import argparse, json, logging, math, os, sys, time
from contextlib import closing
from pathlib import Path
import numpy as np
import torch


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0',help='cuda:0, cuda:1, or cpu')
    p.add_argument('--out',default='results/case40k')
    p.add_argument('--frequency',type=float,default=40000,help='Hz')
    p.add_argument('--tau-us',type=float,default=100)
    p.add_argument('--amplitude',type=float,default=0.001,help='m/s peak normal wall velocity')
    p.add_argument('--source-z',type=float,default=0.90,help='m from channel bottom (1.00 m from microphone at -0.10 m)')
    p.add_argument('--source-y',type=float,default=0.0)
    p.add_argument('--source-diameter',type=float,default=0.005)
    p.add_argument('--dx-mm',type=float,default=1.0)
    p.add_argument('--duration-ms',type=float,default=7.0)
    p.add_argument('--frame-every',type=int,default=45)
    p.add_argument('--receiver-every',type=int,default=1)
    p.add_argument('--air-temp-c',type=float,default=20)
    p.add_argument('--channel-height',type=float,default=1.0)
    p.add_argument('--channel-width',type=float,default=0.020)
    p.add_argument('--y-halfspan',type=float,default=0.080,help='half width of outer air box (m)')
    p.add_argument('--channel-depth',type=float,default=0.100,help='distance between rigid duct sticks (m)')
    p.add_argument('--x-left',type=float,default=-0.035)
    p.add_argument('--x-right',type=float,default=0.055)
    p.add_argument('--z-bottom',type=float,default=-0.16)
    p.add_argument('--z-top-extra',type=float,default=0.10)
    p.add_argument('--mic-z',type=float,default=-0.10)
    p.add_argument('--sponge-mm',type=float,default=20)
    p.add_argument('--sponge-strength',type=float,default=2.5,help='damping exponent per one-way sponge traversal')
    p.add_argument('--no-frames',action='store_true')
    p.add_argument('--volume-stride',type=int,default=3,help='save full 3D pressure every Nth grid cell per axis (1 = full resolution)')
    return p.parse_args()


def main():
    a=parse(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    logger=logging.Logger(__name__)
    formatter=logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    with closing(logging.FileHandler(out/'output.log',mode='a',encoding='utf-8')) as file_handler:
        console_handler=logging.StreamHandler(sys.stdout)
        for handler in (console_handler,file_handler):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        try:
            logger.info('Starting simulation; output directory: %s',out.resolve())
            logger.info('Configuration: %s',json.dumps(vars(a),sort_keys=True))
            simulate(a,out,logger)
        except KeyboardInterrupt:
            logger.warning('Simulation interrupted; outputs may be incomplete')
            raise
        except Exception:
            logger.exception('Simulation failed; outputs may be incomplete')
            raise
        finally:
            for handler in (console_handler,file_handler):
                logger.removeHandler(handler)
            console_handler.close()


def simulate(a,out,logger):
    logger.info('Initializing device %s',a.device)
    torch.set_grad_enabled(False)
    device=torch.device(a.device)
    if device.type=='cuda': torch.cuda.set_device(device)
    if a.volume_stride < 1: raise ValueError("--volume-stride must be >= 1")
    d=a.dx_mm*1e-3
    c=331.3*math.sqrt(1+a.air_temp_c/273.15)
    rho=1.204*(293.15/(273.15+a.air_temp_c))
    bulk=rho*c*c
    x=np.arange(a.x_left,a.x_right+d/2,d,dtype=np.float64)
    y=np.arange(-a.y_halfspan,a.y_halfspan+d/2,d,dtype=np.float64)
    z=np.arange(a.z_bottom,a.channel_height+a.z_top_extra+d/2,d,dtype=np.float64)
    nx,ny,nz=len(x),len(y),len(z)
    if not (0<a.channel_width<a.x_right and a.x_left<0): raise ValueError('x domain must enclose channel and exterior')
    if not (0<a.channel_depth/2<a.y_halfspan): raise ValueError('y domain must enclose duct walls and exterior')
    if not (a.z_bottom<a.mic_z<0): raise ValueError('microphone must be below channel and inside computational box')
    if a.mic_z-a.z_bottom <= a.sponge_mm*1e-3+2*d: raise ValueError('microphone too close to lower sponge: extend --z-bottom')
    if not (-a.channel_depth/2<a.source_y<a.channel_depth/2): raise ValueError('source-y must be between duct sticks')
    dt=0.52*d/(c*math.sqrt(3))
    nsteps=int(math.ceil(a.duration_ms*1e-3/dt))
    dt=a.duration_ms*1e-3/nsteps
    logger.info(f'Grid {nx} x {ny} x {nz} = {nx*ny*nz/1e6:.2f}M cells; dt={dt*1e6:.3f} us; {nsteps} steps; {device}')
    if a.source_z<0 or a.source_z>a.channel_height: raise ValueError('source-z must be inside channel')
    logger.info('Allocating pressure/velocity fields and preparing source, boundaries, and receivers')
    # cell-centered pressure; face-centered velocities
    p=torch.zeros((nx,ny,nz),device=device,dtype=torch.float32)
    vx=torch.zeros((nx+1,ny,nz),device=device,dtype=torch.float32)
    vy=torch.zeros((nx,ny+1,nz),device=device,dtype=torch.float32)
    vz=torch.zeros((nx,ny,nz+1),device=device,dtype=torch.float32)
    # Four rigid channel walls (epoxy radial faces and duct-stick circumferential faces).
    # Outside channel height, all x/y locations are free air. Air mask is truly 3D.
    xx=torch.as_tensor(x,device=device,dtype=torch.float32)
    yy=torch.as_tensor(y,device=device,dtype=torch.float32)
    zz=torch.as_tensor(z,device=device,dtype=torch.float32)
    in_z=(zz>=-d*0.25)&(zz<=a.channel_height+d*0.25)
    in_x=(xx>=-d*0.25)&(xx<=a.channel_width+d*0.25)
    in_y=(yy>=-a.channel_depth/2-d*0.25)&(yy<=a.channel_depth/2+d*0.25)
    air=(~in_z[None,None,:]) | (in_x[:,None,None]&in_y[None,:,None])
    # Adjacent air cells share a face; a face bordering solid is rigid (velocity=0).
    xface=air[1:,:,:]&air[:-1,:,:]
    yface=air[:,1:,:]&air[:,:-1,:]
    zface=air[:,:,1:]&air[:,:,:-1]
    # Source at x=0 wall face, source-z, source-y, circular patch in y-z.
    src_face=int(np.flatnonzero(x>=-d*0.25)[0]) # first fluid cell, left solid/fluid face
    if not (0<src_face<nx): raise RuntimeError('source face outside domain')
    Y,Z=torch.meshgrid(yy,zz,indexing='ij')
    patch=((Y-a.source_y)**2+(Z-a.source_z)**2 <= (a.source_diameter/2)**2) & in_y[:,None] & in_z[None,:]
    if not bool(patch.any()): raise ValueError('source patch too small for grid')
    if not (bool(air[src_face,:,:].logical_xor(air[src_face-1,:,:]).logical_and(patch).any())): raise ValueError('source face does not lie on fluid-solid interface')
    patch=patch.float()
    # Sponge on outer box edges; attenuation applied to pressure and velocities.
    # Damping uses physical length and c; no loss at epoxy walls.
    sp=a.sponge_mm*1e-3
    def edge_profile(v,lo,hi):
        return (torch.clamp((lo+sp-v)/sp,min=0,max=1)**2 + torch.clamp((v-(hi-sp))/sp,min=0,max=1)**2)
    # Sponge only at far edges of external air. Solid walls are lossless and rigid.
    sx=edge_profile(xx,x[0],x[-1])[:,None,None]
    sy=edge_profile(yy,y[0],y[-1])[None,:,None]
    sz=edge_profile(zz,z[0],z[-1])[None,None,:]
    sigma=(a.sponge_strength*c/sp)*(sx+sy+sz)
    damp=torch.exp(-sigma*dt).float()
    # velocity damping at faces from adjacent cell factors
    dampx=0.5*(damp[:-1]+damp[1:]); dampy=0.5*(damp[:,:-1]+damp[:,1:]); dampz=0.5*(damp[:,:,:-1]+damp[:,:,1:])
    i_mic=int(np.argmin(abs(x-a.channel_width/2)))
    j_mic=int(np.argmin(abs(y)))
    k_mic=int(np.argmin(abs(z-a.mic_z)))
    if abs(z[k_mic]-a.mic_z)>d or not bool(air[i_mic,j_mic,k_mic]): raise RuntimeError('mic outside physical air domain')
    k_src=int(np.argmin(abs(z-a.source_z)))
    # Sample pressure close to source but not on rigid wall (diagnostic only).
    i_ref=int(np.argmin(abs(x-min(a.channel_width/2,0.005))))
    traces=[]; frames=[]; volumes=[]; frame_t=[]
    source_on=4*a.tau_us*1e-6
    if device.type=='cuda': torch.cuda.synchronize(device)
    logger.info('Setup complete; starting time stepping (0/%d, 0%%)',nsteps)
    start=time.monotonic()
    last_report=start
    progress_steps=max(1,math.ceil(nsteps/10))
    for step in range(nsteps):
        t=(step+0.5)*dt
        # pressure gradients update velocities (face arrays)
        vx[1:-1].add_(p[1:]-p[:-1],alpha=-dt/(rho*d))
        vy[:,1:-1].add_(p[:,1:]-p[:,:-1],alpha=-dt/(rho*d))
        vz[:,:,1:-1].add_(p[:,:,1:]-p[:,:,:-1],alpha=-dt/(rho*d))
        vx[1:-1].mul_(xface)
        vy[:,1:-1].mul_(yface)
        vz[:,:,1:-1].mul_(zface)
        # Source: wall velocity on face; ramp onset slightly to limit numerical high-frequency shock
        if t<source_on:
            onset=1-math.exp(-t/(2*dt))
            v=a.amplitude*math.exp(-t/(a.tau_us*1e-6))*math.cos(2*math.pi*a.frequency*t)*onset
        else: v=0.0
        # Impose velocity only on source patch at the x=0 fluid/solid face.
        # Positive vx points from the left epoxy wall into the air channel.
        vx[src_face].add_(patch*v) # after masked rigid-wall update; zero elsewhere on face
        # external boundary normal velocities remain zero in sponge; outer reflections attenuated
        vx[1:-1].mul_(dampx)
        vy[:,1:-1].mul_(dampy)
        vz[:,:,1:-1].mul_(dampz)
        # Pressure continuity equation
        div=(vx[1:]-vx[:-1]+vy[:,1:]-vy[:,:-1]+vz[:,:,1:]-vz[:,:,:-1])
        p.add_(div,alpha=-bulk*dt/d)
        p.mul_(air)
        p.mul_(damp)
        if step%a.receiver_every==0:
            traces.append((step*dt,float(p[i_mic,j_mic,k_mic].item()),float(p[i_ref,j_mic,k_src].item()),v))
        if not a.no_frames and step%a.frame_every==0:
            # longitudinal x-z plane and y-z plane through channel mid-gap
            frames.append((p[:,j_mic,:].detach().cpu().numpy().T.copy(),p[i_mic,:,:].detach().cpu().numpy().T.copy()))
            volumes.append(p[::a.volume_stride,::a.volume_stride,::a.volume_stride].detach().cpu().numpy().copy())
            frame_t.append(step*dt)
        completed=step+1
        now=time.monotonic()
        if completed%progress_steps==0 or now-last_report>=5 or completed==nsteps:
            if device.type=='cuda': torch.cuda.synchronize(device)
            now=time.monotonic()
            elapsed=now-start
            rate=completed/elapsed if elapsed>0 else 0.0
            eta=elapsed*(nsteps-completed)/completed
            logger.info('Progress %5.1f%% | step %d/%d | simulated %.3f/%.3f ms | elapsed %.1fs | %.1f steps/s | ETA %.1fs',
                        100*completed/nsteps,completed,nsteps,completed*dt*1e3,a.duration_ms,elapsed,rate,eta)
            last_report=now
    arr=np.asarray(traces,dtype=np.float64)
    logger.info('Saving signals.npz (%d receiver samples)',len(traces))
    np.savez_compressed(out/'signals.npz',t=arr[:,0],p_mic=arr[:,1],p_reference=arr[:,2],v_source=arr[:,3])
    if frames:
        logger.info('Saving frames.npz (%d frames); compression may take a while',len(frames))
        np.savez_compressed(out/'frames.npz',xz=np.stack([f[0] for f in frames]),yz=np.stack([f[1] for f in frames]),volume=np.stack(volumes),volume_stride=a.volume_stride,times=np.asarray(frame_t),x=x,y=y,z=z)
    else:
        logger.info('No frames to save')
    # metrics on post-arrival window
    travel=(a.source_z-a.mic_z)/c
    window=arr[:,0]>=max(0,travel-0.0003)
    peak=float(np.max(np.abs(arr[window,1]))) if np.any(window) else float('nan')
    ref_peak=float(np.max(np.abs(arr[:,2])))
    mic_rms=float(np.sqrt(np.mean(arr[window,1]**2))) if np.any(window) else float('nan')
    ref_rms=float(np.sqrt(np.mean(arr[:,2]**2)))
    relative_peak_db=float(20*np.log10(peak/ref_peak)) if peak>0 and ref_peak>0 else None
    logger.info('Receiver: peak %.4g Pa, windowed RMS %.4g Pa; reference peak %.4g Pa; relative peak %.2f dB',peak,mic_rms,ref_peak,relative_peak_db if relative_peak_db is not None else float('nan'))
    meta=dict(microphone_rms_pressure_pa=mic_rms,reference_peak_pressure_pa=ref_peak,reference_rms_pressure_pa=ref_rms,relative_peak_db=relative_peak_db,boundary_conditions={'epoxy_walls':'rigid','duct_sticks':'rigid','open_ends':'exterior_air_with_outer_sponge'},channel_depth_m=a.channel_depth,frequency_hz=a.frequency,amplitude_m_per_s=a.amplitude,source_z_m=a.source_z,source_y_m=a.source_y,source_diameter_m=a.source_diameter,source_tau_us=a.tau_us,grid_spacing_m=d,volume_stride=a.volume_stride,grid_shape=[nx,ny,nz],time_step_s=dt,nsteps=nsteps,duration_s=a.duration_ms*1e-3,temperature_c=a.air_temp_c,speed_sound_m_per_s=c,air_density_kg_m3=rho,channel_height_m=a.channel_height,channel_width_m=a.channel_width,y_halfspan_m=a.y_halfspan,mic_xyz_m=[float(x[i_mic]),float(y[j_mic]),float(z[k_mic])],nominal_arrival_s=travel,peak_microphone_pressure_pa=peak,run_seconds=time.monotonic()-start,limitations=['Rigid walls, except prescribed wall-velocity source','Four rigid channel walls; outer computational boundaries use sponge','Sponge boundaries are approximate, not a validated PML','No atmospheric absorption, wall viscothermal loss, or mean airflow','Source velocity assumed, not calibrated to PD apparent charge','No frequency-dependent microphone transfer function'])
    logger.info('Saving metadata.json')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    logger.info(f'Done: {out}; peak mic {peak:.4g} Pa; walltime {meta["run_seconds"]:.1f}s')

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Produce independent pressure waveforms and oriented 2D acoustic animation."""
import argparse, importlib.util, json, math, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib import colors


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--amplitude-scale',type=float,default=1.0,help='Linear pressure and source scaling for sensitivity study')
    parser.add_argument('--fps',type=int,default=20)
    parser.add_argument('--format',choices=['mp4','gif'],default='mp4')
    parser.add_argument('--max-frames',type=int,default=140)
    parser.add_argument('--view',choices=['isometric','slices'],default='slices')
    parser.add_argument('--volume-percentile',type=float,default=93.,help='absolute pressure percentile above which 3D points are shown')
    parser.add_argument('--max-points',type=int,default=14000,help='maximum pressure voxels shown in each 3D frame')
    a=parser.parse_args()
    if a.amplitude_scale<=0:parser.error('amplitude-scale must be positive')
    if a.fps<1 or a.max_frames<1 or a.max_points<1:parser.error('fps, max-frames, max-points must be positive')
    if not 0<a.volume_percentile<100:parser.error('volume-percentile must be in (0,100)')
    root=Path(a.input);m=json.loads((root/'metadata.json').read_text())
    with np.load(root/'signals.npz') as signals:
        t=signals['t']*1e3
        source=signals['v_source']*1e3*a.amplitude_scale # mm/s, imposed wall velocity
        mic=signals['p_mic']*1e3*a.amplitude_scale # mPa, received pressure
    fig,ax=plt.subplots(figsize=(10,3.8),constrained_layout=True)
    ax2=ax.twinx()
    l1=ax.plot(t,source,color='#cf623c',lw=1.0,label='Prescribed source wall velocity')[0]
    l2=ax2.plot(t,mic,color='#2176ad',lw=1.0,label='Microphone pressure')[0]
    l3=ax.axvline(m['nominal_arrival_s']*1e3,color='#555',ls='--',lw=1,label='Nominal arrival')
    ax.set_xlabel('Time (ms)');ax.set_ylabel('Source wall velocity (mm/s)',color='#b24b30')
    ax2.set_ylabel('Microphone pressure (mPa)',color='#2176ad')
    # Both axes must have zero at precisely the same vertical position.
    # Symmetric limits preserve the real pressure amplitudes without normalization.
    max_source=max(float(np.max(np.abs(source))),1e-9)*1.08
    max_mic=max(float(np.max(np.abs(mic))),1e-9)*1.08
    ax.set_ylim(-max_source,max_source);ax2.set_ylim(-max_mic,max_mic)
    ax.axhline(0,color='#555',lw=.65,alpha=.65,zorder=0)
    ax.tick_params(axis='y',labelcolor='#b24b30');ax2.tick_params(axis='y',labelcolor='#2176ad')
    ax.grid(alpha=.15);ax.set_xlim(t[0],t[-1])
    ax.legend([l1,l2,l3],[q.get_label() for q in [l1,l2,l3]],loc='upper right',fontsize=9,frameon=False)
    ax.set_title('PD source excitation and microphone detection')
    fig.savefig(root/'waveforms.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3),constrained_layout=True)
    ax.plot(t,mic,lw=1,color='#2176ad');ax.axvline(m['nominal_arrival_s']*1e3,color='gray',ls='--')
    ax.set_xlim(max(0,m['nominal_arrival_s']*1e3-.4),min(t[-1],m['nominal_arrival_s']*1e3+1.5))
    ax.set_xlabel('Time (ms)');ax.set_ylabel('Microphone pressure (mPa)');ax.grid(alpha=.15)
    fig.savefig(root/'microphone_zoom.png',dpi=180);plt.close(fig)
    path=root/'frames.npz'
    if not path.exists():
        print('Saved waveforms.png; no frames.npz found')
        return
    with np.load(path) as d:
        keys=set(d.files)
        x,y,z,times=[d[k].copy() for k in ('x','y','z','times')]
        vol=d['volume'].copy()*a.amplitude_scale if 'volume' in keys else None
        stride=int(d['volume_stride']) if 'volume_stride' in keys else 1
        xz=d['xz'].copy()*a.amplitude_scale if 'xz' in keys else None
        yz=d['yz'].copy()*a.amplitude_scale if 'yz' in keys else None
    if len(times)==0:parser.error('No saved animation frames')
    if a.view=='isometric' and vol is None:
        parser.error('3D animation requires volume data. Existing frames.npz only contains 2D slices. Rerun pd-acoustics-simulate with this updated version to save full 3D pressure.')
    if a.format=='mp4' and not FFMpegWriter.isAvailable():
        if importlib.util.find_spec('imageio_ffmpeg'):
            import imageio_ffmpeg
            matplotlib.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe()
        if not FFMpegWriter.isAvailable():parser.error('MP4 needs ffmpeg; install imageio-ffmpeg or use GIF')
    ids=np.unique(np.linspace(0,len(times)-1,min(len(times),a.max_frames),dtype=int))
    if a.view=='slices':
        if xz is None or yz is None:parser.error('2D slices unavailable')
        peak=max(float(np.percentile(np.abs(np.concatenate((xz[ids].ravel(),yz[ids].ravel()))),99.8)),1e-7)
        # Explicit axes placement eliminates the unused canvas to the right of the colorbar.
        # The orientation key is exported separately, never overlaid on the movie.
        fig=plt.figure(figsize=(9.2,5.25),constrained_layout=False)
        ax1=fig.add_axes([.085,.13,.315,.72])
        ax2=fig.add_axes([.49,.13,.385,.72])
        cax=fig.add_axes([.895,.22,.018,.53])
        im1=ax1.imshow(xz[ids[0]],origin='lower',aspect='auto',extent=[x[0]*1e3,x[-1]*1e3,z[0],z[-1]],cmap='RdBu_r',vmin=-peak,vmax=peak)
        im2=ax2.imshow(yz[ids[0]],origin='lower',aspect='auto',extent=[y[0]*1e3,y[-1]*1e3,z[0],z[-1]],cmap='RdBu_r',vmin=-peak,vmax=peak)
        ax1.set_title('Radial section (x–z) · between windings',fontsize=10)
        ax2.set_title('Circumferential section (y–z) · between duct sticks',fontsize=10)
        ax1.set_xlabel('Radial x (mm)');ax2.set_xlabel('Circumferential y (mm)')
        ax1.set_ylabel('Height above channel bottom (m)')
        ax2.set_ylabel('Height above channel bottom (m)')
        fig.colorbar(im1,cax=cax,label='Pressure (Pa)')
        # Mark the actual microphone sampling point in BOTH orthogonal slices.
        # It sits below the duct outlet (z < 0), so retain the external-air
        # portion of the axes; never clip the marker to the duct rectangle.
        mic_x, mic_y, mic_z = m['mic_xyz_m']
        for axis, horizontal in ((ax1, mic_x*1e3), (ax2, mic_y*1e3)):
            axis.scatter([horizontal], [mic_z], s=85, marker='o',
                         facecolor='#16a34a', edgecolor='white', linewidth=1.7,
                         zorder=12, clip_on=False)
            axis.annotate('Mic', (horizontal, mic_z), xytext=(8, 0),
                          textcoords='offset points', va='center', ha='left',
                          fontsize=9, fontweight='bold', color='#166534',
                          zorder=13, clip_on=False)
        # A separate, high-resolution top-view key for independent slide placement.
        from matplotlib.patches import Rectangle
        width_mm=m.get('channel_width_m',.02)*1e3
        depth_mm=m.get('channel_depth_m',.1)*1e3
        keyfig,key=plt.subplots(figsize=(3.3,4.1),constrained_layout=True)
        key.set_aspect('equal')
        key.add_patch(Rectangle((0,0),width_mm,depth_mm,fill=False,ec='#475569',lw=2))
        key.plot([0,width_mm],[depth_mm/2]*2,color='#cf623c',lw=3)
        key.plot([width_mm/2]*2,[0,depth_mm],color='#2176ad',lw=3)
        key.set_xlim(-width_mm*.5,width_mm*1.9)
        key.set_ylim(-depth_mm*.12,depth_mm*1.12)
        key.set_xticks([]);key.set_yticks([])
        key.set_title('Top view: pressure slice orientation',fontsize=10)
        key.text(width_mm*1.10,depth_mm*.52,'x–z',color='#cf623c',fontsize=10,va='center')
        key.text(width_mm*.55,depth_mm*.91,'y–z',color='#2176ad',fontsize=10)
        for spine in key.spines.values():spine.set_visible(False)
        keyfig.savefig(root/'slice_orientation_key.png',dpi=220,bbox_inches='tight',pad_inches=.10)
        plt.close(keyfig)
        title=fig.suptitle('',fontsize=13,y=.97)
        def update(j):
            i=ids[j];im1.set_data(xz[i]);im2.set_data(yz[i]);title.set_text(f'PD acoustic propagation | {times[i]*1e3:.2f} ms | {m["frequency_hz"]/1000:g} kHz')
            return im1,im2,title
    else:
        # Render 3D pressure voxels inside the physical rectangular air duct.
        # Coordinates use a common mm scale for truthful geometry; box aspect
        # compresses the long 1 m dimension for slide readability.
        xx=x[::stride]*1e3; yy=y[::stride]*1e3; zz=z[::stride]*1e3
        X,Y,Z=np.meshgrid(xx,yy,zz,indexing='ij')
        width=m['channel_width_m']*1e3; depth=m['channel_depth_m']*1e3; height=m['channel_height_m']*1e3
        mask=(X>=0)&(X<=width)&(Y>=-depth/2)&(Y<=depth/2)&(Z>=0)&(Z<=height)
        fig=plt.figure(figsize=(7,7),facecolor='white')
        ax=fig.add_axes([.04,.04,.84,.90],projection='3d',computed_zorder=False)
        ax.view_init(elev=19,azim=-62)
        ax.set_xlim(-width*.4,width*1.4);ax.set_ylim(-depth*.7,depth*.7);ax.set_zlim(-height*.08,height*1.06)
        ax.set_box_aspect((1.0,2.3,4.5),zoom=.91);ax.set_axis_off()
        for px in (0,width):
            for py in (-depth/2,depth/2):ax.plot([px,px],[py,py],[0,height],color='#637d91',alpha=.65,lw=1)
        for pz in (0,height):
            for py in (-depth/2,depth/2):ax.plot([0,width],[py,py],[pz,pz],color='#637d91',alpha=.65,lw=1)
            for px in (0,width):ax.plot([px,px],[-depth/2,depth/2],[pz,pz],color='#637d91',alpha=.65,lw=1)
        src=(0,m.get('source_y_m',0)*1e3,m['source_z_m']*1e3)
        receiver=np.asarray(m['mic_xyz_m'])*1e3
        ax.scatter(*([src[0]],[src[1]],[src[2]]),s=40,c='#db593c',depthshade=False)
        ax.scatter(*([receiver[0]],[receiver[1]],[receiver[2]]),s=40,c='#15935c',depthshade=False)
        ax.text(src[0],src[1],src[2]+height*.025,'PD',color='#ad3923',fontsize=11)
        ax.text(receiver[0],receiver[1],receiver[2]-height*.025,'Mic',color='#0d784b',fontsize=11)
        peak=max(float(np.percentile(np.abs(vol[ids][:,mask]),99.8)),1e-7)
        cmap=matplotlib.colormaps['RdBu_r']; norm=colors.Normalize(vmin=-peak,vmax=peak)
        sm=plt.cm.ScalarMappable(norm=norm,cmap=cmap);sm.set_array([])
        fig.colorbar(sm,ax=ax,shrink=.47,pad=.01,label='Acoustic pressure (Pa)')
        title=fig.suptitle('',fontsize=13,y=.97)
        scatter=[None]
        def update(j):
            i=ids[j]
            if scatter[0] is not None:scatter[0].remove()
            field=vol[i];absval=np.abs(field[mask]);threshold=max(float(np.percentile(absval,a.volume_percentile)),peak*.045)
            candidates=np.flatnonzero(mask & (np.abs(field)>=threshold))
            if len(candidates)>a.max_points:
                candidates=candidates[np.linspace(0,len(candidates)-1,a.max_points,dtype=int)]
            px=X.flat[candidates];py=Y.flat[candidates];pz=Z.flat[candidates];values=field.flat[candidates]
            scatter[0]=ax.scatter(px,py,pz,c=cmap(norm(values)),s=5,alpha=.45,depthshade=False,rasterized=True)
            title.set_text(f'3D acoustic pressure | {times[i]*1e3:.2f} ms | {m["frequency_hz"]/1000:g} kHz')
            return scatter[0],title
    update(0)
    fig.savefig(root/'pressure_isometric_preview.png' if a.view=='isometric' else root/'pressure_slices_preview.png',dpi=160)
    anim=FuncAnimation(fig,update,frames=len(ids),interval=1000/a.fps,blit=False)
    output=root/f'pressure_animation.{a.format}'
    writer=FFMpegWriter(fps=a.fps,bitrate=2400) if a.format=='mp4' else PillowWriter(fps=a.fps)
    print(f'Rendering {len(ids)} acoustic animation frames to {output}',flush=True)
    anim.save(output,writer=writer,dpi=105 if a.format=='mp4' else 85)
    plt.close(fig)
    print(f'Saved: {output}, waveforms.png, microphone_zoom.png',flush=True)

if __name__=='__main__':main()

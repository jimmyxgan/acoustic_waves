"""Illustrative 40-kHz microphone threshold from a linear reference FDTD run.

The 50 uPa/sqrt(Hz) Eta250 figure is specified at 1 kHz, not 40 kHz.
This is a sensitivity scenario, NOT a validated Eta250 40-kHz detection limit.
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--noise-asd-upa',type=float,default=50.,help='ASSUMED 40-kHz pressure noise density (uPa/sqrt Hz); manufacturer 50 value is at 1 kHz only')
    p.add_argument('--bandwidth-hz',type=float,default=1000.,help='Equivalent noise bandwidth of detection processing')
    p.add_argument('--snr',type=float,default=5.,help='Required peak received pressure / RMS noise')
    p.add_argument('--peak-window-ms',type=float,default=2.,help='Post-arrival window for peak detection')
    a=p.parse_args()
    if min(a.noise_asd_upa,a.bandwidth_hz,a.snr,a.peak_window_ms)<=0:p.error('positive noise, bandwidth, SNR and window required')
    root=Path(a.input); meta=json.loads((root/'metadata.json').read_text())
    with np.load(root/'signals.npz') as d:t=d['t'];mic=d['p_mic'];source=d['v_source']
    arrival=meta['nominal_arrival_s']
    win=(t>=arrival)&(t<=arrival+a.peak_window_ms*1e-3)
    if not win.any():p.error('simulation does not include post-arrival samples')
    p_peak=float(np.max(np.abs(mic[win])))
    if p_peak<=0:raise RuntimeError('Zero received reference pressure')
    noise=a.noise_asd_upa*1e-6*math.sqrt(a.bandwidth_hz)
    threshold=a.snr*noise
    scale=threshold/p_peak
    v_reference=float(meta['amplitude_m_per_s'])
    v_min=v_reference*scale
    report=dict(status='illustrative conditional estimate, not experimentally validated',distance_m=meta['source_z_m']-meta['mic_xyz_m'][2],frequency_hz=meta['frequency_hz'],reference_source_peak_velocity_m_per_s=v_reference,reference_mic_peak_pa=p_peak,assumed_noise_asd_pa_per_sqrt_hz=a.noise_asd_upa*1e-6,assumed_noise_bandwidth_hz=a.bandwidth_hz,assumed_mic_noise_rms_pa=noise,required_peak_to_rms_snr=a.snr,required_mic_peak_pa=threshold,amplitude_scale=scale,minimum_source_peak_velocity_m_per_s=v_min,minimum_source_peak_velocity_mm_per_s=v_min*1e3,notes=['Noise ASD of 50 uPa/sqrt(Hz) is published at 1 kHz, not established at 40 kHz','No source wall velocity versus PD apparent charge calibration','No microphone frequency response, electronic filtering, or false alarm characterization','Peak-to-RMS detection criterion; not a statistical probability-of-detection threshold','Linear pressure scaling assumes rigid lossless walls and linear air acoustics'])
    (root/'sensitivity.json').write_text(json.dumps(report,indent=2))
    fig,ax=plt.subplots(figsize=(10,3.7),constrained_layout=True)
    ax.plot(t*1e3,mic*scale*1e3,color='#2176ad',lw=.9,label='Microphone at illustrative detection threshold')
    ax.axhline(threshold*1e3,color='#d07b36',ls='--',lw=1,label=f'±{a.snr:g} × assumed RMS noise')
    ax.axhline(-threshold*1e3,color='#d07b36',ls='--',lw=1)
    ax.axvline(arrival*1e3,color='gray',ls=':',lw=1,label='Nominal arrival')
    ax.set(xlabel='Time (ms)',ylabel='Microphone pressure (mPa)',title='40 kHz received waveform at illustrative detection threshold')
    ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8)
    fig.savefig(root/'sensitivity_threshold_waveform.png',dpi=200);plt.close(fig)
    print(json.dumps(report,indent=2))
    print('To render the threshold-scaled animation and source/microphone plot:')
    print(f'pd-acoustics-visualize --input "{root}" --view slices --format both --amplitude-scale {scale:.12g}')

if __name__=='__main__':main()

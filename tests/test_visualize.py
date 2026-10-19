"""Visualization integration tests for independent outputs and volumetric frames."""
import json, sys
import numpy as np
import pytest
from PIL import Image
from pd_acoustics import visualize

@pytest.fixture
def results(tmp_path):
    (tmp_path/'metadata.json').write_text(json.dumps({
        'frequency_hz':40000,'nominal_arrival_s':.0001,'channel_height_m':.02,
        'channel_width_m':.02,'channel_depth_m':.02,'source_z_m':.014,
        'source_y_m':0.,'mic_xyz_m':[.01,0,-.002]}))
    t=np.array([0,.0001,.0002])
    np.savez(tmp_path/'signals.npz',t=t,v_source=np.array([.001,0,0]),p_mic=np.array([0,.002,0]),p_reference=t)
    x=np.linspace(-.01,.03,9);y=np.linspace(-.02,.02,9);z=np.linspace(-.01,.03,9)
    v=np.zeros((3,9,9,9),dtype=np.float32)
    v[:,2:7,2:7,2:7]=np.random.default_rng(0).normal(0,.01,(3,5,5,5))
    np.savez_compressed(tmp_path/'frames.npz',times=t,x=x,y=y,z=z,volume=v,volume_stride=1,
                        xz=v[:,: ,4,:].transpose(0,2,1),yz=v[:,4,:,:].transpose(0,2,1))
    return tmp_path

def invoke(monkeypatch,root,*extra):
    monkeypatch.setattr(sys,'argv',['pd-acoustics-visualize','--input',str(root),'--max-frames','2',*extra])
    visualize.main()

def test_gif_separate_files(monkeypatch,results):
    invoke(monkeypatch,results,'--format','gif')
    with Image.open(results/'pressure_animation.gif') as im:assert im.n_frames==2
    assert (results/'waveforms.png').stat().st_size>1000
    assert (results/'pressure_slices_preview.png').stat().st_size>1000

def test_default_saves_both_formats(monkeypatch,results):
    invoke(monkeypatch,results)
    assert (results/'pressure_animation.mp4').stat().st_size>0
    with Image.open(results/'pressure_animation.gif') as im:assert im.n_frames==2

def test_legacy_slice_data_still_works(monkeypatch,results,capsys):
    with np.load(results/'frames.npz') as d:
        np.savez(results/'frames.npz',times=d['times'],x=d['x'],y=d['y'],z=d['z'],xz=d['xz'],yz=d['yz'])
    invoke(monkeypatch,results,'--format','gif')
    assert (results/'pressure_animation.gif').exists()

def test_slices_legacy_supported(monkeypatch,results):
    invoke(monkeypatch,results,'--format','gif','--view','slices')
    assert (results/'pressure_animation.gif').exists()

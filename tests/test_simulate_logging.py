"""Exercise simulation logging through the command-line entry point."""
import json
import subprocess
import sys

import numpy as np
import pytest
import torch


def run_simulation(out, *extra):
    return subprocess.run(
        [sys.executable, '-m', 'pd_acoustics.simulate',
         '--device', 'cpu', '--duration-ms', '0.02',
         '--channel-height', '0.02', '--source-z', '0.01',
         '--y-halfspan', '0.008', '--channel-depth', '0.008', '--x-left', '-0.004',
         '--x-right', '0.024', '--z-bottom', '-0.012',
         '--z-top-extra', '0.004', '--mic-z', '-0.005',
         '--sponge-mm', '2', '--frame-every', '2',
         '--out', str(out), *extra],
        capture_output=True, text=True, timeout=60,
    )


@pytest.mark.parametrize('no_frames', [False, True])
@pytest.mark.parametrize('device', [
    'cpu',
    pytest.param('cuda:0', marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason='CUDA is unavailable')),
])
def test_progress_and_outputs(tmp_path, no_frames, device):
    log_path=tmp_path/'output.log'
    log_path.write_text('Previous run\n')
    result=run_simulation(tmp_path, '--device', device,
                          *(['--no-frames'] if no_frames else []))
    assert result.returncode==0, result.stderr
    log=log_path.read_text()
    assert log=='Previous run\n'+result.stdout
    for message in ('Starting simulation', 'Configuration:', f'Initializing device {device}',
                    'Grid ', 'Allocating pressure/velocity', 'Setup complete',
                    'Progress ', '100.0%', 'ETA 0.0s', 'Saving signals.npz',
                    'Saving metadata.json', 'Done:'):
        assert message in log
    meta=json.loads((tmp_path/'metadata.json').read_text())
    assert f'step {meta["nsteps"]}/{meta["nsteps"]}' in log
    with np.load(tmp_path/'signals.npz') as signals:
        assert set(signals.files)=={'t', 'p_mic', 'p_reference', 'v_source'}
        assert all(signals[key].shape==(meta['nsteps'],) for key in signals.files)
        assert all(np.isfinite(signals[key]).all() for key in signals.files)
    assert (tmp_path/'frames.npz').exists() is not no_frames
    assert ('No frames to save' if no_frames else 'Saving frames.npz') in log


def test_failure_is_logged_and_returns_nonzero(tmp_path):
    result=run_simulation(tmp_path, '--source-z', '1')
    assert result.returncode!=0
    log=(tmp_path/'output.log').read_text()
    assert 'Simulation failed' in log
    assert 'Traceback (most recent call last)' in log
    assert 'source-z must be inside channel' in log
    assert 'Done:' not in log
    assert 'ValueError' in result.stderr

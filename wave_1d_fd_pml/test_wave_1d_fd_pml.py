"""Test the propagators."""
import pytest
import numpy as np
from wave_1d_fd_pml.propagators import (Pml1, Pml2)

def ricker(freq, length, dt, peak_time):
    """Return a Ricker wavelet with the specified central frequency."""
    t = np.arange(-peak_time, (length)*dt - peak_time, dt, dtype=np.float32)
    y = ((1.0 - 2.0*(np.pi**2)*(freq**2)*(t**2))
         * np.exp(-(np.pi**2)*(freq**2)*(t**2)))
    return y

def green(x0, x1, dx, dt, t, v, v0, f):
    """Use the 1D Green's function to determine the wavefield at a given
    location and time due to the given source.
    """
    y = np.sum(f[:np.maximum(0, int((t - np.abs(x1-x0)/v)/dt))])*dt*dx*v0/2
    return y

@pytest.fixture
def model_one(nsteps=None, v0=1500, v1=2500, freq=25, dx=5, dt=0.0006, N=100):
    """Create a model with one reflector, and the expected wavefield."""
    rx = int(N/2)
    model = np.ones(N, dtype=np.float32) * v0
    model[rx:] = v1
    max_vel = v1
    if nsteps is None:
        nsteps = np.ceil(0.27/dt).astype(np.int)
    source = ricker(freq, nsteps, dt, 0.05)
    sx = int(.35 * N)
    expected = np.zeros(N)
    # create a new source shifted by the time to the reflector
    time_shift = np.round((rx-sx)*dx / v0 / dt).astype(np.int)
    shifted_source = np.pad(source, (time_shift, 0), 'constant')
    # reflection and transmission coefficients
    r = (v1 - v0) / (v1 + v0)
    t = 1 + r

    # direct wave
    expected[:rx] = np.array([green(x*dx, sx*dx, dx, dt,
                                    (nsteps+1)*dt, v0, v0,
                                    source) for x in range(rx)])
    # reflected wave
    expected[:rx] += r*np.array([green(x*dx, (rx-1)*dx, dx, dt,
                                       (nsteps+1)*dt, v0, v0,
                                       shifted_source) for x in range(rx)])
    # transmitted wave
    expected[rx:] = t*np.array([green(x*dx, rx*dx, dx, dt,
                                      (nsteps+1)*dt, v1, v0,
                                      shifted_source) for x in range(rx, N)])
    return {'model': model, 'dx': dx, 'dt': dt, 'nsteps': nsteps,
            'sources': np.array([source]), 'sx': np.array([sx]),
            'expected': expected}

@pytest.fixture
def model_two():
    """Create a random model and verify small wavefield at large t."""
    N = 100
    np.random.seed(0)
    model = np.random.random(N).astype(np.float32) * 3000 + 1500
    max_vel = 4500
    dx = 5
    dt = 0.6 * dx / max_vel
    nsteps = np.ceil(1.5/dt).astype(np.int)
    num_sources = 10
    sources_x = np.zeros(num_sources, dtype=np.int)
    sources = np.zeros([num_sources, nsteps], dtype=np.float32)
    for sourceIdx in range(num_sources):
        sources_x[sourceIdx] = np.random.randint(N)
        sources[sourceIdx, :] = ricker(25, nsteps, dt, 0.05)
    expected = np.zeros(N, dtype=np.float32)
    return {'model': model, 'dx': dx, 'dt': dt, 'nsteps': nsteps,
            'sources': sources, 'sx': sources_x, 'expected': expected}

@pytest.fixture
def versions():
    """Return a list of implementations."""
    return [Pml2]

def test_one_reflector(model_one, versions):
    """Verify that the numeric and analytic wavefields are similar."""

    for v in versions:
        _test_version(v, model_one, atol=1.0)


def test_allclose(model_two, versions):
    """Verify that wavefield of random model is damped."""

    for v in versions:
        _test_version(v, model_two, atol=1.0)


def _test_version(version, model, atol):
    """Run the test for one implementation."""
    v = version(model['model'], model['dx'], model['dt'])
    y = v.step(model['nsteps'], model['sources'], model['sx'])
    assert np.allclose(y, model['expected'], atol=atol)

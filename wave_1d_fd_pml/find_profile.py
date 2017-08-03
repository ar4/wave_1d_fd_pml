import numpy as np
import scipy.optimize
from hygsa import hygsa
from wave_1d_fd_pml import propagators, test_wave_1d_fd_pml

def evaluate(individual, profile, models, propagator):
    modelsum = 0.0
    for model in models:
        v = propagator(model['model'], model['dx'], model['dt'],
                       len(profile(individual)), profile=profile(individual))
        y = v.steps(model['nsteps'], model['sources'], model['sx'])
        ninf = (np.sum(~np.isfinite(v.current_wavefield)) +
                np.sum(~np.isfinite(v.current_phi)))
        if ninf == 0:
            modelsum += np.sum(np.abs(v.current_wavefield)+np.abs(v.current_phi))
        else:
            # Add a big number to discourage this
            modelsum += 1e15 * ninf
    return modelsum


def _get_prop(pml_version):
    if pml_version == 1:
        return propagators.Pml1
    elif pml_version == 2:
        return propagators.Pml2
    else:
        raise ValueError('unknown pml version')


def _get_models(vs, dx):
    dt = 0.0006 * dx / 5
    N = int(100 * 5 / dx)
    nsteps = int(500 * 0.0006 / dt)
    models = []
    for v0 in vs:
        for v1 in vs:
            models = [test_wave_1d_fd_pml.model_one(nsteps, v0=v0, v1=v1,
                                                    dx=dx, dt=dt, N=N)]
    return models


def find_profile(profile, bounds, optimizer, init=None, vs=[1500],
                 pml_version=2, dx=5, maxiter=500):

    models = _get_models(vs, dx)

    prop = _get_prop(pml_version)

    cost = lambda x: evaluate(x, profile, models, prop)

    if optimizer == 'hygsa':
        if init is None:
            raise ValueError('init needs to be set for HyGSA')
        ret = hygsa(cost, np.array(init), maxiter=maxiter, bounds=bounds)
        xmin = ret.x
        fxmin = ret.fun
    elif optimizer == 'brute':
        ret = scipy.optimize.brute(cost, bounds,
                                   Ns=int(maxiter ** (1/len(bounds))))
        xmin = ret
        fxmin = cost(ret)
    else:
        raise ValueError('unknown optimizer')
    print(optimizer, 'pml ', pml_version,
          ' global minimum: xmin = {0}, f(xmin) = {1}'.format(xmin, fxmin))
    return xmin, profile(xmin), fxmin

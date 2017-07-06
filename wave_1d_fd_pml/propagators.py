"""Propagate a 1D wavefield with a PML.
"""
import numpy as np
from wave_1d_fd_pml import pml

class Propagator(object):
    """A finite difference propagator for the 1D wave equation."""
    def __init__(self, model, dx, dt=None, abc_width=20, pad_width=8):
        self.nx = len(model)
        self.dx = np.float32(dx)
        self.abc_width = abc_width
        self.pad_width = pad_width
        self.total_pad = self.abc_width + self.pad_width
        max_vel = np.max(model)
        if dt:
            self.dt = dt
        else:
            self.dt = 0.6 * self.dx / max_vel
        self.nx_padded = self.nx + 2*self.total_pad
        self.model_padded = np.pad(model,
                                   (self.total_pad, self.total_pad),
                                   'edge')
        self.wavefield = [np.zeros(self.nx_padded, np.float32),
                          np.zeros(self.nx_padded, np.float32)
                         ]
        self.current_wavefield = self.wavefield[0]
        self.previous_wavefield = self.wavefield[1]

    def steps(self, num_steps, sources, sources_x, interval=1):
        num_intervals = int(np.floor(num_steps/interval))
        saved_steps = np.zeros([num_intervals, self.nx_padded])
        for i in range(num_intervals):
            self.step(interval, sources[:, i*interval:(i+1)*interval],
                      sources_x)
            saved_steps[i, :] = self.current_wavefield[:]
        return saved_steps


class Pml(Propagator):
    """Perfectly Matched Layer."""
    def __init__(self, model, dx, dt=None, pml_width=20, profile=None):
        super(Pml, self).__init__(model, dx, dt, pml_width)
        self.phi = [np.zeros(self.nx_padded, np.float32),
                    np.zeros(self.nx_padded, np.float32)
                   ]
        self.current_phi = self.phi[0]
        self.previous_phi = self.phi[1]

        if profile is None:
            profile = dx*np.arange(pml_width, dtype=np.float32)

        self.sigma = np.zeros(self.nx_padded, np.float32)
        self.sigma[self.total_pad-1:self.pad_width-1:-1] = profile
        self.sigma[-self.total_pad:-self.pad_width] = profile
        self.sigma[:self.pad_width] = self.sigma[self.pad_width]
        self.sigma[-self.pad_width:] = self.sigma[-self.pad_width-1]

    def step(self, num_steps, sources, sources_x):
        """Propagate wavefield."""

        num_sources = sources.shape[0]
        source_len = sources.shape[1]
        pml.pml.step(self.current_wavefield, self.previous_wavefield,
                     self.current_phi, self.previous_phi,
                     self.sigma,
                     self.model_padded, self.dt, self.dx,
                     sources, sources_x, num_steps,
                     self.abc_width, self.pad_width)

        if num_steps%2 != 0:
            self.current_wavefield, self.previous_wavefield = \
                    self.previous_wavefield, self.current_wavefield
            self.current_phi, self.previous_phi = \
                    self.previous_phi, self.current_phi

        return self.current_wavefield[self.total_pad: \
                                      self.nx_padded-self.total_pad]

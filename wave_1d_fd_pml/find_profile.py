import random
import multiprocessing
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from wave_1d_fd_pml import propagators, test_wave_1d_fd_pml

nsteps_global = 0
v_global = 0

def model_const(nsteps, v, freq):
    """Create a constant model."""
    N = 9
    sx = 5
    model = np.ones(N, dtype=np.float32) * v
    max_vel = v
    dx = 5
    dt = 0.001
    source = test_wave_1d_fd_pml.ricker(freq, nsteps, dt, 0.05)
    return {'model': model, 'dx': dx, 'dt': dt, 'nsteps': nsteps,
            'sources': np.array([source]), 'sx': np.array([sx])}


def evaluate(individual, profile_len, profile_max, v, freq, nsteps):
    model = model_const(nsteps, v, freq)
    v = propagators.Pml(model['model'], model['dx'], model['dt'], profile_len, profile=profile_max*np.array(individual))
    y = v.steps(model['nsteps'], model['sources'], model['sx'])
    if np.all(np.isfinite(v.current_wavefield)) and np.all(np.isfinite(v.current_phi)):
        return (np.sum(np.abs(v.current_wavefield)+np.abs(v.current_phi)),)
    else:
        return (np.inf,)

def myMut(ind1, toolbox):
    mutant = toolbox.clone(ind1)
    ind2, = tools.mutGaussian(mutant, mu=0.0, sigma=0.2, indpb=0.2)
    ind2[:] = np.abs(ind2[:])
    del mutant.fitness.values
    return (ind2,)


def find_profile(profile_len, profile_max=1000, v=1500, freq=25, nsteps=100, pop_len=500, ngen=50):

    nsteps_global = nsteps
    v_global = v

    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=profile_len)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    toolbox.register("evaluate", evaluate, profile_len=profile_len, profile_max=profile_max, v=v, freq=freq, nsteps=nsteps)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", myMut, toolbox=toolbox)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_len)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=ngen,
    stats=stats, halloffame=hof, verbose=True)

    return hof[0]

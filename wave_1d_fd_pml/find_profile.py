import random
import multiprocessing
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from wave_1d_fd_pml import propagators, test_wave_1d_fd_pml
import scipy.optimize

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


def evaluate(individual, profile_max, model):
    v = propagators.Pml(model['model'], model['dx'], model['dt'], len(individual), profile=profile_max*np.array(individual))
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

    model = model_const(nsteps, v, freq)

    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=profile_len)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    toolbox.register("evaluate", evaluate, profile_max=profile_max, model=model)
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


def find_profile2(profile_len, profile_max=1000, v=1500, freq=25, nsteps=100):
    
    model = model_const(nsteps, v, freq)

    def func(x):
        return evaluate(x, profile_max, model)

    brute_out = scipy.optimize.brute(func, [(0, 1)]*profile_len, disp=True)
    print('brute_out', brute_out['fval'])
    min_out = scipy.optimize.minimize(func, brute_out['x0'], options={'disp': True})
    print('min_out', min_out['fun'])

    return min_out['x']

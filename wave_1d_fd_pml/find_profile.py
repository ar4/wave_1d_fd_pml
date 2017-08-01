import random
import multiprocessing
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from wave_1d_fd_pml import propagators, test_wave_1d_fd_pml
import scipy.optimize
from hygsa import hygsa

def model_const(nsteps, v, freq, dx=5, N=9):
    """Create a constant model."""
    sx = int(N/2) + 1
    model = np.ones(N, dtype=np.float32) * v
    max_vel = v
    dt = 0.0006
    source = test_wave_1d_fd_pml.ricker(freq, nsteps, dt, 0.05)
    return {'model': model, 'dx': dx, 'dt': dt, 'nsteps': nsteps,
            'sources': np.array([source]), 'sx': np.array([sx])}


def evaluate(individual, profile, models, propagator):
    modelsum = 0.0
    for model in models:
        v = propagator(model['model'], model['dx'], model['dt'], len(profile(individual)), profile=profile(individual))
        y = v.steps(model['nsteps'], model['sources'], model['sx'])
        ninf = np.sum(~np.isfinite(v.current_wavefield)) + np.sum(~np.isfinite(v.current_phi))
        if ninf == 0:
            modelsum += np.sum(np.abs(v.current_wavefield)+np.abs(v.current_phi))
        else:
            modelsum += 1e15 * ninf#np.inf
    return (modelsum,)

def myMut(ind1, toolbox):
    mutant = toolbox.clone(ind1)
    ind2, = tools.mutGaussian(mutant, mu=0.0, sigma=0.2, indpb=0.2)
    ind2[:] = np.abs(ind2[:])
    del mutant.fitness.values
    return (ind2,)


def _get_prop(pml_version):
    if pml_version == 1:
        return propagators.Pml1
    elif pml_version == 2:
        return propagators.Pml2
    else:
        raise ValueError('unknown pml version')


def find_profile_freeform_deap(profile_len, profile_max=1000, vs=[1500], freqs=[25], nsteps=100, pop_len=500, ngen=50, pml_version=2):

    models = []
    for v in vs:
        for freq in freqs:
            models.append(model_const(nsteps, v, freq))

    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=profile_len)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    prop = _get_prop(pml_version)

    toolbox.register("evaluate", evaluate, lambda x: profile_max * np.array(x), models=models, propagator=prop)
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


def find_profile_freeform(profile_len, bounds=[0, 5000], vs=[1500], maxiter=500, pml_version=2):

    models = []
    for v0 in vs:
        for v1 in vs:
            models=[test_wave_1d_fd_pml.model_one(500, v0=v0, v1=v1)]

    prop = _get_prop(pml_version)

    profile = lambda x: np.array(x)
    func = lambda x: evaluate(x, profile, models, prop)[0]
    # Setting bounds
    lw = [bounds[0]] * profile_len
    up = [bounds[1]] * profile_len
    # Running the optimization computation
    ret = hygsa(func, np.linspace(0, 1000, num=profile_len), maxiter=maxiter, bounds=(zip(lw, up)))
    # Showing results
    print("global minimum: xmin = {0}, f(xmin) = {1}".format(ret.x, ret.fun))
    return ret.x, profile(ret.x)


def find_profile_linear(profile_len, intercept_bounds=[0, 5000], slope_bounds=[0, 500], vs=[1500], maxiter=500, pml_version=2, dx=5):
    # x0 + x1*x

    dt = 0.0006 * dx / 5
    nsteps = int(500 * 0.0006 / dt)
    models = []
    for v0 in vs:
        for v1 in vs:
            models=[test_wave_1d_fd_pml.model_one(nsteps, v0=v0, v1=v1, dx=dx, dt=dt)]


    prop = _get_prop(pml_version)

    profile = lambda x: x[0] + x[1] * np.arange(profile_len)
    cost = lambda x: evaluate(x, profile, models, prop)[0]
    # Running the optimization computation
    ret = hygsa(cost, np.array([0, 100]), maxiter=maxiter, bounds=(intercept_bounds, slope_bounds))
    # Showing results
    print("global minimum: xmin = {0}, f(xmin) = {1}".format(ret.x, ret.fun))
    return ret.x, profile(ret.x)


def find_profile_linear_brute(profile_len, intercept_bounds=[0, 5000], slope_bounds=[0, 500], vs=[1500], maxiter=500, pml_version=2):
    # x0 + x1*x

    models = []
    for v0 in vs:
        for v1 in vs:
            models=[test_wave_1d_fd_pml.model_one(500, v0=v0, v1=v1)]

    prop = _get_prop(pml_version)

    profile = lambda x: x[0] + x[1] * np.arange(profile_len)
    cost = lambda x: evaluate(x, profile, models, prop)[0]
    # Running the optimization computation
    N=100
    ret = scipy.optimize.brute(cost, [slice(intercept_bounds[0], intercept_bounds[1], (intercept_bounds[1] - intercept_bounds[0])/N), slice(slope_bounds[0], slope_bounds[1], (slope_bounds[1]-slope_bounds[0])/N)])
    # Showing results
    print("global minimum: xmin = {0}, f(xmin) = {1}".format(ret, cost(ret)))
    return ret, profile(ret)


def find_profile_power(profile_len, intercept_bounds=[0, 5000], slope_bounds=[0,500], power_bounds=[1, 10], vs=[1500], maxiter=500, pml_version=2):
    # (x0+x1*x)^x2

    models = []
    for v0 in vs:
        for v1 in vs:
            models=[test_wave_1d_fd_pml.model_one(500, v0=v0, v1=v1)]

    prop = _get_prop(pml_version)

    profile = lambda x: (x[0] + x[1] * np.arange(profile_len))**(x[2])
    cost = lambda x: evaluate(x, profile, models, prop)[0]
    # Running the optimization computation
    ret = hygsa(cost, np.array([80, 260, 1]), maxiter=maxiter, bounds=(intercept_bounds, slope_bounds, power_bounds))
    # Showing results
    print("global minimum: xmin = {0}, f(xmin) = {1}".format(ret.x, ret.fun))
    return ret.x, profile(ret.x)


def find_profile_cosine(profile_len, bounds=((0, 5000), (-5000, 5000), (0, np.pi/2), (0, np.pi/2)), vs=[1500], maxiter=500, pml_version=2):
    # x0 + x1*cos(x2:x2+x3)

    models = []
    for v0 in vs:
        for v1 in vs:
            models=[test_wave_1d_fd_pml.model_one(500, v0=v0, v1=v1)]

    prop = _get_prop(pml_version)

    profile = lambda x: x[0] + x[1]*(np.cos(np.linspace(x[2],x[2]+x[3], profile_len))).astype(np.float32)
    cost = lambda x: evaluate(x, profile, models, prop)[0]
    # Running the optimization computation
    ret = hygsa(cost, np.array([1000, -1000, 0, np.pi/2]), maxiter=maxiter, bounds=bounds)
    # Showing results
    print("global minimum: xmin = {0}, f(xmin) = {1}".format(ret.x, ret.fun))
    return ret.x, profile(ret.x)

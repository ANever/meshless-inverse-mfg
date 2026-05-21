import numpy as np
from clspde.utils import plot, eval_dict
from clspde.prepare import from_file, prepare_settings
from clspde.solution import Solution
import pickle as pkl
import pandas as pd

#Loading high-accuracy solution for the forward porblem
settings_filename = "simplest_mfg.yaml"
settings, sol_mes, iteration_dict = from_file(settings_filename)        
with open('colloc_solution_coefs.pkl', 'rb') as in_file:
    coefs = pkl.load(in_file)
sol_mes.cells_coefs = coefs['coefs']


#set up constants
noise_levels = [0, 0.01, 0.05, 0.1, 0.2]

T = 2
w = 0.1
gamma = 2
c = 4
S0 = 0.7
I0 = 1-S0

n_samples = 100
nn_steps = 10
experiments_results = np.zeros((nn_steps, len(noise_levels), n_samples))

#run the algorithm
number_of_steps = [50*2**i for i in range(nn_steps)]
for n_i, n in enumerate(number_of_steps):    
    S = np.zeros(n)
    I = np.zeros(n)
    dIdt = np.zeros(n)
    pS = np.zeros(n)
    pI = np.zeros(n)
    
    h = T/n
    
    print(n, '--------')
    for noise_i, noise in enumerate(noise_levels):
        print(noise)
        for sample_i in range(n_samples):
            for i in range(n):
                point = np.array([i*h - 1])
                pI[i] = (1-np.exp(gamma*(i*h-T)))*c/gamma
                I[i] = sol_mes.eval(point,[0],func=1) 
                dIdt[i] = sol_mes.eval(point,[1],func=1) 
            
            for i in range(n-1):
                I[i] *= (1+np.random.normal(loc=0.0, scale=noise, size=None))
                dIdt[i] *= (1+np.random.normal(loc=0.0, scale=noise, size=None))
            
            intI = np.zeros(n)
            intI[-1] = np.sum(I)*h
            S = S0 - I + I0 - gamma * intI

            beta = (dIdt/I + gamma)/S
            
            def new_u(u):
                #INT = np.zeros(n)
                #f = 2*u*(1+np.exp(-u)) - u**2
                #for i in range(1,n):
                #    INT[-i-1] = INT[-i] + h * f[-i]
                #return beta * I / 2 / (1+np.exp(-u)) * (INT - pI - w)
                return beta[-1] * I[-1] / 2 / (1+np.exp(-u)) * ( - w)
            
            u = 0
            change = 0.9
            for i in range(int(1e6)):
                _u = new_u(u)
                if (np.max(np.abs(_u - u)) < 1e-15):
                    break
                u = u*(1-change) + _u*(change)

            beta_max = beta[-1] * (1+np.exp(u))
            experiments_results[n_i, noise_i, sample_i] = abs(beta_max - 20)#[-1]

#output results into files
def output(var, name, func):
    data = func(var, axis = 2)
    df = pd.DataFrame(data)
    df.columns = noise_levels
    df['steps'] = number_of_steps
    df.to_csv(name + '.csv')

output(experiments_results, 'mean', np.mean)
output(experiments_results/20, 'rel_mean', np.mean)
output(experiments_results, 'std', np.std)
output(experiments_results/20, 'rel_std', np.std)

print((np.abs(beta_max-20)))
import numpy as np
from clspde.utils import plot, eval_dict
from clspde.prepare import from_file, prepare_settings
from clspde.solution import Solution
import pickle as pkl
import pandas as pd

import matplotlib.pyplot as plt

#Loading high-accuracy solution for the forward porblem
settings_filename = "simplest_mfg.yaml"
settings, sol_mes, iteration_dict = from_file(settings_filename)        
with open('colloc_solution_coefs.pkl', 'rb') as in_file:
    coefs = pkl.load(in_file)
sol_mes.cells_coefs = coefs['coefs']


#set up constants
noise_levels = [0, 0.01, 0.05, 0.1, 0.2]

T = 2
#w = 0.1
gamma = 2
#c = 4
S0 = 0.7
I0 = 1-S0

n_samples = 100
nn_steps = 10
experiments_results = np.zeros((nn_steps, len(noise_levels), n_samples))


#initial guess
beta_max = 20
c = 4
w = 0.1

#run the algorithm
number_of_steps = [50*2**i for i in range(4,nn_steps)]

initial_flag = True
for n_i, n in enumerate(number_of_steps):    
    S = np.zeros(n)
    I = np.zeros(n)
    uS = np.zeros(n)
    uI = np.zeros(n)
    
    dIdt = np.zeros(n)
    pS = np.zeros(n)
    pI = np.zeros(n)
    integral_deriv = np.zeros(n)
    integral = np.zeros(n)
    h = T/n
    
    print(n, '--------')
    for noise_i, noise in enumerate(noise_levels):
        print(noise)
        for sample_i in range(n_samples):
            
            integral[-1] = 0
            integral_deriv[-1] = 0
            
            intI = np.zeros(n)
            
            
            for i in range(n):
                point = np.array([i*h - 1])
                #pI[i] = (1-np.exp(gamma*(i*h-T)))*c/gamma
                S[i] = sol_mes.eval(point,[0],func=0)
                I[i] = sol_mes.eval(point,[0],func=1)
                uS[i] = sol_mes.eval(point,[0],func=2)
                uI[i] = sol_mes.eval(point,[0],func=3)
                
                I[i] *= (1+np.random.normal(loc=0.0, scale=noise, size=None))
                dIdt[i] = sol_mes.eval(point,[1],func=1) 
                dIdt[i] *= (1+np.random.normal(loc=0.0, scale=noise, size=None))
                intI[i] = intI[i-1] + I[i]*h
            
            #c_ = lambda u_: np.abs( beta_max *( uS - uI - w ) * I / 2.)
            #c_sign = np.sign(( uS - uI - w ))
            #u_ = 0
            #alpha = -6.63320111e-01 + np.sqrt(c(u_)) *8.92654544e-01 - c(u_) * 6.99385982e-02 + c(u_)**2* 2.97831057e-04 - c(u_)**3 * 1.08549568e-06 + c(u_)**4*1.68015701e-09 +np.exp(-c(u_))*  3.66041065e-02 + np.exp(-2*c(u_))*  4.74148560e-01
            #u = alpha * c_sign
            #data_validity = eval('beta_max * ( uS - uI - w ) * I / 2. - 2*u*(1+np.exp(u))*(1+np.exp(-u))', locals())
            #print(data_validity)
            #sleep(2)
            
            intI[-1] = np.sum(I)*h
            #S = np.max([S0 - I + I0 - gamma * intI, np.zeros(n)+1e-5], axis=0)
            S = S0 - I + I0 - gamma * intI
            
            beta = (dIdt/I + gamma)/S
            
            if initial_flag:
                beta_max = 2*np.max(beta)
                initial_flag = False
            
            subfunc_deriv = (2*(beta_max)/(beta_max-beta) - 2*np.log(beta_max/beta - 1))
            u = np.log(beta_max/beta - 1)
            subfunc = 2*u*(1+np.exp(-u)) - u**2
            
            for i in range(n-1):
                integral_deriv[-2-i] = integral_deriv[-1-i] + subfunc_deriv[i]*h
                integral[-2-i] = integral[-1-i] + subfunc[i]*h
            
            
            #plt.plot(beta_max-beta)
            #plt.show()
            
            points_for_grads = [int(n/4),int(n/2), n-1]
            npfg = len(points_for_grads)
            grad = np.zeros((npfg,npfg))
            f = np.zeros(npfg)
            
            
            
            
            for i in range(100):
                params = np.array([beta_max, c])
                ps = integral
                dps_dbeta = integral_deriv
                
                def f_complete(params):
                    beta_max, c = params
                    t = h*i
                    u = np.log(beta_max/beta - 1)
                    #return c/gamma*(1-np.exp(gamma*(i*h-T))) + w - integral[i] + (2*u*(1+np.exp(-u))/beta/I)[i]
                    return (-2*u + (beta_max* np.exp(u)*I*(c + c*np.exp(gamma*(t - T)) + gamma*w - gamma*ps))/((1 + np.exp(u)**2)*gamma))
                    
                def f_for_grads(i,params):
                    return f_complete(params)[i]
                    
                for i_i, i in enumerate(points_for_grads):
                    grad_beta = (2/(beta- beta_max) + (beta**2*I*(c + c*np.exp(gamma*(i*h - T)) + gamma*w))/(beta_max**2*gamma) + (beta*I*(-beta*ps + (beta- beta_max)*beta_max*dps_dbeta))/beta_max**2)
                    grad_c = ((beta*np.exp(u)*(1 + np.exp(gamma*(i*h - T)))*I)/((1 + np.exp(u)) * gamma))
                    #grad_w = ((beta*np.exp(u)*I)/(1 + np.exp(u)))
                    #grad[i_i] = np.array([grad_beta[i], grad_c[i], grad_w[i]])
                    grad = np.array([grad_beta, grad_c])#, grad_w])
                    
                    #plt.plot(grad_c)
                    #plt.show()
                    #f[i_i] = f_for_grads(i, params)
                    f = f_complete(params)
                    
                eps = 0.01
                #params = params - eps*np.linalg.inv(grad)@f
                print(grad.shape, f.shape)
                params = params - eps*grad@f/n#np.sum(grad*f, axis=1)/n
                #print('GRAD ', np.linalg.inv(grad)@f)
                print('GRAD ', np.sum(grad, axis=1)/n)
                print('VAL  ', params)
                print(f[:4])
                beta_max, c = params
                
                
            experiments_results[n_i, noise_i, sample_i] = abs(beta_max - 20)#[-1]
        #plt.plot((S - np.roll(S,1))[1:]/h)
        #plt.show()
        print(beta_max, beta_max/np.max(beta))
        plt.plot(beta)
        plt.show()
        '''plt.plot(dIdt)
        plt.show()
        plt.plot(dIdt/I)
        plt.show()
        plt.plot(dIdt/I/S)
        plt.show()
        plt.plot(S)
        plt.plot(I)
        plt.show()
        '''
        
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
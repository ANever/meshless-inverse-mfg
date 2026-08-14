import numpy as np
import matplotlib.pyplot as plt

n = 50*2**10
T = 2

h = T/n

beta_max = 20
gamma = 2
c = 4
w = 0.1
#cI = 0.2

S0 = 0.7
I0 = 0.3

S = np.zeros(n)
I = np.zeros(n)
psi_S = np.zeros(n)
psi_I = np.zeros(n)
u = np.zeros(n)
beta = np.zeros(n)

t = np.linspace(0,T,n)

def u_f(t):
    return (t-T)**2/4
u = u_f(t)

def beta(u):
    return beta_max/(1+np.exp(u))

def euler(x0, f, n=n):
    x = np.zeros((n,len(x0)))
    x[0] = x0
    h = T/n
    for i in range(n-1):
        x[i+1] = x[i] + h*f(x[i],h*i)
    return x

def rk4(x0, f, n=n):
    x = np.zeros((n,len(x0)))
    x[0] = x0
    h = T/n
    for i in range(n-1):
        k1 = h * f(x[i],h*i)
        k2 = h * f(x[i]+k1/2,h*(i+0.5))
        k3 = h * f(x[i]+k2/2,h*(i+0.5))
        k4 = h * f(x[i]+k3,h*(i+1))
        x[i+1] = x[i] + (k1+2*k2+2*k3+k4)/6
    return x


def SInext(x,t):
    res = np.zeros(2)
    S,I = x
    res[0] = -S*I*beta(u_f(t))
    res[1] = S*I*beta(u_f(t)) - gamma*I
    return res

def generate_data():
    x = rk4(np.array([S0,I0]), SInext)    
    S = x[:,0]
    I = x[:,1]
    psi_I = c/gamma * (1-np.exp(gamma*(t-T)))
    psi_S = psi_I + w +2*u*(1+np.exp(-u))/beta(u)/I
    return np.array([S,I,psi_S,psi_I])

x = generate_data()[:2].T
x2 = rk4(np.array([S0,I0]), SInext,2*n)

print(np.max(np.abs(x-x2[::2])))

S = x[:,0]
I = x[:,1]

psi_I = c/gamma * (1-np.exp(gamma*(t-T)))
psi_S = psi_I + w +2*u*(1+np.exp(-u))/beta(u)/I

dpsi_S = np.zeros(n)

for i in range(1,n-1):
    dpsi_S[i] = (psi_S[i+1] - psi_S[i-1])/h/2
dpsi_S[-1] = dpsi_S[-2]
dpsi_S[0] = dpsi_S[1]

psi_eq_correction = dpsi_S - 2*u*(1+np.exp(-u)) + u**2

def generate_correction():
    return ( dpsi_S - I * beta(u) * (psi_S - psi_I - w) + u**2 )
    #dpsi_S - 2*u*(1+np.exp(-u)) + u**2


if __name__=='__main__':
    plt.plot(S)
    plt.plot(I)
    plt.show()

    plt.plot(psi_S)
    plt.plot(psi_I)
    plt.show()

    plt.plot(dpsi_S - 2*u*(1+np.exp(-u)) + u**2)
    plt.show()


#plt.plot(psi_eq_correction - generate_correction())
#plt.plot(generate_correction())
#plt.show()
'''
S = np.array(S0*np.exp(-beta_max*cI*t))
I = S0*beta_max*cI/(beta_max*cI-gamma) * (-np.exp(-beta_max*cI*t) + np.exp(-gamma*t)) + I0*np.exp(-gamma*t)

psi_I = c/gamma * (1-np.exp(-gamma*(t-T)))
u = np.log(I/cI-1)

psi_S = 2*cI/beta_max * u *(1+np.exp(-u)) + psi_S + w

plt.plot(S)
plt.plot(I)
plt.show()

plt.plot(psi_S)
plt.plot(psi_I)
plt.show()
'''
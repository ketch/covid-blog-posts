import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.dates as dates
import data

cases, deaths, today, days = data.jhu_data()

def SIR(u0, beta=0.25, gamma=0.05, N = 1, T=14):

    du = np.zeros(3)
    
    def f(t,u):
        du[0] = -beta*u[1]*u[0]/N
        du[1] = beta*u[1]*u[0]/N - gamma*u[1]
        du[2] = gamma*u[1]
        return du

    times = np.arange(0,T+1)
    solution = solve_ivp(f,[0,T],u0,t_eval=times)
    S = solution.y[0,:]
    I = solution.y[1,:]
    R = solution.y[2,:]
    
    return S, I, R

def forecast(region='Spain',ifr=1,mttd=9,scale='linear',beta=0.25,gamma=0.04,forecast_length=14):

    ifr = ifr/100.
    N = data.population[region]
    
    if scale == 'linear':
        plotfun = plt.plot_date
    else:
        plotfun = plt.semilogy
        
    total_cases, total_deaths = data.load_cases(region)
    start = dates.datestr2num(cases.columns[4])
    my_dates = np.arange(start-mttd,start+len(days))
    total_deaths = np.insert(total_deaths,0,[0]*mttd)
    
    new_deaths = np.diff(total_deaths); new_deaths = np.insert(new_deaths,0,0)
    
    new_infections = np.zeros_like(my_dates)
    total_recovered = np.zeros_like(my_dates)
    
    new_infections[:-mttd] = new_deaths[mttd:]/ifr
    total_infections = np.cumsum(new_infections)
    for i in range(len(my_dates)):
        total_recovered[i] = np.sum(new_infections[:i]*(1-np.exp(-gamma*(i-np.arange(i)))))
    active_infections = total_infections - total_recovered
    
    
    # Initial values, mttd days ago
    I0 = active_infections[-(mttd+1)]
    R0 = total_recovered[-(mttd+1)]
    u0 = np.array([N-I0-R0,I0,R0])

    # Now run the model
    S, I, R = SIR(u0, beta=beta, gamma=gamma, N=N, T=forecast_length)
    
    prediction_dates = my_dates[-(mttd+1)]+range(forecast_length+1)
    predicted_deaths = R*ifr
    predicted_deaths = predicted_deaths - (predicted_deaths[0]-total_deaths[-(mttd+1)])
    plotfun(my_dates,total_deaths,'-',lw=3,label='Deaths (recorded)')
    plotfun(prediction_dates,predicted_deaths,'.',label='Deaths (predicted)')
    plt.legend()



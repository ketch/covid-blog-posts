import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.dates as dates
import data

cases, deaths, today, days = data.jhu_data()

def get_mttd(new_deaths):
    mean=17; std=7
    window=mean+std
    t=np.arange(-2*std,2*std+1);
    p=np.exp(-(t)**2/2/std**2)/np.sqrt(2*np.pi*std**2)
    dist = np.convolve(new_deaths[-window:],p)
    offset = np.sum([(i+0.5)*dist[i] for i in range(len(dist))])/np.sum([dist[i]for i in range(len(dist))])
    return mean-((offset-2*std-window/2))

def SIR(u0, beta=0.25, gamma=0.05, N = 1, T=14, q=0, intervention_start=0, intervention_length=0):

    du = np.zeros(3)
    
    def f(t,u):
        if intervention_start<t<intervention_start+intervention_length:
            qq = q
        else:
            qq = 0.
        du[0] = -(1-qq)*beta*u[1]*u[0]/N
        du[1] = (1-qq)*beta*u[1]*u[0]/N - gamma*u[1]
        du[2] = gamma*u[1]
        return du

    times = np.arange(0,T+1)
    solution = solve_ivp(f,[0,T],u0,t_eval=times,method='RK23')
    S = solution.y[0,:]
    I = solution.y[1,:]
    R = solution.y[2,:]
    
    return S, I, R

def forecast(region='Spain',ifr=1,beta=0.25,gamma=0.04,intervention_level='No action',
             intervention_start=0,intervention_length=30,forecast_length=14,scale='linear'):
    """Forecast with SIR model.  All times are in days.

        Inputs:
         - ifr: infection fatality ratio
         - mttd: mean time to death
         - intervention_level: one of 'No action', 'Limited action', 'Social distancing',
                'Shelter in place', or 'Full lockdown'.
         - intervention_start: when intervention measure starts, relative to today (can be negative)
         - intervention_length (in days from start)
    """

    ifr = ifr/100.
    N = data.population[region]
    q = data.intervention_strength[intervention_level]
    
    if scale == 'linear':
        plotfun = plt.plot_date
    else:
        plotfun = plt.semilogy
        
    total_cases, total_deaths = data.load_cases(region)
    start = dates.datestr2num(cases.columns[4])
    
    new_deaths = np.diff(total_deaths); new_deaths = np.insert(new_deaths,0,0)
    mttd = int(round(get_mttd(new_deaths)))

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
    S, I, R = SIR(u0, beta=beta, gamma=gamma, N=N, T=forecast_length, q=q,
                    intervention_start=intervention_start+mttd,
                    intervention_length=intervention_length)
    
    prediction_dates = my_dates[-(mttd+1)]+range(forecast_length+1)
    predicted_deaths = R*ifr
    predicted_deaths = predicted_deaths - (predicted_deaths[0]-total_deaths[-(mttd+1)])
    plotfun(my_dates,total_deaths,'-',lw=3,label='Deaths (recorded)')
    plotfun(prediction_dates,predicted_deaths,'.',label='Deaths (predicted)')
    plt.legend()

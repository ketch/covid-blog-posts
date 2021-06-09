import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.dates as mdates
import data

hr = 10 # Hospitalizations per death

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
    solution = solve_ivp(f,[0,T],u0,t_eval=times,method='RK23',atol=1.e-3,rtol=1.e-3)
    S = solution.y[0,:]
    I = solution.y[1,:]
    R = solution.y[2,:]
    
    return S, I, R

def forecast(region='Spain',ifr=1,beta=0.25,gamma=0.04,intervention_level='No action',
             intervention_start=0,intervention_length=30,forecast_length=14,scale='linear',
             plot_type='cumulative',plot_value='deaths',plot_past_pred=True,plot_interval=True):
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
    start = mdates.datestr2num(cases.columns[4])
    
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
    S_mean, I_mean, R_mean = SIR(u0, beta=beta, gamma=gamma, N=N, T=mttd+forecast_length, q=q,
                                 intervention_start=intervention_start+mttd,
                                 intervention_length=intervention_length)
    
    S_low, I_low, R_low = S_mean.copy(), I_mean.copy(), R_mean.copy()
    S_high, I_high, R_high = S_mean.copy(), I_mean.copy(), R_mean.copy()
    dd_low = np.diff(R_mean); dd_high = np.diff(R_mean)

    prediction_dates = my_dates[-(mttd+1)]+range(forecast_length+mttd+1)
    predicted_deaths = R_mean*ifr
    predicted_deaths = predicted_deaths - (predicted_deaths[mttd]-total_deaths[-1])

    if plot_interval:
        dr_low = np.diff(R_mean); dr_high = np.diff(R_mean)
        for dbeta in np.linspace(-0.05,0.1,6):
            for dgamma in np.linspace(-0.02,0.08,6):
                S, I, R= SIR(u0, beta=beta+dbeta, gamma=gamma+dgamma, N=N, T=mttd+forecast_length, q=q,
                             intervention_start=intervention_start+mttd,
                             intervention_length=intervention_length)

                S_low = np.minimum(S_low,S)
                I_low = np.minimum(I_low,I)
                R_low = np.minimum(R_low,R)
                S_high = np.maximum(S_high,S)
                I_high = np.maximum(I_high,I)
                R_high = np.maximum(R_high,R)
                dr_low = np.minimum(dr_low,np.diff(R))
                dr_high = np.maximum(dr_high,np.diff(R))
     
        predicted_deaths_low  = R_low*ifr
        predicted_deaths_low = predicted_deaths_low - (predicted_deaths_low[mttd]-total_deaths[-1])
        predicted_deaths_high = R_high*ifr
        predicted_deaths_high = predicted_deaths_high - (predicted_deaths_high[mttd]-total_deaths[-1])

        dd_low = dd_low*ifr; dd_high = dd_high*ifr


    if plot_past_pred: pred_start_ind=0
    else: pred_start_ind = mttd

    if plot_type=='cumulative':
        if plot_value == 'deaths':
            plotfun(my_dates,total_deaths,'-',lw=3,label='Deaths (recorded)')
            plotfun(prediction_dates[pred_start_ind:],predicted_deaths[pred_start_ind:],'-k',label='Deaths (predicted)')
            if plot_interval:
                plt.fill_between(prediction_dates[mttd:],predicted_deaths_low[mttd:],predicted_deaths_high[mttd:],color='grey',zorder=-1)
        elif plot_value == 'hospitalizations':
            plotfun(prediction_dates[mttd:],predicted_deaths[mttd:]*hr,'-k',label='Hospitalizations (predicted)')
            if plot_interval:
                plt.fill_between(prediction_dates[mttd:],predicted_deaths_low[mttd:]*hr,predicted_deaths_high[mttd:]*hr,color='grey',zorder=-1)
    elif plot_type=='daily':
        if plot_value == 'deaths':
            plotfun(my_dates[1:],np.diff(total_deaths),'-',lw=3,label='Deaths (recorded)')
            plotfun(prediction_dates[pred_start_ind+1:],np.diff(predicted_deaths[pred_start_ind:]),'-k',label='Deaths (predicted)')
            if plot_interval:
                plt.fill_between(prediction_dates[mttd+1:],dd_low[mttd:],dd_high[mttd:],color='grey',zorder=-1)
        elif plot_value == 'hospitalizations':
            plotfun(prediction_dates[mttd+1:],np.diff(predicted_deaths[mttd:])*hr,'-k',label='Hospitalizations (predicted)')
            if plot_interval:
                plt.fill_between(prediction_dates[mttd+1:],dd_low[mttd:]*hr,dd_high[mttd:]*hr,color='grey',zorder=-1)

    plt.legend(loc='best')


    ax = plt.gca()
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.title('{} {}-day forecast with {} for {} days'.format(region,forecast_length,intervention_level,intervention_length))
    #ax.xaxis.set_major_locator(mdates.MonthLocator())
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

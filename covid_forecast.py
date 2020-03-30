import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.dates as mdates
import data
import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

hr = 10 # Hospitalizations per death

# data_days is days for which we have recorded data
cases, deaths, today, data_days = data.jhu_data()

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

def retrieve_data(region='Spain'):
    N = data.population[region]
    total_cases, total_deaths = data.load_cases(region)
    data_start = mdates.datestr2num(cases.columns[4])  # First day for which we have data
    return N, total_deaths, data_start
    

def infer_initial_data(total_deaths,data_start,ifr,gamma,N):
    daily_deaths = np.diff(total_deaths); daily_deaths = np.insert(daily_deaths,0,0)
    mttd = int(round(get_mttd(daily_deaths)))

    inferred_data_dates = np.arange(data_start-mttd,data_start+len(data_days))
    total_deaths = np.insert(total_deaths,0,[0]*mttd)
    daily_deaths = np.diff(total_deaths); daily_deaths = np.insert(daily_deaths,0,0)
    
    new_infections = np.zeros_like(inferred_data_dates)
    total_recovered = np.zeros_like(inferred_data_dates)
    
    new_infections[:-mttd] = daily_deaths[mttd:]/ifr
    total_infections = np.cumsum(new_infections)
    for i in range(len(inferred_data_dates)):
        total_recovered[i] = np.sum(new_infections[:i]*(1-np.exp(-gamma*(i-np.arange(i)))))
    active_infections = total_infections - total_recovered
    
    
    # Initial values, mttd days ago
    I0 = active_infections[-(mttd+1)]
    R0 = total_recovered[-(mttd+1)]
    u0 = np.array([N-I0-R0,I0,R0])
    return u0, mttd, inferred_data_dates, total_deaths


def forecast(u0,mttd,N,inferred_data_dates,total_deaths,ifr=0.01,beta=0.25,gamma=0.04,q=1.0,intervention_start=0,
             intervention_length=30,forecast_length=14,compute_interval=True):
    """Forecast with SIR model.  All times are in days.

        Inputs:
         - ifr: infection fatality ratio
         - mttd: mean time to death
         - intervention_level: one of 'No action', 'Limited action', 'Social distancing',
                'Shelter in place', or 'Full lockdown'.
         - intervention_start: when intervention measure starts, relative to today (can be negative)
         - intervention_length (in days from simulation start)
    """

    
    # Now run the model
    S_mean, I_mean, R_mean = SIR(u0, beta=beta, gamma=gamma, N=N, T=mttd+forecast_length, q=q,
                                 intervention_start=intervention_start+mttd,
                                 intervention_length=intervention_length)
    
    S_low, I_low, R_low = S_mean.copy(), I_mean.copy(), R_mean.copy()
    S_high, I_high, R_high = S_mean.copy(), I_mean.copy(), R_mean.copy()
    dd_low = np.diff(R_mean); dd_high = np.diff(R_mean)

    prediction_dates = inferred_data_dates[-(mttd+1)]+range(forecast_length+mttd+1)
    pred_cumu_deaths = R_mean*ifr
    pred_cumu_deaths = pred_cumu_deaths - (pred_cumu_deaths[mttd]-total_deaths[-1])

    if compute_interval:
        pred_daily_deaths_low = np.diff(R_mean); pred_daily_deaths_high = np.diff(R_mean)
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
                pred_daily_deaths_low = np.minimum(pred_daily_deaths_low,np.diff(R))
                pred_daily_deaths_high = np.maximum(pred_daily_deaths_high,np.diff(R))
     
        pred_cumu_deaths_low  = R_low*ifr
        pred_cumu_deaths_low = pred_cumu_deaths_low - (pred_cumu_deaths_low[mttd]-total_deaths[-1])
        pred_cumu_deaths_high = R_high*ifr
        pred_cumu_deaths_high = pred_cumu_deaths_high - (pred_cumu_deaths_high[mttd]-total_deaths[-1])

        pred_daily_deaths_low = pred_daily_deaths_low*ifr; pred_daily_deaths_high = pred_daily_deaths_high*ifr

        return prediction_dates, pred_cumu_deaths, pred_cumu_deaths_low, pred_cumu_deaths_high, pred_daily_deaths_low, pred_daily_deaths_high
    else:
        return prediction_dates, pred_cumu_deaths, None, None, None, None


def plot_forecast(inferred_data_dates, total_deaths, mttd, prediction_dates, pred_cumu_deaths, pred_cumu_deaths_low,
                  pred_cumu_deaths_high, pred_daily_deaths_low, pred_daily_deaths_high, plot_title,
                  plot_past_pred=True, plot_type='cumulative',
                  plot_interval=True, plot_value='deaths',scale='linear'):

    if scale == 'linear':
        plotfun = plt.plot_date
    else:
        plotfun = plt.semilogy
        
    if plot_past_pred: pred_start_ind=0
    else: pred_start_ind = mttd

    if plot_type=='cumulative':
        if plot_value == 'deaths':
            plotfun(inferred_data_dates,total_deaths,'-',lw=3,label='Deaths (recorded)')
            plotfun(prediction_dates[pred_start_ind:],pred_cumu_deaths[pred_start_ind:],'-k',label='Deaths (predicted)')
            if plot_interval:
                plt.fill_between(prediction_dates[mttd:],pred_cumu_deaths_low[mttd:],pred_cumu_deaths_high[mttd:],color='grey',zorder=-1)
        elif plot_value == 'hospitalizations':
            plotfun(prediction_dates[mttd:],pred_cumu_deaths[mttd:]*hr,'-k',label='Hospitalizations (predicted)')
            if plot_interval:
                plt.fill_between(prediction_dates[mttd:],pred_cumu_deaths_low[mttd:]*hr,pred_cumu_deaths_high[mttd:]*hr,color='grey',zorder=-1)
    elif plot_type=='daily':
        if plot_value == 'deaths':
            plotfun(inferred_data_dates[1:],np.diff(total_deaths),'-',lw=3,label='Deaths (recorded)')
            plotfun(prediction_dates[pred_start_ind+1:],np.diff(pred_cumu_deaths[pred_start_ind:]),'-k',label='Deaths (predicted)')
            if plot_interval:
                plt.fill_between(prediction_dates[mttd+1:],dd_low[mttd:],dd_high[mttd:],color='grey',zorder=-1)
        elif plot_value == 'hospitalizations':
            plotfun(prediction_dates[mttd+1:],np.diff(pred_cumu_deaths[mttd:])*hr,'-k',label='Hospitalizations (predicted)')
            if plot_interval:
                plt.fill_between(prediction_dates[mttd+1:],dd_low[mttd:]*hr,dd_high[mttd:]*hr,color='grey',zorder=-1)

    plt.legend(loc='best')


    ax = plt.gca()
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.title(plot_title)


def compute_and_plot(region='Spain',ifr=1,beta=0.25,gamma=0.04,intervention_level='No action',
             intervention_start=0,intervention_length=30,forecast_length=14,scale='linear',
             plot_type='cumulative',plot_value='deaths',plot_past_pred=True,plot_interval=True):

    ifr = ifr/100.

    N, total_deaths, data_start = retrieve_data(region)

    u0, mttd, inferred_data_dates, total_deaths = infer_initial_data(total_deaths,data_start,ifr,gamma,N)

    q = data.intervention_strength[intervention_level]

    prediction_dates, pred_cumu_deaths, pred_cumu_deaths_low, \
      pred_cumu_deaths_high, pred_daily_deaths_low, pred_daily_deaths_high \
      = forecast(u0,mttd,N,inferred_data_dates,total_deaths,ifr,beta,gamma,q,intervention_start,intervention_length,forecast_length,plot_interval)

    plot_title = '{} {}-day forecast with {} for {} days'.format(region,forecast_length,intervention_level,intervention_length)
    plot_forecast(inferred_data_dates, total_deaths, mttd, prediction_dates, pred_cumu_deaths, pred_cumu_deaths_low,
                  pred_cumu_deaths_high, pred_daily_deaths_low, pred_daily_deaths_high,
                  plot_title, plot_past_pred=plot_past_pred, plot_type=plot_type,
                  plot_interval=plot_interval, plot_value=plot_value, scale=scale)


def write_JSON(regions, forecast_length=14):

    output = {}
    ifr = 1/100
    gamma = 0.05
    beta = 0.25

    for region in regions:
        
        # These should be adjusted for each region:
        intervention_level='No action'
        intervention_start=0
        intervention_length=30

        N, total_deaths, data_start = retrieve_data(region)

        u0, mttd, inferred_data_dates, total_deaths = infer_initial_data(total_deaths,data_start,ifr,gamma,N)

        q = data.intervention_strength[intervention_level]

        prediction_dates, pred_cumu_deaths, pred_cumu_deaths_low, \
          pred_cumu_deaths_high, pred_daily_deaths_low, pred_daily_deaths_high \
          = forecast(u0,mttd,N,inferred_data_dates,total_deaths,ifr,beta,gamma,
                     q,intervention_start,intervention_length,forecast_length,compute_interval=True)
        
        pred_daily_deaths = np.diff(pred_cumu_deaths);
        output[region] = {}
        output[region]['dates'] = prediction_dates[mttd+1:]
        output[region]['deaths'] = pred_daily_deaths[mttd:]
        output[region]['deaths_low'] = pred_daily_deaths_low[mttd:]
        output[region]['deaths_high'] = pred_daily_deaths_high[mttd:]
        
    with open('forecast.json', 'w') as file:
        json.dump(output, file, cls=NumpyEncoder)

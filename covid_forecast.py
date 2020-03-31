import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.dates as mdates
import data
import json
from scipy import optimize
from utils import NumpyEncoder

"""
Code for modeling and predicting the COVID-19 outbreak.
General nomenclature:
    - cum_* = cumulative values
    - daily_* = daily increments
    - inf = inferred (inference model output)
    - pred = predicted (SIR model output)
"""

# ifr = 0.067
hr = 10 # Hospitalizations per death

def get_mttd(daily_deaths):
    """
    Determine approximate mean time to death (in days).
    This is not the mean time for an individual, but rather
    the mean time of illness of individuals who died today (for a given region).
    These are quite different when the rate of infection is rapidly increasing
    or decreasing.

    This would more appropriately be done by deconvolution, but I
    haven't found a stable way to do that, so this is an approximate
    way of getting the deconvolved mean.
    """
    mean=17; std=7
    window=mean+std
    t=np.arange(-2*std,2*std+1);
    p=np.exp(-(t)**2/2/std**2)/np.sqrt(2*np.pi*std**2)
    dist = np.convolve(daily_deaths[-window:],p)
    offset = np.sum([(i+0.5)*dist[i] for i in range(len(dist))])/np.sum([dist[i]for i in range(len(dist))])
    return mean-((offset-2*std-window/2))


def SIR(u0, beta=0.25, gamma=0.05, N = 1, T=14, q=0, intervention_start=0, intervention_length=0):
    """
    Run the SIR model with initial data u0 and given parameters.
        - q: intervention strength (1=no human contact; 0=normal contact)
        - intervention_start, intervention_length: measured from simulation start (t=0) in days

    In this version there is only one intervention period.  See SIR2() for a version
    with any number of intervention periods.
    """

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


def SIR2(u0, beta=0.25, gamma=0.05, N = 1, T=14, q=[0], intervention_dates=[0,0]):
    """
    Run the SIR model with initial data u0 and given parameters.
        - q: list of intervention strengths (1=no human contact; 0=normal contact)
        - intervention_dates: dates when we switch from one q value to the next.
          First entry is start of first intervention.

    In this version there is only one intervention period.  See SIR2() for a version
    with any number of intervention periods.
    """
    du = np.zeros(3)
    
    def f(t,u):
        i = np.argmax(intervention_dates>t)-1
        if i == -1:
            qq = 0.
        else:
            qq = q[i]

        du[0] = -(1-qq)*beta*u[1]*u[0]/N
        du[1] = (1-qq)*beta*u[1]*u[0]/N - gamma*u[1]
        du[2] = gamma*u[1]
        return du

    times = np.arange(0,T+1)
    # Perhaps we should break this up into one call for each intervention interval.
    solution = solve_ivp(f,[0,T],u0,t_eval=times,method='RK23',atol=1.e-3,rtol=1.e-3)
    S = solution.y[0,:]
    I = solution.y[1,:]
    R = solution.y[2,:]
    
    return S, I, R


def infer_initial_data(cum_deaths,data_start,ifr,gamma,N):
    """
    Given a sequence of cumulative deaths, infer current values of S, I, and R
    for a population.  The inference dates are offset (backward) from
    the input time series by the mean time to death.

    It is assumed that for each death on day n, there were n/ifr new infections
    on day n-mttd.

    Inputs:
        - cum_deaths: time series of cumulative deaths
        - data_start: starting date of time series
        - ifr: infected fatality ratio
        - gamma: SIR model parameter (1/(time to recovery))
        - N: population size
    """
    daily_deaths = np.diff(cum_deaths); daily_deaths = np.insert(daily_deaths,0,cum_deaths[0])
    mttd = int(round(get_mttd(daily_deaths)))

    inferred_data_dates = np.arange(data_start-mttd,data_start+len(cum_deaths))
    cum_deaths = np.insert(cum_deaths,0,[0]*mttd)
    
    inf_daily_infections = np.zeros_like(inferred_data_dates)
    cum_recovered = np.zeros_like(inferred_data_dates)
    
    inf_daily_infections[:-mttd] = daily_deaths/ifr  # Inferred new infections each day

    for i in range(len(inferred_data_dates)):
        cum_recovered[i] = np.sum(inf_daily_infections[:i]*(1-np.exp(-gamma*(i-np.arange(i)))))
    active_infections = np.cumsum(inf_daily_infections) - cum_recovered
    
    
    # Initial values, mttd days ago
    I0 = active_infections[-(mttd+1)]
    R0 = cum_recovered[-(mttd+1)]
    u0 = np.array([N-I0-R0,I0,R0])
    return u0, mttd, inferred_data_dates


def forecast(u0,lag,N,inferred_data_dates,cum_deaths,ifr=0.007,beta=0.25,gamma=0.05,q=0.,intervention_start=0,
             intervention_length=30,forecast_length=14,compute_interval=True):
    """Forecast with SIR model.  All times are in days.

        Inputs:
         - u0: initial data [S, I, R]
         - lag: difference (in days) between today and simulation start
         - ifr: infection fatality ratio
         - intervention_level: one of 'No action', 'Limited action', 'Social distancing',
                'Shelter in place', or 'Full lockdown'.
         - intervention_start: when intervention measure starts, relative to today (can be negative)
         - intervention_length (in days from simulation start)
         - if compute_interval is True, then we simulate with a range of parameter values
           and return the min and max values for each day.
        
        Note that the simulation starts from t=0 in simulation time, but that
        is denoted as t=lag in terms of the inference and prediction dates.

        Also note that the maximum and minimum daily values are not the successive
        differences of the maximum and minimum cumulative values.

        We could change this to just return daily values, since cumulative values
        can be constructed from those.
    """
    S_mean, I_mean, R_mean = SIR(u0, beta=beta, gamma=gamma, N=N, T=lag+forecast_length, q=q,
                                 intervention_start=intervention_start+lag,
                                 intervention_length=intervention_length)
    

    prediction_dates = inferred_data_dates[-(lag+1)]+range(forecast_length+lag+1)
    pred_cumu_deaths = R_mean*ifr
    # Match values for today:
    pred_cumu_deaths = pred_cumu_deaths - (pred_cumu_deaths[lag]-cum_deaths[-1])

    if not compute_interval:
        return prediction_dates, pred_cumu_deaths, None, None, None, None, S_mean

    else:
        S_low, I_low, R_low = S_mean.copy(), I_mean.copy(), R_mean.copy()
        S_high, I_high, R_high = S_mean.copy(), I_mean.copy(), R_mean.copy()
        dd_low = np.diff(R_mean); dd_high = np.diff(R_mean)

        pred_daily_deaths_low = np.diff(R_mean); pred_daily_deaths_high = np.diff(R_mean)
        for dbeta in np.linspace(-0.05,0.1,6):
            for dgamma in np.linspace(-0.02,0.08,6):
                S, I, R= SIR(u0, beta=beta+dbeta, gamma=gamma+dgamma, N=N, T=lag+forecast_length, q=q,
                             intervention_start=intervention_start+lag,
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
        pred_cumu_deaths_low = pred_cumu_deaths_low - (pred_cumu_deaths_low[lag]-cum_deaths[-1])
        pred_cumu_deaths_high = R_high*ifr
        pred_cumu_deaths_high = pred_cumu_deaths_high - (pred_cumu_deaths_high[lag]-cum_deaths[-1])

        pred_daily_deaths_low = pred_daily_deaths_low*ifr; pred_daily_deaths_high = pred_daily_deaths_high*ifr

        return prediction_dates, pred_cumu_deaths, pred_cumu_deaths_low, pred_cumu_deaths_high, pred_daily_deaths_low, pred_daily_deaths_high, S_mean


def forecast2(u0,lag,N,inferred_data_dates,cum_deaths,ifr=0.007,beta=0.25,gamma=0.04,q=[.0],intervention_dates=[0,30],
             forecast_length=14,compute_interval=True):
    """Forecast with SIR model, including multiple intervention periods.  All times are in days.

        Inputs:
         - ifr: infection fatality ratio
         - lag: difference (in days) between today and simulation start
         - intervention_level: one of 'No action', 'Limited action', 'Social distancing',
                'Shelter in place', or 'Full lockdown'.
         - intervention_start: when intervention measure starts, relative to today (can be negative)
         - intervention_length (in days from simulation start)
    """

    intervention_dates = np.array(intervention_dates)+lag
    
    # Now run the model
    S_mean, I_mean, R_mean = SIR2(u0, beta=beta, gamma=gamma, N=N, T=lag+forecast_length, q=q,intervention_dates=intervention_dates)
    
    prediction_dates = inferred_data_dates[-(lag+1)]+range(forecast_length+lag+1)
    pred_cumu_deaths = R_mean*ifr
    pred_cumu_deaths = pred_cumu_deaths - (pred_cumu_deaths[lag]-cum_deaths[-1])

    if not compute_interval:
        return prediction_dates, pred_cumu_deaths, None, None, None, None

    else:
        S_low, I_low, R_low = S_mean.copy(), I_mean.copy(), R_mean.copy()
        S_high, I_high, R_high = S_mean.copy(), I_mean.copy(), R_mean.copy()
        dd_low = np.diff(R_mean); dd_high = np.diff(R_mean)

        pred_daily_deaths_low = np.diff(R_mean); pred_daily_deaths_high = np.diff(R_mean)
        for dbeta in np.linspace(-0.05,0.05,6):
            for dgamma in np.linspace(-0.02,0.02,6):
                S, I, R= SIR(u0, beta=beta+dbeta, gamma=gamma+dgamma, N=N, T=lag+forecast_length, q=q,
                             intervention_start=intervention_start+lag,
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
        pred_cumu_deaths_low = pred_cumu_deaths_low - (pred_cumu_deaths_low[lag]-cum_deaths[-1])
        pred_cumu_deaths_high = R_high*ifr
        pred_cumu_deaths_high = pred_cumu_deaths_high - (pred_cumu_deaths_high[lag]-cum_deaths[-1])

        pred_daily_deaths_low = pred_daily_deaths_low*ifr; pred_daily_deaths_high = pred_daily_deaths_high*ifr

        return prediction_dates, pred_cumu_deaths, pred_cumu_deaths_low, pred_cumu_deaths_high, pred_daily_deaths_low, pred_daily_deaths_high


def plot_forecast(inferred_data_dates, cum_deaths, lag, prediction_dates, pred_cumu_deaths, pred_cumu_deaths_low,
                  pred_cumu_deaths_high, pred_daily_deaths_low, pred_daily_deaths_high, plot_title,
                  plot_past_pred=True, plot_type='cumulative',
                  plot_interval=True, plot_value='deaths',scale='linear'):

    if scale == 'linear':
        plotfun = plt.plot_date
    else:
        plotfun = plt.semilogy
        
    if plot_past_pred: pred_start_ind=0
    else: pred_start_ind = lag

    if plot_type=='cumulative':
        if plot_value == 'deaths':
            plotfun(inferred_data_dates,cum_deaths,'-',lw=3,label='Deaths (recorded)')
            plotfun(prediction_dates[pred_start_ind:],pred_cumu_deaths[pred_start_ind:],'-k',label='Deaths (predicted)')
            if plot_interval:
                plt.fill_between(prediction_dates[lag:],pred_cumu_deaths_low[lag:],pred_cumu_deaths_high[lag:],color='grey',zorder=-1)
        elif plot_value == 'hospitalizations':
            plotfun(prediction_dates[lag:],pred_cumu_deaths[lag:]*hr,'-k',label='Hospitalizations (predicted)')
            if plot_interval:
                plt.fill_between(prediction_dates[lag:],pred_cumu_deaths_low[lag:]*hr,pred_cumu_deaths_high[lag:]*hr,color='grey',zorder=-1)
    elif plot_type=='daily':
        if plot_value == 'deaths':
            plotfun(inferred_data_dates[1:],np.diff(cum_deaths),'-',lw=3,label='Deaths (recorded)')
            plotfun(prediction_dates[pred_start_ind+1:],np.diff(pred_cumu_deaths[pred_start_ind:]),'-k',label='Deaths (predicted)')
            if plot_interval:
                plt.fill_between(prediction_dates[lag+1:],pred_daily_deaths_low[lag:],pred_daily_deaths_high[lag:],color='grey',zorder=-1)
        elif plot_value == 'hospitalizations':
            plotfun(prediction_dates[lag+1:],np.diff(pred_cumu_deaths[lag:])*hr,'-k',label='Hospitalizations (predicted)')
            if plot_interval:
                plt.fill_between(prediction_dates[lag+1:],pred_daily_deaths_low[lag:]*hr,pred_daily_deaths_high[lag:]*hr,color='grey',zorder=-1)

    plt.legend(loc='best')


    ax = plt.gca()
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.title(plot_title)


def compute_and_plot(region='Spain',ifr=0.7,beta=0.25,gamma=0.04,intervention_level='No action',
             intervention_start=0,intervention_length=30,forecast_length=14,scale='linear',
             plot_type='cumulative',plot_value='deaths',plot_past_pred=True,plot_interval=True):

    ifr = ifr/100.

    N = data.population[region]
    data_dates, total_cases, cum_deaths = data.load_time_series(region)
    data_start = mdates.date2num(data_dates[0])  # First day for which we have data

    u0, mttd, inferred_data_dates = infer_initial_data(cum_deaths,data_start,ifr,gamma,N)
    cum_deaths = np.insert(cum_deaths,0,[0]*mttd)

    q = data.intervention_strength[intervention_level]

    prediction_dates, pred_cumu_deaths, pred_cumu_deaths_low, \
      pred_cumu_deaths_high, pred_daily_deaths_low, pred_daily_deaths_high, S \
      = forecast(u0,mttd,N,inferred_data_dates,cum_deaths,ifr,beta,gamma,q,intervention_start,intervention_length,forecast_length,plot_interval)

    plot_title = '{} {}-day forecast with {} for {} days'.format(region,forecast_length,intervention_level,intervention_length)
    plot_forecast(inferred_data_dates, cum_deaths, mttd, prediction_dates, pred_cumu_deaths, pred_cumu_deaths_low,
                  pred_cumu_deaths_high, pred_daily_deaths_low, pred_daily_deaths_high,
                  plot_title, plot_past_pred=plot_past_pred, plot_type=plot_type,
                  plot_interval=plot_interval, plot_value=plot_value, scale=scale)


def write_JSON(regions, forecast_length=14, print_estimates=False):

    output = {}
    ifr = 0.7/100
    gamma = 0.05
    beta = 0.25

    for region in regions:
        
        # These should be adjusted for each region:
        intervention_level='No action'
        intervention_start=0
        intervention_length=30

        N = data.population[region]
        data_dates, total_cases, cum_deaths = data.load_time_series(region)
        data_start = mdates.date2num(data_dates[0])  # First day for which we have data
        if cum_deaths[-1]<50: continue

        u0, mttd, inferred_data_dates = infer_initial_data(cum_deaths,data_start,ifr,gamma,N)
        cum_deaths = np.insert(cum_deaths,0,[0]*mttd)

        q, q_interval = assess_intervention_effectiveness(region)

        apparent_R = (1-q)*beta/gamma

        q = min(q,1)
        apparent_R = max(apparent_R,0)

        intervention_length=forecast_length*2
        intervention_start = -mttd*2

        prediction_dates, pred_cumu_deaths, pred_cumu_deaths_low, \
          pred_cumu_deaths_high, pred_daily_deaths_low, pred_daily_deaths_high, \
          S = forecast(u0,mttd,N,inferred_data_dates,cum_deaths,ifr,beta,gamma,
                     q,intervention_start,intervention_length,forecast_length,compute_interval=True)
        
        pred_daily_deaths = np.diff(pred_cumu_deaths);
        estimated_immunity = (N-S[mttd])/N

        if print_estimates:
            print('{:>15}: {:.2f} {:.2f} {:.3f}'.format(region,q,apparent_R, estimated_immunity))

        from datetime import datetime
        formatted_dates = [datetime.strftime(mdates.num2date(ddd),"%m/%d/%Y") for ddd in prediction_dates[mttd+1:]]

        output[region] = {}
        output[region]['dates'] = formatted_dates
        output[region]['deaths'] = pred_daily_deaths[mttd:]
        output[region]['deaths_low'] = pred_daily_deaths_low[mttd:]
        output[region]['deaths_high'] = pred_daily_deaths_high[mttd:]
        output[region]['intervention effectiveness'] = q
        output[region]['intervention effectiveness interval'] = q_interval
        output[region]['estimated immunity'] = estimated_immunity
        
    with open('forecast.json', 'w') as file:
        json.dump(output, file, cls=NumpyEncoder)


def assess_intervention_effectiveness(region, plot_result=False):
    ifr = 0.7/100

    beta = 0.25
    gamma = 0.05

    N = data.population[region]
    data_dates, total_cases, cum_deaths = data.load_time_series(region)
    data_start = mdates.date2num(data_dates[0])  # First day for which we have data

    u0, lag, inferred_data_dates = infer_initial_data(cum_deaths,data_start,ifr,gamma,N)
    cum_deaths = np.insert(cum_deaths,0,[0]*lag)

    intervention_start=-lag # Could just set -infinity
    intervention_length=lag
    forecast_length=0

    def fit_q(q):
        prediction_dates, pred_cumu_deaths, pred_cumu_deaths_low, \
          pred_cumu_deaths_high, pred_daily_deaths_low, pred_daily_deaths_high, S \
          = forecast(u0,lag,N,inferred_data_dates,cum_deaths,ifr,beta,gamma,q,
                     intervention_start,intervention_length,forecast_length,False)

        log_daily_deaths = np.log(np.maximum(np.diff(cum_deaths)[-lag:],1.e-1))
        residual = np.linalg.norm(np.log(np.diff(pred_cumu_deaths))-log_daily_deaths)
        return residual

    q = optimize.fsolve(fit_q,0.)[0]

    if plot_result:
        prediction_dates, pred_cumu_deaths, pred_cumu_deaths_low, \
          pred_cumu_deaths_high, pred_daily_deaths_low, pred_daily_deaths_high \
          = forecast(u0,lag,N,inferred_data_dates,cum_deaths,ifr,beta,gamma,q,
                     intervention_start,intervention_length,forecast_length,False)

        plt.semilogy(prediction_dates[1:],np.diff(pred_cumu_deaths))
        plt.semilogy(prediction_dates[1:],np.diff(cum_deaths[-lag-1:]))

    return q, lag

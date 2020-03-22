import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from ipywidgets import interact, widgets
import matplotlib.dates as dates
from scipy.integrate import solve_ivp
import pandas as pd
from IPython.display import Image
plt.style.use('seaborn-poster')
matplotlib.rcParams['figure.figsize'] = (10., 6.)

url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv"
cases = pd.read_csv(url)
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv"
recovered = pd.read_csv(url)
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv"
deaths = pd.read_csv(url)
today = cases.columns[-1]
days = pd.date_range(start='1/22/20',end=today)
dd = np.arange(len(days))

def load_cases(region):
    if region == 'World':
        rows = list(range(len(cases.index)))
    elif region == 'US':
        tophalf = cases.iloc[:200]  # In lower part of file, duplicate data is given for cities
        rows = tophalf['Country/Region'].isin([region])
    elif region in ['Hubei']:
        rows = cases['Province/State'].isin([region])
    else:
        rows = cases['Country/Region'].isin([region])
        
    if region == 'US':
        total_cases = [cases.iloc[:200][day.strftime('%-m/%-d/%y')][rows].sum() for day in days]
        total_recovered = [recovered.iloc[:200][day.strftime('%-m/%-d/%y')][rows].sum() for day in days]
        total_deaths = [deaths.iloc[:200][day.strftime('%-m/%-d/%y')][rows].sum() for day in days]        
    else:
        total_cases = [cases[day.strftime('%-m/%-d/%y')][rows].sum() for day in days]
        total_recovered = [recovered[day.strftime('%-m/%-d/%y')][rows].sum() for day in days]
        total_deaths = [deaths[day.strftime('%-m/%-d/%y')][rows].sum() for day in days]
    return np.array(total_cases), np.array(total_recovered), np.array(total_deaths)

population = {
    'World' : 7.77e9,
    'Austria': 8.822e6,
    'France': 66.99e6,
    'Germany': 82.79e6,
    'Korea, South': 51.47e6,
    'Italy' : 60.48e6,
    'Netherlands': 17.18e6,
    'Spain' : 46.66e6,
    'Switzerland': 8.57e6,
    'United Kingdom': 66.44e6,
    'US' : 372.2e6    
}

def SIR_mitigated(region='world', start_date=today, beta=0.25, gamma=0.05,\
                  confirmed=25, critical=10, fatal=2,
                  use_mitigation=False,
                  mitigation_factor=0.5, mitigation_interval=[0,180],
                  plotS=True,plotI=True,plotR=True,
                  Axis='Linear'):
    """ Model the current outbreak using the SIR model."""

    total_cases, total_recovered, total_deaths = load_cases(region)
    active_confirmed = total_cases - total_recovered - total_deaths
    confirmed_fraction = confirmed/100.
    N = population[region]
    
    du = np.zeros(3)
    u0 = np.zeros(3)
    
    def f(t,u):
        if mitigation_interval[0]<t<mitigation_interval[1] and use_mitigation:
            qval = mitigation_factor
        else:
            qval = 1.
        du[0] = -qval*beta*u[1]*u[0]/N
        du[1] = qval*beta*u[1]*u[0]/N - gamma*u[1]
        du[2] = gamma*u[1]
        return du

    # Initial values
    u0[2] = (total_recovered[-1]+total_deaths[-1])/confirmed_fraction  # Initial recovered
    u0[1] = active_confirmed[-1]/confirmed_fraction-u0[2] # Initial infected
    u0[0] = N - u0[1] - u0[2]

    T = 400
    times = np.arange(0,T)
    solution = solve_ivp(f,[0,T],u0,t_eval=times,method='RK23',rtol=1.e-3,atol=1.e-3)
    S = solution.y[0,:]
    I = solution.y[1,:]
    R = solution.y[2,:]

    if Axis == 'Linear': 
        plotfun = plt.plot_date
        scale = 1.e6
        ylabel = 'Individuals (in millions)'
    elif Axis =='Logarithmic': 
        plotfun = plt.semilogy
        scale = 1.
        ylabel = 'Individuals'
    
    start = dates.datestr2num(start_date)
    mydates = np.arange(T)+start
    
    fig = plt.figure(figsize=(12,8))
    if plotS:
        plotfun(mydates,S/scale,'-b',lw=3,label='Susceptible')
    if plotI:
        plotfun(mydates,I/scale,'-',color='brown',lw=3,label='Infected')
        plotfun(mydates,I*confirmed/100./scale,'-',lw=3,label='Active confirmed')
        plotfun(mydates,I*critical/100./scale,'-',lw=3,label='Critical')
        plotfun(days,total_cases/scale,'.k',label='Total Confirmed (data)')
    if plotR:
        plotfun(mydates,R*(100-fatal)/100/scale,'-g',lw=3,label='Recovered')
        plotfun(mydates,R*fatal/100./scale,'-',lw=3,label='Deaths')
    ax = plt.gca()
    ax.xaxis.set_major_locator(dates.MonthLocator())
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b'))
    fig.autofmt_xdate()
    plt.legend()
    plt.ylabel(ylabel)
    plt.xlim(start-60,start+T)
    plt.ylim(-N/10/scale,N/scale)
    plt.title(region);
    plt.savefig('temp.png')
    return fig, S, I, R, start

mystyle = {'description_width':'initial'}

from ipywidgets import interact, interactive, widgets, Box, Layout

widget_layout = Layout(display='flex',
                       flex_flow='row',
                       justify_content='space-between')

region_w = widgets.Dropdown(options=population.keys(),value='World',description='Region to model:',style=mystyle)
beta_w = widgets.FloatSlider(min=0.01,max=0.5,step=0.01,value=0.25,description=r'$\beta$ (rate of contact)',style=mystyle)
gamma_w = widgets.FloatSlider(min=0.01,max=0.5,step=0.01,value=0.05,description=r'$\gamma$ (rate of recovery)',style=mystyle)
critical_w = widgets.FloatSlider(min=0.01,max=100.,step=0.1,value=10.,
                                        description=r'% of cases critical',style=mystyle)
fatal_w = widgets.FloatSlider(min=0.1,max=100.,step=0.1,value=2.,
                                        description=r'% of cases fatal',style=mystyle)
confirmed_w = widgets.IntSlider(min=1,max=100,step=1,value=50,
                                        description=r'% of cases confirmed',style=mystyle)
mitigation_factor_w = widgets.FloatSlider(min=0.01, max=1.0, step=0.01, value=0.5,style=mystyle,
                                         description='Mitigation Factor')
mitigation_interval_w = widgets.IntRangeSlider(min=0, max=400, step=5, value=(0,180),style=mystyle,
                                              description='Mitigation Interval')
mitigation_enabled_w = widgets.Checkbox(value=False,description='Use mitigation')

Axis_w = widgets.RadioButtons(options=['Linear','Logarithmic'])
plotS_w = widgets.Checkbox(value=False,description='Plot S')
plotI_w = widgets.Checkbox(value=True,description='Plot I')
plotR_w = widgets.Checkbox(value=False,description='Plot R')

stats_w = widgets.Output(style={'border':True})
plot_w = widgets.Output()

model_column1 = widgets.VBox([region_w, beta_w, gamma_w, confirmed_w, critical_w, fatal_w],
                             layout=Layout(display='flex', flex_flow='column',
                             align_items='stretch',width='40%'))

mitigation_column1 = widgets.VBox([mitigation_enabled_w, mitigation_factor_w, mitigation_interval_w,stats_w],
                             layout=Layout(display='flex', flex_flow='column',
                             align_items='stretch',width='50%'))

model_tab = widgets.VBox([widgets.HBox([model_column1,mitigation_column1],layout=Layout(display='flex',
                                       align_items='stretch',height='200px')),plot_w])


mitigation_tab = widgets.VBox([widgets.HBox([mitigation_column1],layout=Layout(display='flex',
                                       align_items='stretch',height='200px')),plot_w])

plotting_tab = widgets.VBox([widgets.VBox([plotS_w,plotI_w,plotR_w,Axis_w],
                            layout=Layout(display='flex',
                                          align_items='stretch',width='50%',height='200px')),plot_w])

stats_tab = widgets.VBox([widgets.VBox([stats_w],layout=Layout(display='flex',
                                          align_items='stretch',width='50%',height='200px')),plot_w])

SIR_gui = widgets.Tab(children=[model_tab, plotting_tab, stats_tab])
SIR_gui.set_title(0,'Model')
SIR_gui.set_title(1,'Plotting')
SIR_gui.set_title(2,'Statistics')

def SIR_output(region='world', start_date=today, beta=0.25, gamma=0.05,\
                  confirmed=25, critical=5, fatal=1, use_mitigation=False,
                  mitigation_factor=0.5, mitigation_interval=[0,180],
                  plotS=True,plotI=True,plotR=True,
                  Axis='Linear'):
    
    plot_w.clear_output(wait=True)
    stats_w.clear_output(wait=True)
    
    fig, S, I, R, start = SIR_mitigated(region, start_date, beta, gamma,
                  confirmed, critical, fatal, use_mitigation,
                  mitigation_factor, mitigation_interval,
                  plotS, plotI, plotR, Axis)

    with plot_w:
        plt.show(fig)
    
    I_max, I_max_date, I_total = np.max(I), start+np.argmax(I), R[-1]
    
    with stats_w:
        print('Date of infection peak:                 {}'.format(dates.num2date(I_max_date).strftime('%-m/%-d/%y')))
        print('Maximum simultaneous infections: {:12.2f} million'.format(I_max/1e6))
        print('Maximum simultaneous critical cases: {:8.2f} million'.format(I_max/1e6*critical/100))
        print('Total infected: {:29.0f} million'.format(I_total/1e6))
        print('Total deaths: {:33.0f}'.format(I_total*fatal/100.))

SIR_widget = widgets.interactive_output(SIR_output,{'region':region_w,'beta':beta_w,
                                               'gamma':gamma_w, 'confirmed':confirmed_w,
                                               'critical':critical_w, 'fatal':fatal_w,
                                               'use_mitigation':mitigation_enabled_w,
                                               'mitigation_factor':mitigation_factor_w,
                                               'mitigation_interval':mitigation_interval_w,
                                               'Axis':Axis_w,
                                               'plotS':plotS_w,'plotI':plotI_w,'plotR':plotR_w
                                                });

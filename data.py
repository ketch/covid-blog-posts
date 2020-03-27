import pandas as pd
import numpy as np

def jhu_data():
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    cases = pd.read_csv(url)
    # No more recovered data:
#    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv"
    #recovered = pd.read_csv(url)
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
    deaths = pd.read_csv(url)
    today = cases.columns[-1]
    days = pd.date_range(start='1/22/20',end=today)
    return cases, deaths, today, days

def load_cases(region):

    cases, deaths, today, days = jhu_data()

    if region == 'World':
        rows = list(range(len(cases.index)))
    elif region in ['Hubei']:
        rows = cases['Province/State'].isin([region])
    else:
        rows = cases['Country/Region'].isin([region])
        
    total_cases = [cases[day.strftime('%-m/%-d/%y')][rows].sum() for day in days]
    total_deaths = [deaths[day.strftime('%-m/%-d/%y')][rows].sum() for day in days]
    return np.array(total_cases), np.array(total_deaths)

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


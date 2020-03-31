import pandas as pd
import numpy as np
import json
import os
import time

def jhu_data(remote=False):
    """Fetch data from JHU set on Github and partially parse it.

       Inputs:
            - remote: if True, pull files from Github.  Otherwise, use local copies.

       Outputs:
            - cases_df: # of confirmed cases (pandas dataframe)
            - deaths_df: # of confirmed deaths (pandas dataframe)
            - data_dates: full range of dates of data (human-readable format)
    """
    if remote:
        url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    else:
        url = "/Users/ketch/Research/Projects/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
        # Check that file has been updated today
        timestamp = os.path.getmtime(url)
        if (time.time()-timestamp)/3600 > 12:
            print('Warning: data file is more than 12 hours old.  Please update.')

    cases_df = pd.read_csv(url)
    # No more recovered data:
#    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv"
    #recovered = pd.read_csv(url)
    if remote:
        url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
    else:
        url = "/Users/ketch/Research/Projects/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
    deaths_df = pd.read_csv(url)
    today = cases_df.columns[-1]
    data_dates = pd.date_range(start='1/22/20',end=today)
    return cases_df, deaths_df, data_dates

def load_time_series(region):
    """
    Fetch data and parse to get time series of confirmed cases and deaths.
    For now, only works with JHU data.
    """

    cases_df, deaths_df, data_dates = jhu_data()

    #if region in US_state_population.keys():
    #    postal_code = us_state_abbrev[region]
    #    state_str = postal_code+', USA'
    #    with open('/Users/ketch/Downloads/timeseries-byLocation.json') as file:
    #        state_data = json.load(file)
    #    for date in data_dates:
    #        print(date, state_data[state_str]['dates'][date.strftime('%Y-%-m-%-d')].keys())
    #    cum_deaths = [state_data[state_str]['dates'][date.strftime('%Y-%-m-%-d')]['deaths_df'] for date in data_dates]
    #    cum_cases  = [state_data[state_str]['dates'][date.strftime('%Y-%-m-%-d')]['cases_df'] for date in data_dates]
    #    
    #else:
    rows = cases_df['Country/Region'].isin([region])
    cum_cases = [cases_df[day.strftime('%-m/%-d/%y')][rows].sum() for day in data_dates]
    if not np.any(rows==True):
        raise(Exception)
        
    rows = deaths_df['Country/Region'].isin([region])
    cum_deaths = [deaths_df[day.strftime('%-m/%-d/%y')][rows].sum() for day in data_dates]
    return data_dates, np.array(cum_cases), np.array(cum_deaths)

population_smallset = {
    'Austria': 8.822e6,
    'France': 66.99e6,
    'Germany': 82.79e6,
    'Korea, South': 51.47e6,
    'Italy' : 60.48e6,
    'Netherlands': 17.18e6,
    'Spain' : 46.66e6,
    'Switzerland': 8.57e6,
    'United Kingdom': 66.44e6,
    'US' : 372.2e6,
    'Canada': 37.59e6,
    'Belgium': 11.4e6,
    'Denmark': 5.603e6,
    'Sweden': 10.12e6,
    'Brazil': 209.3e6,
    'Australia': 24.6e6,
    'Saudi Arabia': 34813.867e3,
}

intervention_strength = {
    'No action': 0.,
    'Limited action': 0.1,
    'Social distancing': 0.35,
    'Shelter in place': 0.6,
    'Full lockdown': 0.85
}

# Values below are taken from https://population.un.org/wpp/Download/Standard/CSV/.
population = {
    'Afghanistan': 38928.341e3,
    'Albania': 2877.8e3,
    'Algeria': 43851.043e3,
    'Andorra': 77.265e3,
    'Angola': 32866.268e3,
    'Antigua and Barbuda': 97.928e3,
    'Argentina': 45195.777e3,
    'Armenia': 2963.2340000000004e3,
    'Australia': 25499.881e3,
    'Austria': 9006.4e3,
    'Azerbaijan': 10139.175e3,
    'Bahamas': 393.24800000000005e3,
    'Bahrain': 1701.5829999999999e3,
    'Bangladesh': 164689.383e3,
    'Barbados': 287.371e3,
    'Belarus': 9449.321e3,
    'Belgium': 11589.616000000002e3,
    'Belize': 397.621e3,
    'Benin': 12123.198e3,
    'Bhutan': 771.612e3,
    'Bolivia': 11673.028999999999e3,
    'Bosnia and Herzegovina': 3280.815e3,
    'Brazil': 212559.40899999999e3,
    'Bulgaria': 6948.445e3,
    'Burkina Faso': 20903.278000000002e3,
    'Cabo Verde': 555.988e3,
    'Cambodia': 16718.971e3,
    'Cameroon': 26545.863999999998e3,
    'Canada': 37742.157e3,
    'Central African Republic': 4829.764e3,
    'Chad': 16425.859e3,
    'Chile': 19116.209e3,
    'China': 1439323.774e3,
    'Colombia': 50882.884000000005e3,
    'Costa Rica': 5094.1140000000005e3,
    'Croatia': 4105.268e3,
    'Cuba': 11326.616000000002e3,
    'Cyprus': 1207.361e3,
    'Czechia': 10708.982e3,
    'Denmark': 5792.203e3,
    'Djibouti': 988.002e3,
    'Dominica': 71.991e3,
    'Dominican Republic': 10847.903999999999e3,
    'Ecuador': 17643.06e3,
    'Egypt': 102334.403e3,
    'El Salvador': 6486.201e3,
    'Equatorial Guinea': 1402.985e3,
    'Eritrea': 3546.427e3,
    'Estonia': 1326.539e3,
    'Eswatini': 1160.164e3,
    'Ethiopia': 114963.58300000001e3,
    'Fiji': 896.444e3,
    'Finland': 5540.718000000001e3,
    'France': 65273.512e3,
    'Gabon': 2225.728e3,
    'Gambia': 2416.6639999999998e3,
    'Georgia': 3989.175e3,
    'Germany': 83783.945e3,
    'Ghana': 31072.945e3,
    'Greece': 10423.056e3,
    'Grenada': 112.51899999999999e3,
    'Guatemala': 17915.567e3,
    'Guinea': 13132.792e3,
    'Guinea-Bissau': 1967.9979999999998e3,
    'Guyana': 786.559e3,
    'Haiti': 11402.533000000001e3,
    'Holy See': 0.809e3,
    'Honduras': 9904.608e3,
    'Hungary': 9660.35e3,
    'Iceland': 341.25e3,
    'India': 1380004.385e3,
    'Indonesia': 273523.62100000004e3,
    'Iran': 83992.95300000001e3,
    'Iraq': 40222.503e3,
    'Ireland': 4937.795999999999e3,
    'Israel': 8655.541e3,
    'Italy': 60461.828e3,
    'Jamaica': 2961.1609999999996e3,
    'Japan': 126476.45800000001e3,
    'Jordan': 10203.14e3,
    'Kazakhstan': 18776.707e3,
    'Kenya': 53771.3e3,
    'Kuwait': 4270.563e3,
    'Kyrgyzstan': 6524.191e3,
    'Laos': 7275.556e3,
    'Latvia': 1886.2020000000002e3,
    'Lebanon': 6825.441999999999e3,
    'Liberia': 5057.677e3,
    'Libya': 6871.286999999999e3,
    'Liechtenstein': 38.137e3,
    'Lithuania': 2722.2909999999997e3,
    'Luxembourg': 625.976e3,
    'Madagascar': 27691.019e3,
    'Malaysia': 32365.998e3,
    'Maldives': 540.5419999999999e3,
    'Mali': 20250.834e3,
    'Malta': 441.539e3,
    'Mauritania': 4649.66e3,
    'Mauritius': 1271.767e3,
    'Mexico': 128932.75300000001e3,
    'Monaco': 39.244e3,
    'Mongolia': 3278.292e3,
    'Montenegro': 628.062e3,
    'Morocco': 36910.558e3,
    'Mozambique': 31255.435e3,
    'Namibia': 2540.916e3,
    'Nepal': 29136.807999999997e3,
    'Netherlands': 17134.873e3,
    'New Zealand': 4822.233e3,
    'Nicaragua': 6624.554e3,
    'Niger': 24206.636000000002e3,
    'Nigeria': 206139.587e3,
    'North Macedonia': 2083.38e3,
    'Norway': 5421.241999999999e3,
    'Oman': 5106.622e3,
    'Pakistan': 220892.331e3,
    'Panama': 4314.768e3,
    'Papua New Guinea': 8947.027e3,
    'Paraguay': 7132.53e3,
    'Peru': 32971.846e3,
    'Philippines': 109581.085e3,
    'Poland': 37846.605e3,
    'Portugal': 10196.707e3,
    'Qatar': 2881.06e3,
    'Korea, South': 51269.183e3,
    'Romania': 19237.682e3,
    'Russia': 145934.46e3,
    'Rwanda': 12952.208999999999e3,
    'Saint Kitts and Nevis': 53.192e3,
    'Saint Lucia': 183.62900000000002e3,
    'Saint Vincent and the Grenadines': 110.947e3,
    'San Marino': 33.938e3,
    'Senegal': 16743.93e3,
    'Serbia': 8737.37e3,
    'Seychelles': 98.34e3,
    'Singapore': 5850.343000000001e3,
    'Slovakia': 5459.643e3,
    'Slovenia': 2078.9320000000002e3,
    'Somalia': 15893.219e3,
    'South Africa': 59308.69e3,
    'Spain': 46754.782999999996e3,
    'Sri Lanka': 21413.25e3,
    'Sudan': 43849.269e3,
    'Suriname': 586.634e3,
    'Sweden': 10099.27e3,
    'Switzerland': 8654.618e3,
    'Syria': 17500.657e3,
    'Thailand': 69799.978e3,
    'Timor-Leste': 1318.442e3,
    'Togo': 8278.737e3,
    'Trinidad and Tobago': 1399.491e3,
    'Tunisia': 11818.618e3,
    'Turkey': 84339.067e3,
    'Uganda': 45741.0e3,
    'Ukraine': 43733.759000000005e3,
    'United Arab Emirates': 9890.4e3,
    'United Kingdom': 67886.004e3,
    'US': 331002.647e3,
    'Uruguay': 3473.7270000000003e3,
    'Uzbekistan': 33469.199e3,
    'Venezuela': 28435.943e3,
    'Vietnam': 97338.58300000001e3,
    'Zambia': 18383.956000000002e3,
    'Zimbabwe': 14862.927e3
}

US_state_population = {
    'Alabama': 4903185,
    'Alaska': 731545,
    'Arizona': 7278717,
    'Arkansas': 3017804,
    'California': 39512223,
    'Colorado': 5758736,
    'Connecticut': 3565287,
    'Delaware': 973764,
    'District of Columbia': 705749,
    'Florida': 21477737,
    'Georgia': 10617423,
    'Hawaii': 1415872,
    'Idaho': 1787065,
    'Illinois': 12671821,
    'Indiana': 6732219,
    'Iowa': 3155070,
    'Kansas': 2913314,
    'Kentucky': 4467673,
    'Louisiana': 4648794,
    'Maine': 1344212,
    'Maryland': 6045680,
    'Massachusetts': 6892503,
    'Michigan': 9986857,
    'Minnesota': 5639632,
    'Mississippi': 2976149,
    'Missouri': 6137428,
    'Montana': 1068778,
    'Nebraska': 1934408,
    'Nevada': 3080156,
    'New Hampshire': 1359711,
    'New Jersey': 8882190,
    'New Mexico': 2096829,
    'New York': 19453561,
    'North Carolina': 10488084,
    'North Dakota': 762062,
    'Ohio': 11689100,
    'Oklahoma': 3956971,
    'Oregon': 4217737,
    'Pennsylvania': 12801989,
    'Rhode Island': 1059361,
    'South Carolina': 5148714,
    'South Dakota': 884659,
    'Tennessee': 6829174,
    'Texas': 28995881,
    'Utah': 3205958,
    'Vermont': 623989,
    'Virginia': 8535519,
    'Washington': 7614893,
    'West Virginia': 1792147,
    'Wisconsin': 5822434,
    'Wyoming': 578759,
    'Puerto Rico': 3193694
}

us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

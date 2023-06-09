import numpy as np

def truncate(num):
    return 1 if num < 1 else round(num)

def introduce_errors(series, mag, loc):
    i = truncate(loc())
    while i < len(series):
        series[i] += mag()
        i += truncate(loc()) 
    return series
 
def normal_dist(mean, std):
    return lambda : np.random.normal(mean, std)
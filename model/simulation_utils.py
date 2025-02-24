# simulation_utils.py
"""
Mostly reference ODE functions
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint


#################################################################
# Reference ODE Functions
#################################################################
def dx_dt(x, t, alpha, beta, gamma, delta):
    s, w = x
    ds_dt = alpha * s - beta * s * w
    dw_dt = -gamma * w + delta * beta * s * w
    return [ds_dt, dw_dt]

def get_reference_ODE(model_params, model_time):
    alpha = model_params['alpha']
    beta = model_params['beta']
    gamma = model_params['gamma']
    delta = model_params['delta']

    t_end = model_time['time']
    times = np.linspace(0, t_end, model_time['tmax'])
    x0 = [model_params['s_start'], model_params['w_start']]

    integration = odeint(dx_dt, x0, times, args=(alpha, beta, gamma, delta)) # via cursor, verify this
    ode_df = pd.DataFrame({
        't': times,
        's': integration[:,0],
        'w': integration[:,1]
    })
    return ode_df

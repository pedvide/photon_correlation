# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 18:52:13 2017

@author: villanueva
"""

import warnings

import numpy as np
from scipy.optimize import curve_fit


def exp_decay(t, A, tau, *others):
    return np.piecewise(t, [t>=0, t<0], [lambda t: 1, lambda t: A*np.exp(t/tau) + 1])

def exp_decay_sigma(cov, t, A, tau, *others):
    left_side = lambda t: np.exp(t/tau)*np.sqrt(cov[0,0] + (cov[0,1]+ cov[1,0])*A*t + cov[1,1]*A**2*t**2)
    return np.piecewise(t, [t>=0, t<0], [lambda t: 0, left_side])

def symmetric_decay(t, A, tau, t0, *others):
    return 1 + A*np.exp(-np.abs(t-t0)/tau)**2

def symmetric_decay_sigma(cov, t, A, tau, t0, *others):
    common_part = (cov[0,0] +
                   cov[1,1]*A**2*np.abs(t-t0)**2/tau**4 +
                   (cov[1,2]+cov[2,1])*A**2*np.abs(t-t0)/tau**3 +
                   cov[2,2]*A**2/tau**2 +
                   (cov[1,0]+cov[0,1])*A*np.abs(t-t0)/tau**2 +
                   (cov[2,0]+cov[0,2])*A/tau)
    abs_exp = np.exp(-np.abs(t-t0)/tau)**2
    return np.sqrt(np.abs(common_part*abs_exp))


def noise_amp_fun(t, A):
    return A*t**(-0.5)

def get_signal_and_noise_from_hist(norm_hist, hist_time, fit_function_name='symmetric',
                                   expected_tau=None, expected_time_offset=None):
    '''Fits the folded histogram to a exponential decay.
    Returns the decay parameters and std, the background mean, and noise.'''
    fit_function = globals()[fit_function_name + '_decay']

    bin_width = hist_time[1] - hist_time[0]
    hist_len = len(norm_hist)
    signal_data = norm_hist
    signal_time = hist_time
    background_data = np.concatenate((signal_data[:hist_len//4], signal_data[-hist_len//4:]))
    background_mean = np.mean(background_data)
    noise = np.std(background_data)

    if fit_function_name is 'symmetric':
        # initial estimate of the anplitude
        amp_p0 = np.max(signal_data[hist_len//2-20:hist_len//2+20])
        # initial estimate of time offset
        if expected_time_offset:
            t0_range = (signal_time[hist_len*25//100], signal_time[hist_len*75//100])
        else:
            expected_time_offset = signal_time[np.argmax(signal_data[hist_len*25//100, hist_len*75//100])]
            t0_range = (signal_time[hist_len*25//100], signal_time[hist_len*75//100])
        # initial estimate of tau
        if expected_tau:
            tau_range = (expected_tau*0.1, expected_tau*10)
        else:
            expected_tau = 10*bin_width
            tau_range = (2*bin_width, 50*bin_width)

        init_params = (amp_p0, expected_tau, expected_time_offset)
        bounds=([0, tau_range[0], t0_range[0]], [2*np.max(signal_data), tau_range[-1], t0_range[-1]])
    elif fit_function_name is 'exp':
        amp_p0 = np.max(signal_data[hist_len//2-20:hist_len//2+20])
        init_params = (amp_p0*2, 20*bin_width)
        bounds=(0, np.inf)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            decay_params, decay_cov = curve_fit(fit_function, signal_time, signal_data,
                                                p0=init_params, bounds=bounds)
        except RuntimeError:
            num_params = len(init_params)
            decay_params = np.zeros(num_params)
            decay_cov = np.zeros((num_params, num_params))
            background_mean = 0
            noise = 0

    return decay_params, decay_cov, background_mean, noise

def get_noise_fit(noise_amplitudes, experimental_time):
    if not np.any(noise_amplitudes>0):
        return 0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            (A, ), p_cov = curve_fit(noise_amp_fun, experimental_time[noise_amplitudes>0], noise_amplitudes[noise_amplitudes>0],
                                     p0=(noise_amplitudes[noise_amplitudes>0][0]*2), bounds=(0, np.inf))
            A_err = np.sqrt(np.diag(p_cov))[0]
        except RuntimeError:
            A = 0
            A_err = 0

    return A, A_err


_si_prefix = {-24: 'y',  # yocto
             -21: 'z',  # zepto
             -18: 'a',  # atto
             -15: 'f',  # femto
             -12: 'p',  # pico
              -9: 'n',  # nano
              -6: 'u',  # micro
              -3: 'm',  # mili
              -2: 'c',  # centi
              -1: 'd',  # deci
               3: 'k',  # kilo
               6: 'M',  # mega
               9: 'G',  # giga
              12: 'T',  # tera
              15: 'P',  # peta
              18: 'E',  # exa
              21: 'Z',  # zetta
              24: 'Y',  # yotta
            }

def get_SI_repr(number, separator=' '):
    float_part, exp_part, unit = get_SI_parts(number)
    return f'{float_part}{separator}{unit}'

def get_SI_parts(number):
    exp_part = np.int64(np.log10(number))
    orders_from_3 = abs(exp_part) % 3
    exp_part -=  np.int64(np.sign(exp_part)*orders_from_3)
    float_part = np.round(number/np.power(10.0, exp_part), 5)
    unit = _si_prefix[exp_part]

    return float_part, exp_part, unit
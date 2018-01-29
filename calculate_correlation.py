# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 18:46:10 2017

@author: villanueva
"""
import time
import logging
import re
import pathlib

import tqdm

import numpy as np

from multiprocessing import Process, Queue

import ptu_parser
from plotter_pyqtgraph import Plotter
from common_util import get_signal_and_noise_from_hist, get_noise_fit, get_SI_repr

# fast
import pyximport; pyximport.install()
from calculate_correlation_continuous_cython import calculate_correlation
# slow
#from calculate_correlation_continuous import calculate_correlation

#### CHANGE THIS ####
bin_width = 0.25e-9  # in seconds
hist_len = 201  # bins
# help the fit by providing an approximate lifetime
expected_tau = 2e-9 # in seconds

filename = r"..\data\CdSe_CdS\Pradip samples\CdSe_2.7nm_CdS_3ML_Exc335nm_LP460nm_001.ptu"
#filename = r"..\data\Pr references\1Pr_NaLaF4_overnight_switched_PMT_position_switched_HV_supply_no_lamp.ptu"
# if series = False, only the filename will be analyzed
# if series = True, a series of files with names filename_NNN.ptu will be analyzed
series = True

# for continuous measurements use a rate of at least 100 s,
# so the total number of updates is not too high (depends on RAM)
PLOT_UPDATE_RATE = 500 # s, of the measurement

# symmetric fits a symmetric decaying exponential, appropiate for measurements with a beam splitter
# exp fits a single decaying exponential, appropiate for experiments with a dichroic filter
fit_function_name = 'symmetric' # 'symmetric' or 'exp'

# the setup has a time offset of about -2 ns
expected_time_offset = -2e-9 # seconds
#####################


path = pathlib.Path(filename)
basename = path.stem
folder = path.parent

plotter = Plotter(hist_len, bin_width, fit_function_name=fit_function_name, title=basename)

# everything inside the if runs only in the main process, not in the plot process
if __name__=='__main__':

    if series:
        # get the part of the filename before the _NNN (if it exists)
        basename = re.split(r'_\d\d\d', basename)[0]
        # get all files with basename_NNN.ptu, where N is an integer
        files = sorted(folder.glob(basename + '_[0-9][0-9][0-9].ptu'))
    else:
        files = [filename]

    binwidth_repr = get_SI_repr(bin_width, separator='')

    logger = logging.getLogger(__name__)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    # create console handler and set level to INFO
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create file handler which logs even debug messages

    fh = logging.FileHandler(folder.joinpath(f'correlation_{basename}_{binwidth_repr}s.log'), mode='w')
    fh.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch and fh
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add ch and fh to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    # starting of script
    start_time = time.time()
    logger.info("Script started.")

    # Start plot as a new process
    plot_queue = Queue()
    process = Process(target=plotter, args=(plot_queue,), daemon=True, name='Plot')
    process.start()

    # set variables to zero
    current_time = 0
    synccnt = 0
    inputcnt = 0
    total_time = 0
    hist = np.zeros((hist_len, ))
    amplitudes = []
    hist_time = (np.linspace(-hist_len//2+1, hist_len//2, num=hist_len)-1)*bin_width # time in s
    last_time = 0
    last_synccnt = 0
    last_inputcnt = 0
    last_hist = np.zeros((hist_len, ))

    total_num_events = 0
    total_exp_time = 0
    logger.info(f'Files to analyze in folder "{folder}":')
    for file in files:
        logger.info(f'{pathlib.Path(file).stem}.ptu.')
        # get header with all the information about the file
        header = ptu_parser.parse_header(file)
        total_num_events += header['TTResult_NumberOfRecords']
        if total_num_events != 0:
            total_exp_time += header['TTResult_StopAfter']*1e-3  # in s
        resolution = header['MeasDesc_GlobalResolution']

    logger.info("Total number of events (0 if running): {}.".format(total_num_events))
    logger.info("Resolution: {} s.".format(resolution))
    logger.info("Total measurement time: {:.2g} s.".format(total_exp_time))
    logger.info(f"Bin width: {binwidth_repr}s.")

    pbar = tqdm.tqdm(desc='Calculating correlation', unit='exp. s', total=int(np.ceil(total_exp_time)))

    for file in files:
        pbar.write("Current filename: {}.".format(file))

        # get header with all the information about the file
        header = ptu_parser.parse_header(file)

        last_time = total_time
        last_synccnt = synccnt
        last_inputcnt = inputcnt
        last_hist = hist

        # prepare reading data file
        with open(file, "rb") as data_file:
            # calculate correlation
            for (temp_hist, current_time,
                 new_synccnt, new_inputcnt) in calculate_correlation(data_file, header,
                                                                     bin_width, hist_len,
                                                                     plot_update_rate=PLOT_UPDATE_RATE):
                total_time = current_time + last_time
                pbar.update(int(total_time - pbar.n))

                hist = temp_hist + last_hist
                synccnt = new_synccnt + last_synccnt
                inputcnt = new_inputcnt + last_inputcnt
                norm_factor = synccnt * inputcnt / total_time * bin_width
                norm_hist =  hist/norm_factor

                (decay_params, decay_cov,
                 noise_mean, noise_std) = get_signal_and_noise_from_hist(norm_hist, hist_time,
                                                                         fit_function_name, expected_tau,
                                                                         expected_time_offset)

                decay_error = np.sqrt(np.diag(decay_cov))
                amplitudes.append((np.float64(total_time), decay_params[0], decay_error[0], noise_mean, noise_std))

                plot_queue.put((norm_hist, decay_params, decay_cov, total_time, synccnt, inputcnt))

    pbar.close()

    full_hist = np.array(np.column_stack((hist_time, norm_hist)))
    signal_noise_amplitudes = np.array(amplitudes)

    # Get final signal amplitude and standard deviation, and noise amplitude
    decay_params, decay_cov, background_mean, noise_amp = get_signal_and_noise_from_hist(norm_hist, hist_time,
                                                                                         fit_function_name=fit_function_name,
                                                                                         expected_tau=expected_tau,
                                                                                         expected_time_offset=expected_time_offset)

    decay_error = np.sqrt(np.diag(decay_cov))
    signal_amplitude = decay_params[0]
    signal_amplitude_std = decay_error[0]
    signal_decay_tau = decay_params[1]
    signal_decay_tau_std = decay_error[1]
    if len(decay_params) > 2:
        signal_time_offset = decay_params[2]
        signal_time_offset_std = decay_error[2]

    experimental_time = signal_noise_amplitudes[:, 0]
    noise_amplitudes = signal_noise_amplitudes[:, 4]

    noise_A, noise_A_std = get_noise_fit(noise_amplitudes, experimental_time)
    if noise_amplitudes[-1] < signal_amplitude:
        crossing_time = (noise_A/signal_amplitude)**2
        crossing_time_std = 2*crossing_time*np.sqrt((signal_amplitude_std/signal_amplitude)**2 + (noise_A_std/noise_A)**2)
    else:
        crossing_time = crossing_time_std = 0.0

    # log some info to file
    logger.info("Counts on SYNC: {}.".format(synccnt))
    logger.info("Counts on INPUT: {}.".format(inputcnt))
    logger.info("Average counts on SYNC: {:.0f} cps.".format(synccnt/total_time))
    logger.info("Average counts on INPUT: {:.0f} cps.".format(inputcnt/total_time))
    logger.info("Ratio SYNC/INPUT: {:.1f}.".format(synccnt/inputcnt))
    logger.info("Signal amplitude: {:.3g}\u00B1{:.2g}.".format(signal_amplitude, signal_amplitude_std))
    logger.info("Signal decay lifetime: {:.3g}\u00B1{:.1g} s.".format(signal_decay_tau, signal_decay_tau_std))
    logger.info("Background mean residual\u00B1noise: {:.2g}\u00B1{:.2g}.".format(background_mean-1, noise_amp))
    logger.info("Noise fit parameter: {:.3g}\u00B1{:.2g}.".format(noise_A, noise_A_std))
    if len(decay_params) > 2:
        logger.info("Signal fit time offset: {:.3g}\u00B1{:.3g} s.".format(signal_time_offset, signal_time_offset_std))
    logger.info("Crossing time: {:.3g}\u00B1{:.2g} s.".format(crossing_time, crossing_time_std))
    logger.info('Experiment time (header) {:.2g} s ({:.2g} s).'.format(total_time, total_exp_time))

    # write results to file
    with open(folder.joinpath(f'histogram_{basename}_{binwidth_repr}s.txt'), 'wt') as output_file:
        output_file.write('# Filename: {}\n'.format(filename))
        output_file.write('# delay (seconds)\tnormalized histogram\n')
        for bin_time, hist_count in full_hist:
            output_file.write("{:e}\t{:f}\n".format(bin_time, hist_count))
    # write results to file
    with open(folder.joinpath(f'signal_and_noise_{basename}_{binwidth_repr}s.txt'), 'wt') as output_file:
        output_file.write('# Filename: {}\n'.format(filename))
        output_file.write('# Signal time (s)\tSignal amplitude\tSignal STD\tBackground level\tNoise Amplitude\n')
        for signal_time, signal_amp, signal_std, noise_mean, noise_std in signal_noise_amplitudes:
            output_file.write("{:e}\t{:f}\t{:f}\t{:f}\t{:f}\n".format(signal_time, signal_amp, signal_std, noise_mean, noise_std))

    # finish script
    logger.info("Analysis time: {:.1f} s.".format(time.time() - start_time))
    logging.shutdown()
    for handler in logger.handlers:
        logger.removeHandler(handler)

    input('Press any key to continue.')
    plot_queue.put(None)  # put None to close the plot window
    process.join()

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 22:19:25 2017

@author: Villanueva
"""
import numpy as np
import warnings
#import sys
#import time

import common_util
from common_util import noise_amp_fun, get_noise_fit

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

import matplotlib.pyplot as plt
#import matplotlib.transforms as mtransforms
plt.ion()

import queue

# histogram data is black cirles
DATA_COLOR = 'k'
DATA_MARKER = '.'
DATA_LINESTYLE = ''
# fit data and signal amplitude are red
FIT_COLOR = 'r'
FIT_LINESTYLE = '-'
FIT_ERROR_COLOR = (1.0, 0.0, 0.0, 0.5)  # red, alpha
# background and noise are blue
NOISE_COLOR = (0.0, 0.0, 1.0)  # blue

class Plotter:

    def __init__(self, hist_len, bin_width, fit_function_name='symmetric'):
        self.hist_len = hist_len
        self.bin_width = bin_width
        self.fit_function_name = fit_function_name

    def setup_plots(self):
        # setup the plots
        self.fig, (self.ax_histo, self.ax_signal_noise, self.ax_cps) = plt.subplots(nrows=3, ncols=1)

        # increase the height
        width, height = tuple(self.fig.get_size_inches())
        self.fig.set_size_inches(width, height*1.5)

        self.setup_plot_histogram(self.hist_len, self.bin_width, self.fit_function_name)
        self.setup_plot_signal_noise(self.hist_len)
        self.setup_plot_cps()
        plt.tight_layout()

    def update_plots(self, next_val):
        temp_norm_hist, popt, perr, current_time, synccnt, inputcnt = next_val
        self.background_data = np.concatenate((temp_norm_hist[:self.hist_len//4],
                                               temp_norm_hist[-self.hist_len//4:]))
        self.noise_mean = np.mean(self.background_data)
        self.noise_std = np.std(self.background_data)

        self.update_plot_histogram(temp_norm_hist, popt, perr, current_time)
        self.update_plot_signal_noise(temp_norm_hist, popt, perr, current_time)
        self.update_plot_cps(current_time, synccnt, inputcnt)
#            plt.draw()
#            plt.pause(0.001)
#            self.fig.canvas.draw()
        #self.fig.canvas.flush_events()

        plt.pause(0.01)
        self.fig.canvas.draw()

    def __call__(self, q):
        '''This will be called continuously after creating the process.'''
        self.setup_plots()
        self.queue = q
        while True:
            try:
                val = q.get(block=False)
                if val is None:
                    return
                self.update_plots(val)
            except queue.Empty:
                plt.pause(0.1)

    def setup_plot_histogram(self, hist_len, bin_width, fit_function_name):

        self.fit_function = getattr(common_util, fit_function_name + '_decay')
        self.fit_function_sigma = getattr(common_util, fit_function_name + '_decay_sigma')

        # setup the plots
        float_part = 10**(np.log10(bin_width)-int(np.log10(bin_width)))
        exp_part = int(np.log10(bin_width))
        hist_time = (np.linspace(-hist_len//2+1, hist_len//2, num=hist_len)-1)*float_part # time in s

        self.fit_time = (np.linspace(-hist_len//2+1, hist_len//2, num=hist_len*100)-1)*bin_width # time in s
        fit_plot_time = (np.linspace(-hist_len//2+1, hist_len//2, num=hist_len*100)-1)*float_part # time in s

        self.ax_histo.set_xlim(left=1.01*hist_time[0], right=1.01*hist_time[-1])
    #    ax.set_ylim(bottom=0.5, top=1.5)

        x_label_exp = '{' + str(exp_part) + '}'
        self.ax_histo.set_xlabel(r'Delay $\tau$ (10$^{}$ s)'.format(x_label_exp))
        self.ax_histo.set_ylabel('Correlation g$^{(2)}$')
        self.ax_histo.set_title('Correlation histogram')
        self.ax_histo.ticklabel_format(style='sci', axis='x', scilimits=(-3,3), useMathText=True, useOffset=False)
        self.plot_data, = self.ax_histo.plot(hist_time, np.ones_like(hist_time),
                                             color=DATA_COLOR, linestyle=DATA_LINESTYLE, marker=DATA_MARKER,
                                             label=r'Correlation g$^{(2)}$ histogram')
        self.plot_fit, = self.ax_histo.plot(fit_plot_time, self.fit_function(self.fit_time, 0, 5*bin_width, 0),
                                            ls=FIT_LINESTYLE, color=FIT_COLOR,
                                            label='Exponential decay fit \u00B1 ' + r'2$\sigma$')
        self.plot_bkg = self.ax_histo.axhspan(ymin=1-0, ymax=1+0,
                                              color=NOISE_COLOR, alpha=0.25,
                                              label='Background level \u00B1 noise')
        self.plot_signal_CB = self.ax_histo.fill_between([], y1=0, y2=0, color=FIT_ERROR_COLOR)

        handles, labels = self.ax_histo.get_legend_handles_labels()
        handles[1] = (self.plot_fit, self.plot_signal_CB)
        self.ax_histo.legend(handles, labels)

    def update_plot_histogram(self, temp_norm_hist, decay_params, decay_cov, current_time):
        decay_cov = np.array(decay_cov)

        self.plot_data.set_ydata(temp_norm_hist)
        fit_y = self.fit_function(self.fit_time, *decay_params)
        fit_std = self.fit_function_sigma(decay_cov, self.fit_time, *decay_params)
        self.plot_fit.set_ydata(fit_y)
        self.plot_bkg.set_xy([[0, self.noise_mean-self.noise_std], [0, self.noise_mean+self.noise_std],
                        [1, self.noise_mean+self.noise_std], [1, self.noise_mean-self.noise_std]])
        self.plot_signal_CB.remove()
        self.plot_signal_CB = self.ax_histo.fill_between(self.plot_fit.get_xdata(),
                                         y1=fit_y+2*fit_std,
                                         y2=fit_y-2*fit_std,
                                         color=FIT_ERROR_COLOR)

        self.ax_histo.relim()
        self.ax_histo.autoscale_view(scalex=False, scaley=True)
        self.ax_histo.set_ylim((np.min(temp_norm_hist)*0.995, np.max(temp_norm_hist)*1.01))

    def setup_plot_signal_noise(self, hist_len):
        '''Plots the signal and the noise as a function of the experimental time.'''
        self.ax_signal_noise.set_ylabel('Amplitude')
        self.ax_signal_noise.set_xlabel('Measurement time (s)')
        self.ax_signal_noise.set_title('Correlation signal and noise')
        self.ax_signal_noise.ticklabel_format(style='sci', axis='x', useMathText=True)

        self.signal_std_arr = np.array([])

        self.plot_signal_amp, = self.ax_signal_noise.plot([], [], color=FIT_COLOR, linestyle='',
                                                          marker='o', markersize=2, label='Signal amplitude \u00B1 '+r'2$\sigma$')
        self.plot_signal_err = self.ax_signal_noise.fill_between([], y1=[], y2=[], color=FIT_ERROR_COLOR)
        self.plot_signal_line = self.ax_signal_noise.axhline(y=0, xmin=-hist_len//2, xmax=hist_len//2,
                                                             alpha=0.5, color=FIT_COLOR, label='Last signal amplitude')
        self.plot_noise_amp, = self.ax_signal_noise.plot([], [], color=NOISE_COLOR, linestyle='', marker='o',
                                                         markersize=2, label=r'Noise amplitude and fit to t$^{-\frac{1}{2}}$')
        self.plot_noise_fit, = self.ax_signal_noise.plot([], [], color=NOISE_COLOR, alpha=0.5,
                                                         linestyle='-', label='Noise fit')

        # sort the legend so the 3 signal labels are together
        handles, labels = self.ax_signal_noise.get_legend_handles_labels()
        handles.insert(2, handles[-1])
        labels.insert(2, labels[-1])
        del handles[-1]
        del labels[-1]
        handles[0] = (self.plot_signal_amp, self.plot_signal_err)
        handles[2] = (self.plot_noise_amp, self.plot_noise_fit)
        del handles[-1]
        del labels[-1]
        self.ax_signal_noise.legend(handles, labels)

    def update_plot_signal_noise(self, temp_norm_hist, decay_params, decay_cov, current_time):
        decay_error = np.sqrt(np.diag(decay_cov))
        signal_amp = decay_params[0]
        signal_std = decay_error[0]
        self.signal_std_arr = np.append(self.signal_std_arr, signal_std)
        # time
        time_arr = np.append(self.plot_signal_amp.get_xdata(), current_time)
        # signal amplitude
        self.plot_signal_amp.set_ydata(np.append(self.plot_signal_amp.get_ydata(), signal_amp))
        self.plot_signal_amp.set_xdata(time_arr)
        # confidence bands
        self.plot_signal_err.remove()
        self.plot_signal_err = self.ax_signal_noise.fill_between(self.plot_signal_amp.get_xdata(),
                                          y1=self.plot_signal_amp.get_ydata()+2*self.signal_std_arr,
                                          y2=(self.plot_signal_amp.get_ydata()-2*self.signal_std_arr).clip(min=0),
                                          color=FIT_ERROR_COLOR)
        # last signal amplitude
        self.plot_signal_line.set_ydata(signal_amp)
        # noise amplitude
        self.plot_noise_amp.set_ydata(np.append(self.plot_noise_amp.get_ydata(), self.noise_std))
        self.plot_noise_amp.set_xdata(time_arr)

        # noise fit to t^(-1/2)
        if len(time_arr) > 3:
            noise_data = self.plot_noise_amp.get_ydata()
            noise_A, noise_A_err = get_noise_fit(noise_data, time_arr)
            # extrapolate to the time at which the noise amplitude is equal to the max signal (t_1=(A/max_signal)**2)
            # if there are no data points before
#            max_signal = np.max(noise_data)
#            init_time = min((noise_A/max_signal)**2, time_arr[0])
            interpolated_meas_time = np.linspace(time_arr[0], time_arr[-1], len(time_arr)*100)
            fit_noise_amplitude = np.interp(interpolated_meas_time, time_arr, noise_amp_fun(time_arr, noise_A))
            # update noise fit
#            self.plot_noise_fit.set_ydata(noise_amp_fun(time_arr, noise_A))
            self.plot_noise_fit.set_ydata(fit_noise_amplitude)
            self.plot_noise_fit.set_xdata(interpolated_meas_time)
            if self.noise_std < signal_amp:
                self.ax_signal_noise.set_title('Correlation signal and noise (T={:.0f} s)'.format((noise_A/signal_amp)**2))
            else:
                self.ax_signal_noise.set_title('Correlation signal and noise')

        self.ax_signal_noise.relim()
        self.ax_signal_noise.autoscale()
#        self.ax_signal_noise.autoscale_view(scalex=True, scaley=True)
#        self.ax_signal_noise.set_ylim((0, self.ax_signal_noise.get_ylim()[1]*1.01))

    def setup_plot_cps(self):
        '''Plots the synch and input detector counts as a function of the experimental time.'''
        self.ax_cps.set_ylabel('Detector counts (cps)')
        self.ax_cps.set_xlabel('Measurement time (s)')
        self.ax_cps.set_title('Detector count rates')
        self.ax_cps.ticklabel_format(style='sci', axis='x', useMathText=True)

        self.plot_synch, = self.ax_cps.plot([], [], 'bo-', markersize=2, label='SYNC')
        self.plot_input, = self.ax_cps.plot([], [], 'ro-', markersize=2, label='INPUT')

        self.ax_cps_ratio = self.ax_cps.twinx()
        self.plot_ratio, = self.ax_cps_ratio.plot([], [], 'ko-', markersize=2, label='SYNC/INPUT ratio')

        self.last_synch = 0
        self.last_input = 0
        self.last_time = 0

        self.ax_cps.legend(loc='upper left')
        self.ax_cps_ratio.legend(loc='upper right')

    def update_plot_cps(self, current_time, synccnt, inputcnt):
        common_time = np.append(self.plot_synch.get_xdata(), current_time)
        # counts
        new_synch = (synccnt - self.last_synch)/(current_time - self.last_time)
        new_input = (inputcnt - self.last_input)/(current_time - self.last_time)
        ratio = new_synch/new_input
        self.last_synch = synccnt
        self.last_input = inputcnt
        self.last_time = current_time

        # synch
        self.plot_synch.set_ydata(np.append(self.plot_synch.get_ydata(), new_synch))
        self.plot_synch.set_xdata(common_time)
        # input
        self.plot_input.set_ydata(np.append(self.plot_input.get_ydata(), new_input))
        self.plot_input.set_xdata(common_time)
        # ratio
        self.plot_ratio.set_ydata(np.append(self.plot_ratio.get_ydata(), ratio))
        self.plot_ratio.set_xdata(common_time)
        min_y = np.round(np.min(self.plot_ratio.get_ydata()), 1) - 0.1
        max_y = np.round(np.max(self.plot_ratio.get_ydata()), 1) + 0.1
        self.ax_cps_ratio.set_yticks(np.arange(min_y, max_y, 0.1))
#        np.round((min_y-max_y)/10, 1)
        self.ax_cps_ratio.set_ylim((min_y, max_y))

        self.ax_cps.relim()
        self.ax_cps_ratio.relim()
        self.ax_cps.autoscale_view(scalex=True, scaley=True)
        self.ax_cps_ratio.autoscale_view(scalex=True, scaley=True)



#def autoscale_based_on(ax, lines, scalex=True, scaley=True):
#    ax.dataLim = mtransforms.Bbox.unit()
#    ax.dataLim.y0 = 0.1
#    ax.dataLim.y1 = 0.1
#    for line in lines:
#        xy = np.vstack(line.get_data()).T
#        ax.dataLim.update_from_data_xy(xy, ignore=False)
##        ax.dataLim.y0 = np.min(xy[:, 1]) - 0.1
#    ax.autoscale_view(scalex=scalex, scaley=scaley)
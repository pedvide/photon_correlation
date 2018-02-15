# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 22:19:25 2017

@author: Villanueva
"""
import os
import sys
import traceback

import numpy as np

import common_util
from common_util import noise_amp_fun, get_noise_fit, get_SI_parts, get_SI_repr

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOptions(antialias=True)

import queue

# histogram data is black cirles
DATA_COLOR = 'k'
DATA_MARKER = 'o'
DATA_LINESTYLE = None
DATA_SIZE = 5
# fit data and signal amplitude are red
FIT_COLOR = 'r'
FIT_LINESTYLE = '-'
FIT_WIDTH = 3
FIT_ERROR_COLOR = (255, 0.0, 0.0, 127)  # red, alpha
# background and noise are blue
NOISE_COLOR = (0.0, 0.0, 255, 65)  # blue, alpha=0.25

class Plotter:

    def __init__(self, hist_len, bin_width, fit_function_name='symmetric', title=''):
        self.hist_len = hist_len
        self.bin_width = bin_width
        self.fit_function_name = fit_function_name
        self.title = title

#    def mouseMoved_histo(self, evt):
#        mousePoint = self.ax_histo.vb.mapSceneToView(evt[0])
#        self.cursor_label.setText('{}, {}'.format(mousePoint.x(), mousePoint.y()))


    def __call__(self, q):
        '''This will be called continuously after creating the process.'''
        try:
            self.setup_plots()
            self.queue = q
            while True:
                try:
                    val = q.get(block=False)
                    if val is None:
                        return
                    self.update_plots(val)
                except queue.Empty:
                    pass
                self.app.processEvents()
        except Exception as exc:
            sys.stdout = open(str(os.getpid()) + "_error_log.txt", "w")
            traceback.print_exc()
            raise


    def setup_plots(self):
        # setup the plots

        self.app = QtGui.QApplication([])

        self.fig = pg.GraphicsWindow(title='Correlation')
        self.ax_histo = self.fig.addPlot(row=0, col=0)
        self.ax_signal_noise = self.fig.addPlot(row=1, col=0)
        self.ax_cps = self.fig.addPlot(row=2, col=0)

        self.fig.setWindowTitle('Correlation experiment from ' + self.title + ', bin_width = ' + get_SI_repr(self.bin_width, separator='') + 's')

        self.setup_plot_histogram(self.hist_len, self.bin_width, self.fit_function_name)
        self.setup_plot_signal_noise(self.hist_len)
        self.setup_plot_cps()

#        self.cursor_label = pg.LabelItem(justify = "right")
#        self.fig.addItem(self.cursor_label)
#        self.proxy1 = pg.SignalProxy(self.ax_histo.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved_histo)


    def update_plots(self, next_val):
        temp_norm_hist, decay_params, decay_cov, current_time, synccnt, inputcnt = next_val
        self.background_data = np.concatenate((temp_norm_hist[:self.hist_len//4],
                                               temp_norm_hist[-self.hist_len//4:]))
        self.noise_mean = np.mean(self.background_data)
        self.noise_std = np.std(self.background_data)

        self.update_plot_histogram(temp_norm_hist, decay_params, decay_cov, current_time)
        self.update_plot_signal_noise(temp_norm_hist, decay_params, decay_cov, current_time)
        self.update_plot_cps(current_time, synccnt, inputcnt)


    def setup_plot_histogram(self, hist_len, bin_width, fit_function_name):

        self.fit_function = getattr(common_util, fit_function_name + '_decay')
        self.fit_function_sigma = getattr(common_util, fit_function_name + '_decay_sigma')

        self.hist_len = hist_len

        # setup the plots
        float_part, exp_part, unit = get_SI_parts(bin_width)
        hist_time = (np.linspace(-hist_len//2+1, hist_len//2, num=hist_len)-1)*float_part # time in s
        self.hist_time = hist_time

        self.fit_time = (np.linspace(-hist_len//2+1, hist_len//2, num=hist_len*100)-1)*bin_width # time in s
        fit_plot_time = (np.linspace(-hist_len//2+1, hist_len//2, num=hist_len*100)-1)*float_part # time in s

        self.ax_histo.setXRange(min=1.01*hist_time[0], max=1.01*hist_time[-1])
        self.ax_histo.addLegend()

        self.ax_histo.setLabel('bottom', text=fr'Delay &tau; ({unit}s)')
        self.ax_histo.setLabel('left', text='Correlation g<sup>(2)</sup>')
        self.ax_histo.setTitle('Correlation histogram, bin width = ' + get_SI_repr(self.bin_width, separator='') + 's')
        #self.ax_histo.ticklabel_format(style='sci', axis='x', scilimits=(-3,3), useMathText=True, useOffset=False)
        self.plot_data = self.ax_histo.plot(hist_time, np.ones_like(hist_time),
                                            pen=DATA_LINESTYLE, symbolPen=None,
                                            symbol=DATA_MARKER, symbolSize=DATA_SIZE,
                                            symbolBrush=pg.mkBrush(DATA_COLOR),
                                            name=r'Correlation g<sup>(2)</sup> histogram')
        self.plot_fit = self.ax_histo.plot(fit_plot_time, self.fit_function(self.fit_time, 0, 5*bin_width, 0),
                                            pen=pg.mkPen(FIT_COLOR, width=FIT_WIDTH),
                                            name='Exponential decay fit \u00B1 2&sigma;')

        self.plot_bkg = pg.QtGui.QGraphicsRectItem(hist_time[0], 1, self.hist_time[-1]- self.hist_time[0], 1)
        self.plot_bkg.setPen(pg.mkPen(None))
        self.plot_bkg.setBrush(pg.mkBrush(NOISE_COLOR))
        self.ax_histo.addItem(self.plot_bkg)
        style = pg.PlotDataItem(pen=pg.mkPen(NOISE_COLOR, width=3))
        self.ax_histo.legend.addItem(style, 'Background level \u00B1 noise')

        upper = pg.PlotDataItem(fit_plot_time, np.ones_like(fit_plot_time)+0.5)
        lower = pg.PlotDataItem(fit_plot_time, np.ones_like(fit_plot_time)-0.5)
        self.plot_signal_CB = pg.FillBetweenItem(upper, lower, brush=pg.mkBrush(FIT_ERROR_COLOR))
        self.ax_histo.addItem(self.plot_signal_CB)

    def update_plot_histogram(self, temp_norm_hist, decay_params, decay_cov, current_time):
        decay_cov = np.array(decay_cov)

        self.plot_data.setData(self.plot_data.getData()[0], temp_norm_hist)

        fit_y = self.fit_function(self.fit_time, *decay_params)
        self.plot_fit.setData(self.plot_fit.getData()[0], fit_y)

        self.plot_bkg.setRect(self.hist_time[0], self.noise_mean-self.noise_std,
                              self.hist_time[-1]- self.hist_time[0], 2*self.noise_std)

        fit_std = self.fit_function_sigma(decay_cov, self.fit_time, *decay_params)
        upper = pg.PlotDataItem(self.plot_fit.getData()[0], fit_y+2*fit_std)
        lower = pg.PlotDataItem(self.plot_fit.getData()[0], fit_y-2*fit_std)
        self.plot_signal_CB.setCurves(upper, lower)

        self.ax_histo.setYRange(min=np.min(temp_norm_hist)*0.995, max=np.max(temp_norm_hist)*1.01)

    def setup_plot_signal_noise(self, hist_len):
        '''Plots the signal and the noise as a function of the experimental time.'''
        self.ax_signal_noise.setLabel('left', text='Amplitude')
        self.ax_signal_noise.setLabel('bottom', text='Measurement time (s)')
        self.ax_signal_noise.setTitle('Correlation signal and noise')
        self.ax_signal_noise.setLimits(yMin=0)
        self.ax_signal_noise.addLegend()
#        self.ax_signal_noise.ticklabel_format(style='sci', axis='x', useMathText=True)

        self.signal_std_arr = np.array([])

        # signal amplitude
        self.plot_signal_amp = self.ax_signal_noise.plot([0], [0],
                                                         pen=pg.mkPen(FIT_COLOR), symbolPen=None,
                                                         symbol='o', symbolSize=5,
                                                         symbolBrush=pg.mkBrush(FIT_COLOR),
                                                         name='Signal amplitude \u00B1 2&sigma;')
        self.plot_signal_amp.setData([],[])

        # confidence bands
        upper = pg.PlotDataItem([0], [0])
        lower = pg.PlotDataItem([0], [0])
        self.plot_signal_err = pg.FillBetweenItem(upper, lower, brush=pg.mkBrush(FIT_ERROR_COLOR))
        self.ax_signal_noise.addItem(self.plot_signal_err)

        # last signal amplitude
        self.plot_signal_line = pg.InfiniteLine(pos=0, angle=0, movable=False,
                                                pen=pg.mkPen(FIT_COLOR, width=2), name='Last signal amplitude')
        self.ax_signal_noise.addItem(self.plot_signal_line)

        # noise amplitude
        self.plot_noise_amp = self.ax_signal_noise.plot([0], [0],
                                                        pen=pg.mkPen(NOISE_COLOR), symbolPen=None,
                                                        symbol='o', symbolSize=5,
                                                        symbolBrush=pg.mkBrush(NOISE_COLOR),
                                                        name='Noise amplitude and fit to t<sup>-\u00BD</sup>')
        self.plot_noise_amp.setData([],[])

        # noise fit to t^(-1/2)
        self.plot_noise_fit = self.ax_signal_noise.plot([0], [0], pen=pg.mkPen(NOISE_COLOR))
        self.plot_noise_fit.setData([],[])

    def update_plot_signal_noise(self, temp_norm_hist, decay_params, decay_cov, current_time):
        decay_error = np.sqrt(np.diag(decay_cov))
        signal_amp = decay_params[0]
        signal_std = decay_error[0]
        self.signal_std_arr = np.append(self.signal_std_arr, signal_std)
        # time
        time_arr = np.append(self.plot_signal_amp.getData()[0], current_time)
        # signal amplitude
        self.plot_signal_amp.setData(time_arr, np.append(self.plot_signal_amp.getData()[1], signal_amp))
        # confidence bands
        upper = pg.PlotDataItem(self.plot_signal_amp.getData()[0], self.plot_signal_amp.getData()[1]+2*self.signal_std_arr)
        lower = pg.PlotDataItem(self.plot_signal_amp.getData()[0], self.plot_signal_amp.getData()[1]-2*self.signal_std_arr)
        self.plot_signal_err.setCurves(upper, lower)
        # last signal amplitude
        self.plot_signal_line.setValue(signal_amp)
        # noise amplitude
        self.plot_noise_amp.setData(time_arr, np.append(self.plot_noise_amp.getData()[1], self.noise_std))

        # restrict y range to the maximum amplitude
        self.ax_signal_noise.setLimits(yMax=np.max(self.plot_signal_amp.getData()[1])*1.1)

        # noise fit to t^(-1/2)
        if len(time_arr) > 3:
            noise_data = self.plot_noise_amp.getData()[1]
            noise_A, noise_A_err = get_noise_fit(noise_data, time_arr)
            # extrapolate to the time at which the noise amplitude is equal to the max signal (t_1=(A/max_signal)**2)
            # if there are no data points before
            init_time = min((noise_A/signal_amp)**2, time_arr[0])
            interpolated_meas_time = np.linspace(init_time, time_arr[-1], len(time_arr)*100)
            fit_noise_amplitude = noise_amp_fun(interpolated_meas_time, noise_A)
            # update noise fit
            self.plot_noise_fit.setData(interpolated_meas_time, fit_noise_amplitude)
            if self.noise_std < signal_amp:
                self.ax_signal_noise.setTitle('Correlation signal and noise (T={:.0f} s)'.format((noise_A/signal_amp)**2))
            else:
                self.ax_signal_noise.setTitle('Correlation signal and noise')


    def setup_plot_cps(self):
        '''Plots the synch and input detector counts as a function of the experimental time.'''
        self.ax_cps.setLabel('left', text='Detector counts (cps)')
        self.ax_cps.setLabel('bottom', text='Measurement time (s)')
        self.ax_cps.setTitle('Detector count rates')
        self.ax_cps.addLegend()
#        self.ax_cps.ticklabel_format(style='sci', axis='x', useMathText=True)

        self.plot_synch = self.ax_cps.plot([0], [0],
                                           pen=pg.mkPen('b'), symbolPen=None,
                                           symbol='o', symbolSize=5,
                                           symbolBrush=pg.mkBrush('b'),
                                           name='SYNC')
        self.plot_synch.setData([],[])
        self.plot_input = self.ax_cps.plot([0], [0],
                                           pen=pg.mkPen('r'),
                                           symbol='o', symbolSize=5, symbolPen=None,
                                           symbolBrush=pg.mkBrush('r'),
                                           name='INPUT')
        self.plot_input.setData([],[])

        # ratio axis
        self.ax_cps_ratio = self.fig.addPlot(row=2, col=0)
        self.ax_cps_ratio.setLabel('bottom', text='Measurement time (s)')
        self.ax_cps_ratio.setTitle('')
        self.ax_cps_ratio.showAxis('right')
        self.ax_cps_ratio.hideAxis('left')
        self.ax_cps_ratio.hideAxis('bottom')
        self.ax_cps_ratio.setXLink(self.ax_cps)
        self.ax_cps_ratio.setLabel('right', 'SYNC/INPUT ratio')
        self.plot_ratio = pg.PlotCurveItem(pen=pg.mkPen('k'),
                                           symbol='o', symbolSize=5,
                                           symbolBrush=pg.mkBrush('k'),
                                           name='SYNC/INPUT ratio')
        self.ax_cps_ratio.addItem(self.plot_ratio)
#        self.ax_cps_ratio.getAxis('right').setHeight(self.ax_cps.getAxis('left').height())
#        self.ax_cps.getAxis('bottom').setWidth(self.ax_cps.getAxis('bottom').width())
#
        self.last_synch = 0
        self.last_input = 0
        self.last_time = 0

#        self.ax_cps.legend(loc='upper left')
#        self.ax_cps_ratio.legend(loc='upper right')

    def update_plot_cps(self, current_time, synccnt, inputcnt):
        common_time = np.append(self.plot_synch.getData()[0], current_time)
        # counts
        new_synch = (synccnt - self.last_synch)/(current_time - self.last_time)
        new_input = (inputcnt - self.last_input)/(current_time - self.last_time)
        ratio = new_synch/new_input
        self.last_synch = synccnt
        self.last_input = inputcnt
        self.last_time = current_time

        # synch
        self.plot_synch.setData(common_time, np.append(self.plot_synch.getData()[1], new_synch))
        # input
        self.plot_input.setData(common_time, np.append(self.plot_input.getData()[1], new_input))
        # ratio
        self.plot_ratio.setData(common_time, np.append(self.plot_ratio.getData()[1], ratio))
#        min_y = np.round(np.min(self.plot_ratio.get_ydata()), 1) - 0.1
#        max_y = np.round(np.max(self.plot_ratio.get_ydata()), 1) + 0.1
#        self.ax_cps_ratio.set_yticks(np.arange(min_y, max_y, 0.1))
#        np.round((min_y-max_y)/10, 1)
#        self.ax_cps_ratio.set_ylim((min_y, max_y))



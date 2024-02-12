# -*- coding: utf-8 -*-
"""
Created on Mar  18 18:49:50 2023

@author: Sergey Stasenko
"""

import os
import multiprocessing
import time
from scipy import signal
from brian2 import *
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from detecta import detect_peaks
import seaborn as sns
from matplotlib.ticker import FuncFormatter

class ProgressBar(object):
    def __init__(self, toolbar_width=40):
        self.toolbar_width = toolbar_width
        self.ticks = 0
    def __call__(self, elapsed, complete, start, duration):
        if complete == 0.0:
            # setup toolbar
            sys.stdout.write("[%s]" % (" " * self.toolbar_width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (self.toolbar_width + 1)) 
        else:
            ticks_needed = int(round(complete * self.toolbar_width))
            if self.ticks < ticks_needed:
                sys.stdout.write("-" * (ticks_needed-self.ticks))
                sys.stdout.flush()
                self.ticks = ticks_needed
        if complete == 1.0:
            sys.stdout.write("\n")

def run_sim(gammaY):
    pid = os.getpid()
    print(f'RUNNING {pid}')
    defaultclock.dt = 0.01*ms
    running_time = 10000*ms
    num_mean = 5
    NE = 8000          
    NI = NE/4          
    noise = 0.05*mV/ms
    seed(11922)
    tau_ampa = 5.0*ms  
    tau_gaba = 10.0*ms  
    epsilon = 0.02     
    tau_stdp = 20*ms    
    gl = 10.0*nsiemens   
    el = -60*mV         
    er = -80*mV          
    vt = -50.*mV         
    memc = 200.0*pfarad  
    bgcurrent = 200*pA   
    A_plus = 0.074; A_minus = -0.047
    mu_plus = 26.7*ms; mu_minus = -22.3*ms
    tau_plus = 9.3*ms; tau_minus = 10.8*ms
    Astro_state = 1
    Y_thr = 2.3 
    X_thr = 4
    astro_beta = 1.0 
    tauY = 120 *ms
    tauX= 20 *ms
    astro_gamma = gammaY 
    eqs_neurons='''
    dv/dt=(-gl*(v-el)-(g_ampa*v+g_gaba*(v-er))+bgcurrent)/memc : volt (unless refractory)
    dg_ampa/dt = -g_ampa/tau_ampa : siemens
    dg_gaba/dt = -g_gaba/tau_gaba : siemens
    '''
    neurons = NeuronGroup(NE+NI, model=eqs_neurons, threshold='v > vt',
                      reset='v=el', refractory=5*ms, method='euler')
    Pe = neurons[:NE]
    Pi = neurons[NE:]
    eqs_exc = '''
    dX/dt = -X / tauX : 1 (clock-driven)
    dY/dt = -Y / tauY : 1 (clock-driven)
    '''
    con_e = Synapses(Pe, neurons, model=eqs_exc, 
                 on_pre= '''
                                         X += 1
                        Y += astro_beta / (1 + exp(-X + X_thr))
                 g_ampa += 1.5*nS*(1 + Astro_state * astro_gamma/ (1 + exp(-Y + Y_thr)))''', method='euler')
    con_e.connect(p=epsilon)
    con_ii = Synapses(Pi, Pi, on_pre='g_gaba += 3*nS')
    con_ii.connect(p=epsilon)
    eqs_stdp_inhib = '''
    w : 1
    '''
    con_ie = Synapses(Pi, Pe, model=eqs_stdp_inhib,
                  on_pre='''lastspike_pre = t
    delta_t = (lastspike_post - lastspike_pre)
    w += A_minus * w * (1 + tanh((delta_t - mu_minus)/tau_minus))*eta
                           g_gaba += w*nS''',
                  on_post='''
                          lastspike_post = t

    delta_t = (lastspike_post - lastspike_pre)

    w += A_plus * w * (1 + tanh(-(delta_t - mu_plus)/tau_plus))*eta
                       ''', method='euler')
    con_ie.connect(p=epsilon)
    con_ie.w = 1e-10
    spike_monitor = SpikeMonitor(Pe)
    rate_monitor = PopulationRateMonitor(Pe)
    store()
    max_psd_freq = []
    spike_counts = []
    for i in range(num_mean):
        print('NUMBER_LOOP='+str(i))
        restore()
        eta = 0          
        run(1*second, report=ProgressBar(), report_period=1*second)
        eta = 1          
        run(running_time-1*second, report=ProgressBar(), report_period=1*second)
        x = rate_monitor.smooth_rate(window="gaussian",width=0.5*ms)/Hz
        ts = rate_monitor.t / ms
        t_min = 1000
        t_max = max(rate_monitor.t / ms)
        idx_rate = (ts >= t_min) & (ts <= t_max)
        ind = detect_peaks(x[idx_rate],show=False, mph=70, mpd=10)
	max_psd_freq.append((len(ind)/((t_max- t_min)/1000)))
        spike_counts.append(spike_monitor.count)
    mean_spike_counts = int(np.mean(spike_counts))
    max_psd_freq_mean = int(np.mean(max_psd_freq))
    res = (gammaY,max_psd_freq_mean,mean_spike_counts)
    print(f'FINISHED {pid}')
    return gammaY,max_psd_freq_mean,mean_spike_counts
 
if __name__ == "__main__":
    script_dir = os.path.abspath('')
    results_dir = os.path.join(script_dir, 'Results/Parameters/')
     if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    start_time = time.time()
    control_parameter = np.arange(0.0,17.0,0.3)
    if multiprocessing.cpu_count() >= len(control_parameter):
        num_proc = len(control_parameter)
    else:
        num_proc = multiprocessing.cpu_count()
    with multiprocessing.Pool(num_proc) as p:
        results = p.map(run_sim, control_parameter)
    elapsed_time = time.time() - start_time
    print(f'Running time is {elapsed_time/60} minutes')
    numpy_array = np.array(results)
    transpose = numpy_array.T
    transpose_list = transpose.tolist()
    gammaY, max_frequencies, mean_spike_counts = transpose_list
    np.savetxt("Results/Parameters/dataset.csv", np.transpose([gammaY, max_frequencies, mean_spike_counts]), fmt='%1.3f', delimiter=';')
    params = {'legend.fontsize': 20,
          'axes.labelsize': 14,
         'axes.titlesize':14,
         'xtick.labelsize':12,
         'ytick.labelsize':12,
         'font.family':"DejaVu Sans"}
    plt.rcParams.update(params)
    formatter = FuncFormatter(lambda x, _: f'{x:.0f}' if abs(x < 9999.5) else f'{x:,.0f}')
    data1 = pd.DataFrame(np.column_stack([gammaY, max_frequencies]), 
                               columns=['gammaY', 'max_frequencies'])
     g = sns.lmplot(data=data1,
           x='gammaY',
           y='max_frequencies',
           aspect=2,
           order=2,
           line_kws={'color': 'red', "lw":1},height=4)
    for ax in g.axes.flat:
        ax.yaxis.set_major_formatter(formatter)
    plt.ylabel(u"Mean bursts frequency [Hz]")
    plt.xlabel(u"$\gamma_Y$")
    plt.grid()
    plt.savefig(results_dir +'bursts_frequency_gamma.png', dpi=300)
    plt.savefig(results_dir +'bursts_frequency_gamma.pdf', dpi=300)
    data2 = pd.DataFrame(np.column_stack([gammaY, mean_spike_counts]), 
                               columns=['gammaY', 'mean_spike_counts'])
    g = sns.lmplot(data=data2,
           x='gammaY',
           y='mean_spike_counts',
           aspect=2,
           order=2,
           line_kws={'color': 'blue', "lw":1},height=4)
    for ax in g.axes.flat:
        ax.yaxis.set_major_formatter(formatter)
    plt.ylabel(u"Mean spikes count")
    plt.xlabel(u"$\gamma_Y$")
    plt.grid()
    plt.savefig(results_dir +'mean_spike_counts_gamma.png', dpi=300)
    plt.savefig(results_dir +'mean_spike_counts_gamma.pdf', dpi=300)


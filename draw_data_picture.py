#!/usr/bin/env python
# coding:utf-8
"""
Author  : Xiwen Lu
Time    : 11/1/22 2:59 PM
Desc    : draw the brownianMotion data
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.ndimage import shift
import time

def probability(configs, m:int,is_pair:bool):
    R = configs["receiver_radius"]
    d = configs["distance"]
    D = configs["diffusion"]
    ts = configs["delta_t"]
    h = configs["h"]
    Vm = configs["velocity"][0]
    particle_num = configs["particle_num"]
    t = ts * m
    vr = 4/3*np.pi*R**3  #接收器体积
    if is_pair:
        p = (vr/pow((4*np.pi*D*t),1.5))*np.exp(-(d-t*Vm)**2/(4*D*t))
    else:
        p = (vr/pow((4*np.pi*D*t),1.5))*np.exp(-((d-t*Vm)**2+h**2)/(4*D*t))
    return particle_num*p

def getAffectStep(configs):
    for step in range(1, 10):
        if probability(configs, step * configs["sample_index"]) - 0 < 1e-6 * configs["particle_num"]:
            break
    return step

def BrownianMotionAndDrift(configs,release_signals,file_folder="./Data/data_new_5000/"):
    particle_num = configs["particle_num"]
    total_time = configs["total_time"]
    delta_t = configs["delta_t"]
    sample_time = configs["sample_time"]
    release_time = configs["release_time"]
    receiver_radius = configs["receiver_radius"]
    # distance = configs["distance"]
    receiver_location = configs["receiver_location"]
    velocity = configs["velocity"]
    diffusion = configs["diffusion"]
    sample_index = int(np.round(sample_time / delta_t))
    release_times = len(release_signals)
    release_indexs = int(np.round(release_time / delta_t))

    # affect_step = getAffectStep(configs)
    affect_step = 10
    # create a total array than contains all the information
    num_hit = np.zeros(int(total_time / delta_t))
    num_hit_sample = np.zeros(release_times)
    # do iteration for the release times, caculate the brownian motion of the particles
    t = trange(release_times)
    for i in t:
        t.set_postfix({'Sample_index': j})
        if release_signals[i] == 0:
            continue
        total_data = np.zeros((affect_step * release_indexs, particle_num, 4))

        # for j in range(particle_num):
        # curr_release_time = int(np.round(i * release_time/delta_t))
        curr_release_time = 0
        remain_steps = total_data.shape[0] - curr_release_time - 1
        total_data[curr_release_time + 1:, :, 0] = np.cumsum(np.random.normal(
            velocity[0] * delta_t, math.sqrt(2 * diffusion * delta_t),
            size=(remain_steps, particle_num)), axis=0)
        # consider that the velocity y = z, so here otal_data.shape[0]-curr_release_time-1combain the two part
        total_data[curr_release_time + 1:, :, 1:3] = np.cumsum(np.random.normal(
            velocity[1] * delta_t, math.sqrt(2 * diffusion * delta_t),
            size=(remain_steps, particle_num, 2)), axis=0)

        tmp = np.sqrt(np.sum((total_data[:, :, 0:3] - receiver_location) ** 2, axis=2))
        total_data[tmp < receiver_radius, 3] = 1

        try:
            num_hit[i * release_indexs:(i + affect_step) * release_indexs] += np.sum(total_data[:, :, 3], axis=1)
        except:
            num_hit[i * release_indexs:] += np.sum(total_data[:, :, 3], axis=1)[:len(num_hit) - i * release_indexs]
    # np.save('./Data/data_new_5000/d_{}_r_{}_num_{}_{}_nhit'.format(distance,receiver_radius,particle_num,j),num_hit)
    # here, the start symbol from sample_index-1, to ensure the sample dots equal the signals
    num_hit_sample = num_hit[sample_index - 1::sample_index]
    # np.save('{}d_{}_r_{}_num_{}_{}'.format(file_folder, distance, receiver_radius, particle_num, j),
    #         [release_signals[:num_hit_sample.shape[0]], num_hit_sample])

    return num_hit,num_hit_sample

def calc_theoretical_nhit_num(configs,release_signals:np.ndarray,is_pair:bool):
    sample_indexs = int(np.round(configs["release_time"] / configs["delta_t"]))
    p = [probability(configs, i,is_pair=is_pair) for i in range(1, sample_indexs * len(release_signals))]
    p = [0, *p]
    tmp = np.array([shift(p, i * sample_indexs, cval=0) for i in range(len(release_signals))])
    p_true = np.sum(tmp*release_signals[:,np.newaxis],axis=0)

    return p_true

def draw_simulate_results(configs,single_simulation,avg_simulation,release_signals,num_hit_sample):
    # sample_index = int(np.round(configs["sample_time"] / configs["delta_t"]))
    theoretical_nums_11 = calc_theoretical_nhit_num(configs, release_signals[:len(release_signals)//2], is_pair=True)
    theoretical_nums_21 = calc_theoretical_nhit_num(configs, release_signals[len(release_signals)//2:], is_pair=False)
    theoretical_nums = theoretical_nums_11+theoretical_nums_21
    plt.figure(figsize=(10,8))
    plt.plot(single_simulation[:len(theoretical_nums)],color='orange')
    plt.plot(avg_simulation[:len(theoretical_nums)],color='green')
    plt.plot(theoretical_nums,color='blue')
    if num_hit_sample!=None:
        plt.scatter(configs["sample_index"]*np.arange(1,num_hit_sample.shape[0]+1)*release_signals,num_hit_sample*release_signals,color='red',s=50,zorder=3)
        plt.scatter(configs["sample_index"]*np.arange(1,num_hit_sample.shape[0]+1)*(1-release_signals),num_hit_sample*(1-release_signals),color='blue',s=50,zorder=3)
    plt.title('d={},r={},num={}\nsignals={}'.format(configs["distance"],configs["receiver_radius"],configs["particle_num"],release_signals))
    plt.savefig('new.svg',dpi=600)
    plt.show()

if __name__ == '__main__':
    # here, wo should ensure that all time params can be divided by delta_time
    """
    unit:
    total_time: s
    delta_time: s
    diffusion: μm^2/s
    sphere_radius: μm
    velocity:μm/s
    """
    configs = {
        'particle_num': 10000,
        'receiver_radius': 20,
        'distance': 200.,
        'receiver_location': [200, 0, 0],
        'diffusion': 800,
        'delta_t': 0.01,
        'total_time': 80,
        'release_time': 8,
        'sample_time': 8,
        'velocity': [30, 0, 0],
        'h':200.
    }
    receiver_location_1 = configs['receiver_location']
    receiver_location_2 = [configs['distance'],configs['h'],0]
    # computational attributes
    configs["sample_index"] = int(np.round(configs["sample_time"] / configs["delta_t"]))
    configs["release_times"] = release_times = int(np.round(configs["total_time"] / configs["release_time"]))

    # release_signals = np.ones(release_times)
    num_hit_avg = np.zeros(int(configs["total_time"] / configs["delta_t"]))
    start_runing_time = time.time()
    random_samples = 5
    # to generate different data, this code should be put in the loop
    # release_signals = np.random.binomial(1, 0.5, release_times)
    release_signals = np.array([1,0,0,1,1,0,1,0,1,0])
    # release_signals = np.ones(release_times)
    for j in range(random_samples):
        configs['receiver_location'] = receiver_location_1
        num_hit_11, _= BrownianMotionAndDrift(configs, release_signals[:len(release_signals)//2],file_folder="./Data/data_new_5000/")
        configs['receiver_location'] = receiver_location_2
        num_hit_21, _= BrownianMotionAndDrift(configs, release_signals[len(release_signals)//2:], file_folder="./Data/data_new_5000/")
        num_hit = num_hit_11 + num_hit_21
        num_hit_avg += num_hit
    num_hit_avg = num_hit_avg / random_samples
    end_running_time = time.time()
    draw_simulate_results(configs,num_hit,num_hit_avg,release_signals,num_hit_sample=None)
    print("Simulation Finished.\nTotal running time is {}.".format(end_running_time - start_runing_time))

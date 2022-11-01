# ################################################
# This file includes functions to generate data and calculate BLER
# ################################################


import math
import numpy as np
from scipy.stats import norm
import scipy.special as sc
import matplotlib.pyplot as plt

# Functions for data generation
Rr=2e-5
d=2e-4
h = 2e-4
D=8e-10
#ts = d** 2 / (4 * D * sc.erfinv(0.4)** 2)
ts = 8
delta_t=0.01
sigma_noise = 100 #noise
vm = 3e-5
#vm = 1e-5 0
class Point():
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, other):
        return math.sqrt((self.x-other.x)**2+
                         (self.y-other.y)**2+(self.z-other.z)**2)

class Tx():
    #n = 100  #molecular per bit
    def __init__(self,position:Point):
        self.pos = position
        self.send_signal = []
    def send(self,signal,number):
        self.send_signal = signal
        self.send_num = signal * number

class Rx():
    def __init__(self,position:Point):
        self.pos = position

    def receive(self,N_hit):
        self.receive_data = N_hit

def gen_noise(length):
    noise = np.random.normal(0,np.sqrt(sigma_noise),length)
    return noise

def probability(vm,R,d,D,t):
    vr = 4/3*np.pi*R**3  #接收器体积
    p = (vr/pow((4*np.pi*D*t),1.5))*np.exp(-(d-t*vm)**2/(4*D*t))
    return p

def un_probability(vm,R,d,h,D,t):
    vr = 4/3*np.pi*R**3  #接收器体积
    p = (vr/pow((4*np.pi*D*t),1.5))*np.exp(-((d-t*vm)**2+h**2)/(4*D*t))
    return p


def gen_traindata(number,X,tx1,tx2,d,h):
    data_len = len(X)

    n_hit1 = np.zeros(data_len // 2)
    n_hit2 = np.zeros(data_len // 2)

    transby1 = tx1.send_signal  # tx1传输的序列
    transby2 = tx2.send_signal  # tx2传输的序列

    A0 = probability(vm, Rr, d, D, ts)
    A1 = probability(vm, Rr, d, D, ts * 2)
    B0 = un_probability(vm, Rr, d, h, D, ts)
    B1 = un_probability(vm, Rr, d, h, D, ts * 2)

    n_1Ts1 = np.random.normal(number * A0, np.sqrt(number * A0 * (1 - A0)))  # rx1第一个时隙内到达的粒子总数
    n_1Ts2 = np.random.normal(number * A0, np.sqrt(number * A0 * (1 - A0)))  # rx2第一个时隙内到达的粒子总数

    n_hit1[0] = n_1Ts1 * transby1[0]  # rx1第一个时隙到达的粒子数
    n_hit2[0] = n_1Ts2 * transby2[0]  # rx2第一个时隙到达的粒子数
    first_ili1 = np.random.normal(number * B0, np.sqrt(number * B0 * (1 - B0)))  # 计算rx1接收到来自tx2的ILI 只考虑前一个时隙
    first_ili2 = np.random.normal(number * B0, np.sqrt(number * B0 * (1 - B0)))  # 计算rx2接收到来自tx1的ILI 只考虑前一个时隙
    if transby1[0] == 1:
        n_hit2[0] += first_ili1
    if transby2[0] == 1:
        n_hit1[0] += first_ili2

    for i in range(1, data_len // 2):
        n_tc1 = np.random.normal(number * A0, np.sqrt(number * A0 * (1 - A0)))  # 本时隙粒子到达的数量
        n_tc2 = np.random.normal(number * A0, np.sqrt(number * A0 * (1 - A0)))  # 本时隙粒子到达的数量
        n_c1 = n_tc1 * transby1[i]  # 本时隙rx1到达的粒子数
        n_c2 = n_tc2 * transby2[i]  # 本时隙rx2到达的粒子数

        tmpisi1 = np.random.normal(number * A1, np.sqrt(number * A1 * (1 - A1)))  # 计算rx1到达的ISI 只考虑前一个时隙
        tmpisi2 = np.random.normal(number * A1, np.sqrt(number * A1 * (1 - A1)))  # 计算rx2到达的ISI 只考虑前一个时隙

        tmpili1 = np.random.normal(number * B1, np.sqrt(number * B1 * (1 - B1)))  # 计算rx1接收到来自tx2的ILI 只考虑前一个时隙
        tmpili2 = np.random.normal(number * B1, np.sqrt(number * B1 * (1 - B1)))  # 计算rx2接收到来自tx1的ILI 只考虑前一个时隙

        if transby1[i - 1] == 1:
            nisi1 = tmpisi1
            nili2 = tmpili2  # 前一时隙tx1对rx2的影响
            if transby1[i] == 1:
                nili2 += np.random.normal(number * B0, np.sqrt(number * B0 * (1 - B0)))  # 本时隙tx1对rx2的影响
        else:
            nisi1 = 0
            nili2 = 0
            if transby1[i] == 1:
                nili2 += np.random.normal(number * B0, np.sqrt(number * B0 * (1 - B0)))  # 本时隙tx1对rx2的影响

        if transby2[i - 1] == 1:
            nisi2 = tmpisi2
            nili1 = tmpili1  # 前一时隙tx2对rx1的影响
            if transby2[i] == 1:
                nili1 += np.random.normal(number * B0, np.sqrt(number * B0 * (1 - B0)))  # 本时隙tx2对rx1的影响
        else:
            nisi2 = 0
            nili1 = 0
            if transby2[i] == 1:
                nili1 += np.random.normal(number * B0, np.sqrt(number * B0 * (1 - B0)))  # 本时隙tx2对rx1的影响

        n_hit1[i] = n_c1 + nisi1 + nili1
        n_hit2[i] = n_c2 + nisi2 + nili2

    N_hit = np.append(n_hit1, n_hit2)
    # N_hit = N_hit.astype(int)

    return (X, N_hit)  # ,thresh)

def gen_testdata(number,X,tx1,tx2,d,h):
    data_len = len(X)
    noise1 = gen_noise(data_len)
    noise2 = gen_noise(data_len)
    n_hit1 = np.zeros(data_len//2)
    n_hit2 = np.zeros(data_len // 2)

    transby1 = tx1.send_signal  #tx1传输的序列
    transby2 = tx2.send_signal  #tx2传输的序列

    A0 = probability(vm,Rr,d,D,ts)
    A1 = probability(vm, Rr, d, D, ts*2)
    B0 = un_probability(vm,Rr,d,h,D,ts)
    B1 = un_probability(vm, Rr, d, h, D, ts*2)

    n_1Ts1 = np.random.normal(number*A0,np.sqrt(number*A0*(1-A0)))  #rx1第一个时隙内到达的粒子总数
    n_1Ts2 = np.random.normal(number*A0,np.sqrt(number*A0*(1-A0)))  #rx2第一个时隙内到达的粒子总数

    n_hit1[0] = n_1Ts1 * transby1[0] + noise1[0]  # rx1第一个时隙到达的粒子数
    n_hit2[0] = n_1Ts2 * transby2[0] + noise2[0]  # rx2第一个时隙到达的粒子数
    first_ili1 = np.random.normal(number * B0, np.sqrt(number * B0 * (1 - B0)))  # 计算rx1接收到来自tx2的ILI 只考虑前一个时隙
    first_ili2 = np.random.normal(number * B0, np.sqrt(number * B0 * (1 - B0)))  # 计算rx2接收到来自tx1的ILI 只考虑前一个时隙
    if transby1[0] == 1:
        n_hit2[0] += first_ili1
    if transby2[0] == 1:
        n_hit1[0] += first_ili2
    for i in range(1,data_len//2):
        n_tc1 = np.random.normal(number*A0,np.sqrt(number*A0*(1-A0)))  #本时隙粒子到达的数量
        n_tc2 = np.random.normal(number*A0,np.sqrt(number*A0*(1-A0)))  # 本时隙粒子到达的数量
        n_c1 = n_tc1 * transby1[i]  #本时隙rx1到达的粒子数
        n_c2 = n_tc2 * transby2[i]  #本时隙rx2到达的粒子数

        tmpisi1 = np.random.normal(number*A1,np.sqrt(number*A1*(1-A1)))  #计算rx1到达的ISI 只考虑前一个时隙
        tmpisi2 = np.random.normal(number*A1,np.sqrt(number*A1*(1-A1)))# 计算rx2到达的ISI 只考虑前一个时隙

        tmpili1 = np.random.normal(number * B1,np.sqrt(number*B1*(1-B1)))  # 计算rx1接收到来自tx2的ILI 只考虑前一个时隙
        tmpili2 = np.random.normal(number * B1,np.sqrt(number*B1*(1-B1))) # 计算rx2接收到来自tx1的ILI 只考虑前一个时隙

        if transby1[i-1]==1:
            nisi1 = tmpisi1
            nili2 = tmpili2   #前一时隙tx1对rx2的影响
            if transby1[i]==1:
                nili2 += np.random.normal(number * B0,np.sqrt(number*B0*(1-B0)))  #本时隙tx1对rx2的影响
        else:
            nisi1 = 0
            nili2 = 0
            if transby1[i] == 1:
                nili2 += np.random.normal(number * B0,np.sqrt(number*B0*(1-B0)))  #本时隙tx1对rx2的影响

        if transby2[i-1]==1:
            nisi2 = tmpisi2
            nili1 = tmpili1  #前一时隙tx2对rx1的影响
            if transby2[i]==1:
                nili1 += np.random.normal(number * B0,np.sqrt(number*B0*(1-B0))) #本时隙tx2对rx1的影响
        else:
            nisi2 = 0
            nili1 = 0
            if transby2[i] == 1:
                nili1 += np.random.normal(number * B0,np.sqrt(number*B0*(1-B0))) #本时隙tx2对rx1的影响


        n_hit1[i] = n_c1 + nisi1 + nili1 + noise1[i]
        n_hit2[i] = n_c2 + nisi2 + nili2 + noise2[i]


    N_hit = np.append(n_hit1,n_hit2)
    # N_hit = N_hit.astype(int)

    return (X,N_hit)#,thresh)


def generate_transmit_data(M, num, seed=0):
    #print('Generate transmit data: M = %d, seed = %d' %(M, seed))
    np.random.seed(seed)
    #symbol_index = np.random.randint(M,size=num)
    #symbol_index = np.random.binomial(1, 0.5, size=num)
    symbol_index1 = np.random.binomial(1, 0.5, size=num // 2)
    symbol_index2 = np.random.binomial(1, 0.5, size=num // 2)
    symbol_index = np.concatenate((symbol_index1 , symbol_index2),axis=0)
    X = np.zeros((M,num), dtype = 'float32')
    Y = np.zeros((M,num), dtype = 'float32')
    for i in range(num):
        X[symbol_index[i],i] = 1
    Y = X    
    return symbol_index, X, Y

def BER(n, M, sym_index_test, Y_pred, num_test):
    err = 0
    for i in range(num_test):
        y_pred_index = np.argmax(Y_pred[:,i])
        if (y_pred_index != sym_index_test[i]):
          err = err + 1       
    ber = err/num_test
    return ber

def BER_my(data):
    X = data[0]
    N_hit = data[1]
    # thres = 50
    thres = data[2]
    num = len(X)
    right = 0
    for i in range(num):
        current_x = X[i]
        current_y = N_hit[i]
        if current_x == 0 and current_y < thres:
            right += 1
        elif current_x == 1 and current_y >= thres:
            right += 1
        else:
            continue
    BER = (num - right) / num
    return BER

def cul_H(number):
    E = np.identity(2)
    p = probability(vm, Rr, d, D,ts)
    H = (1/(number*p))*E
    return H

def cul_thresh(number):
    number = number.astype(np.int64)
    A0 = probability(vm, Rr, d, D, ts)
    A1 = probability(vm, Rr, d, D, 2 * ts)
    B0 = un_probability(vm, Rr, d, h, D, ts)
    B1 = un_probability(vm, Rr, d, h, D, 2 * ts)

    sum_p = A1 + B0 + B1
    miu_I = 0.5 * number * sum_p
    miu_0 = miu_I / (number * A0)
    sigma_I = 0.5 * 0.5 * number * number * (A1 * A1 + B0 * B0 + B1 * B1) + 0.5 * number * (
            A1 * (1 - A1) + B0 * (1 - B0) + B1 * (1 - B1))
    sigma_0 = (sigma_I + sigma_noise) / (number * number * A0 * A0)
    sigma_1 = ((1 - A0) / (number * A0)) + sigma_0
    beta = sigma_1 / sigma_0
    aita_p1 = miu_0 + ((-1 - np.sqrt(1 + (beta - 1) * (1 + sigma_0 * beta * np.log(beta)))) / (beta - 1))
    aita_p2 = miu_0 + ((-1 + np.sqrt(1 + (beta - 1) * (1 + sigma_0 * beta * np.log(beta)))) / (beta - 1))
    return aita_p1, aita_p2

    # number = number.astype(np.int64)
    # A0 = probability(vm, Rr, d, D,ts)
    # A1 = probability(vm, Rr, d, D,2*ts)
    # B0 = un_probability(vm, Rr, d,h,D,ts)
    # B1 = un_probability(vm, Rr, d,h,D,2*ts)
    #
    # # sum_p = A1 + B0 + B1
    # # miu_I = 0.5*number*sum_p
    # # miu_0 = miu_I/(number*A0)
    # # sigma_I = 0.5*0.5*number*number*(A1*A1+B0*B0+B1*B1) + 0.5*number*( A1*(1-A1) + B0*(1-B0) + B1*(1-B1))
    # # sigma_0 = (sigma_I+sigma_noise)/(number*number*A0*A0)
    # # sigma_1 = ((1-A0)/(number*A0)) + sigma_0
    # sum_p = A1 + B0 + B1
    # miu_I = 0.5 * number * sum_p
    # miu_0 = miu_I
    # miu_1 = number * A0 + miu_0
    # sigma_I = 0.5 * 0.5 * number * number * (A1 * A1 + B0 * B0 + B1 * B1) + 0.5 * number * (
    #         A1 * (1 - A1) + B0 * (1 - B0) + B1 * (1 - B1))
    # sigma_0 = (sigma_I + sigma_noise)
    # sigma_1 = ((1 - A0) * (number * A0)) + sigma_0
    # beta = sigma_1/sigma_0
    # aita_p1 = miu_0 + ((-1-np.sqrt(1+(beta-1)*(1+sigma_0*beta*np.log(beta))))/(beta-1))
    # aita_p2 = miu_0 + ((-1+np.sqrt(1+(beta-1)*(1+sigma_0*beta*np.log(beta))))/(beta-1))
    # return aita_p1,aita_p2

def Normal(x,miu,sigma):
    return (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-((x-miu)*(x-miu)/(2*sigma*sigma)))

def fir_cul_pro(p,x11,x12,x21,x22,y11,y12,y21,y22,number):
    number = number.astype(np.int64)
    A0 = probability(vm, Rr, d, D, ts)
    B0 = 0
    A1 = 0
    B1 = 0
    x = 0
    if p == 0:
        x = y11
        if x21 == 1:  #此处应为x21
            B0 = un_probability(vm, Rr, d, h, D, ts)
    if p == 1:
        x = y12
        if x11 == 1:
            A1 = probability(vm, Rr, d, D, 2 * ts)
        if x21 == 1:
            B1 = un_probability(vm, Rr, d, h, D, 2 * ts)
        if x22 == 1:
            B0 = un_probability(vm, Rr, d, h, D, ts)
    if p == 2:
        x = y21
        if x11 == 1:
            B0 = un_probability(vm, Rr, d, h, D, ts)
    if p == 3:
        x = y22
        if x11 == 1:
            B1 = un_probability(vm, Rr, d, h, D, 2 * ts)
        if x12 == 1:
            B0 = un_probability(vm, Rr, d, h, D, ts)
        if x21 == 1:
            A1 = probability(vm, Rr, d, D, 2 * ts)

    sum_p = A1 + B0 + B1
    miu_I = 0.5 * number * sum_p
    miu_0 = miu_I
    miu_1 = number * A0 + miu_0
    sigma_I = 0.5 * 0.5 * number * number * (A1 * A1 + B0 * B0 + B1 * B1) + 0.5 * number * (
            A1 * (1 - A1) + B0 * (1 - B0) + B1 * (1 - B1))
    sigma_0 = (sigma_I + sigma_noise)
    sigma_1 = ((1 - A0) * (number * A0)) + sigma_0

    # Note: sigma actually is the square value
    pro_0 = Normal(x,miu_0,np.sqrt(sigma_0)) #考虑为0,0情况下 值为nan的情况
    if math.isnan(pro_0):
        if x == 0:
            pro_0 = 1
        else:
            pro_0 = 0
    pro_1 = Normal(x, miu_1, np.sqrt(sigma_1))
    return pro_0,pro_1

def cul_pro(j,x10,x11,x12,x20,x21,x22,y11,y12,y21,y22,number):
    number = number.astype(np.int64)
    A0 = probability(vm, Rr, d, D, ts)
    B0 = 0
    A1 = 0
    B1 = 0
    x = 0
    if j == 0:
        x = y11
        if x10 == 1:
            A1 = probability(vm, Rr, d, D, 2 * ts)
        if x20 == 1:
            B1 = un_probability(vm, Rr, d, h, D, 2 * ts)
        if x21 == 1:
            B0 = un_probability(vm, Rr, d, h, D, ts)
    if j == 1:
        x = y12
        if x11 == 1:
            A1 = probability(vm, Rr, d, D, 2 * ts)
        if x21 == 1:
            B1 = un_probability(vm, Rr, d, h, D, 2 * ts)
        if x22 == 1:
            B0 = un_probability(vm, Rr, d, h, D, ts)
    if j == 2:
        x = y21
        if x20 == 1:
            A1 = probability(vm, Rr, d, D, 2 * ts)
        if x10 == 1:
            B1 = un_probability(vm, Rr, d, h, D, 2 * ts)
        if x11 == 1:
            B0 = un_probability(vm, Rr, d, h, D, ts)
    if j == 3:
        x = y22
        if x11 == 1:
            B1 = un_probability(vm, Rr, d, h, D, 2 * ts)
        if x12 == 1:
            B0 = un_probability(vm, Rr, d, h, D, ts)
        if x21 == 1:
            A1 = probability(vm, Rr, d, D, 2 * ts)

    sum_p = A1 + B0 + B1
    miu_I = 0.5*number*sum_p
    miu_0 = miu_I
    miu_1 = number*A0 + miu_0
    sigma_I = 0.5 * 0.5 * number * number * (A1 * A1 + B0 * B0 + B1 * B1) + 0.5 * number * (
                A1 * (1 - A1) + B0 * (1 - B0) + B1 * (1 - B1))
    sigma_0 = (sigma_I + sigma_noise)
    sigma_1 = ((1 - A0) * (number * A0)) + sigma_0
    # Note: sigma actually is the square value
    pro_0 = Normal(x, miu_0, np.sqrt(sigma_0))
    if math.isnan(pro_0):
        if x == 0:
            pro_0 = 1
        else:
            pro_0 = 0
    pro_1 = Normal(x, miu_1, np.sqrt(sigma_1))
    return pro_0, pro_1



def draw():
    number = 30000
    length = 10
    x= np.random.binomial(1, 0.5, size=length)
    x1 = x[0:length//2]
    x2 = x[length//2:length]
    t = np.linspace(delta_t, 0.5*length * ts, num=int(0.5*length * ts / delta_t))
    t1 = np.linspace(delta_t, ts, num=int(ts / delta_t))
    t2 = np.linspace(delta_t+ts, 2*ts, num=int(ts / delta_t))

    p_cur = probability(vm, Rr, d, D, t1)  #一个时隙 当前时隙到达的
    p_isi = probability(vm, Rr, d, D, t2)  #一个时隙 上一个时隙到达的
    p_lastili = un_probability(vm, Rr, d, h, D, t2 )   #一个时隙 上一个时隙到达的
    p_curili = un_probability(vm, Rr, d, h, D, t1 )   #一个时隙 当前时隙到达的

    res1 = np.zeros(len(t))
    res2 = np.zeros(len(t))
    if x1[0] == 1:
        res1[0:len(t1)] += p_cur*number
        res2[0:len(t1)] += p_curili * number
    if x2[0] == 1:
        res2[0:len(t1)] += p_cur*number
        res1[0:len(t1)] += p_curili * number

    for i in range(1,len(x1)):
        cur_trans1 = x1[i]
        cur_trans2 = x2[i]
        if cur_trans1 == 1:
            res1[i*len(t1):(i+1) * len(t1)] += p_cur*number
            res2[i * len(t1):(i+1) * len(t1) ] += p_curili * number
        if cur_trans2 == 1:
            res1[i * len(t1):(i+1) * len(t1)] += p_curili * number
            res2[i*len(t1):(i+1) * len(t1)] += p_cur*number

        last_trans1 = x1[i - 1]
        last_trans2 = x2[i - 1]
        if last_trans1 == 1:
            res1[i*len(t1):(i+1) * len(t1)] += p_isi*number
            res2[i*len(t1):(i+1) * len(t1)] += p_lastili*number
        if last_trans2 == 1:
            res2[i*len(t1):(i+1) * len(t1)] += p_isi*number
            res1[i*len(t1):(i+1) * len(t1)] += p_lastili * number


    plt.plot(t,res1)
    plt.show()
    plt.plot(t,res2)
    plt.show()
#  #################################################################
#  Python code to reproduce the works on RA-GAN research for end-to-end communications.
#  In AWGN/Rayleigh fading channel model
##  #################################################################
import mmap

import scipy.io as sio                     # import scipy.io for .mat file I/O 
import numpy as np                         # import numpy
import function_autoencoder as af          # import our function file
import AWGN_mc as dfp_mc
import AWGN_e_gan as dfp_e_gan
import AWGN_gan as dfp_gan
import AWGN_zf as dfp_zf
import AWGN_map as dfp_map
import matplotlib.pyplot as plt
import random
import math
from torchsummary import summary
from scipy.io import loadmat
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# Problem Setup
k = 1
n = 1 #经过tans编码后为7位
M = 2**k #16位one-hot向量
num_train = 10000    # number of training samples
num_test = 100000    # number of testing samples
training_epochs = 100 #100#800#80


# Training Eb/N0
# AWGN:3dB Rayleigh fading:13dB DeepMIMO:13dB 

n_train = np.arange(5000,55000,5000)
h = 2e-4
d = 2e-4
Tx1 = af.Tx(af.Point(0, 0, 0))
Tx2 = af.Tx(af.Point(0, 0, h))
Rx1 = af.Rx(af.Point(d, 0, 0))
Rx2 = af.Rx(af.Point(d, 0, h))

# Training processing

location = "./DNNmodel/AWGN_"

# train_time = len(n_train)
# for time in range(train_time):
#
#     # Generate training data
#     # 分别为：为1的下标，one-hot,one-hot,x&y相同。每次都使用不同的种子，每次都是随机的
#     sym_index_train, X_train, Y_train = af.generate_transmit_data(M, num_train, seed=random.randint(0, 1000))
#
#     Tx1.send(sym_index_train[0:num_train // 2],n_train[time])
#     Tx2.send(sym_index_train[num_train // 2:num_train],n_train[time])
#     mc_data_train = af.gen_traindata(n_train[time],sym_index_train,Tx1,Tx2,d,h)
#
#     dfp_mc.train(Tx1, Tx2, d, h, mc_data_train, sym_index_train, X_train, Y_train, M, k, n, n_train[time],
#                  location, training_epochs, batch_size=256)  # 500#256#256
#     dfp_gan.train(Tx1, Tx2, d, h, mc_data_train, sym_index_train, X_train, Y_train, M, k, n, n_train[time], location,
#                   training_epochs, batch_size=256)
#     dfp_e_gan.train(Tx1, Tx2, d, h, mc_data_train, sym_index_train, X_train, Y_train, M, k, n, n_train[time],
#                     location, training_epochs, batch_size=256)
#


SNR_dB = np.arange(5000,55000,5000)
ber = np.zeros([5,SNR_dB.shape[0]])
print('EPOCH:    TESTING:   MAP      ZF    DNN       GAN      E-GAN   ')
for i_snr in range(SNR_dB.shape[0]):
    map_list = []
    dnn_list = []
    zf_list = []
    gan_list = []
    egan_list = []
    for time in range(1):
        SNR = SNR_dB[i_snr]
        sym_index_test, X_test, Y_test = af.generate_transmit_data(M, num_test, seed=random.randint(0, 1000))
        Tx1.send(sym_index_test[0:num_test // 2],SNR)
        Tx2.send(sym_index_test[num_test // 2:num_test],SNR)
        mc_data_test = af.gen_testdata(SNR, sym_index_test,Tx1,Tx2,d,h)

        map_list.append(dfp_map.decode(mc_data_test, SNR))

        zf_list.append(dfp_zf.decode(mc_data_test, SNR))

        Y_pred = dfp_mc.test(mc_data_test, sym_index_test, X_test, M, k, n, SNR, location)
        dnn_list.append(af.BER(n, M, sym_index_test, Y_pred, num_test))

        Y_pred = dfp_gan.test(mc_data_test, sym_index_test, X_test, M, k, n, SNR, location)
        gan_list.append(af.BER(n, M, sym_index_test, Y_pred, num_test))

        Y_pred = dfp_e_gan.test(mc_data_test, sym_index_test, X_test, M, k, n, SNR, location)
        egan_list.append(af.BER(n, M, sym_index_test, Y_pred, num_test))

        print('ber at epoch %d: N=%d: %0.6f  %0.6f %0.6f  %0.6f  %0.6f' % ( time, SNR_dB[i_snr],  np.mean(map_list),np.mean(zf_list),np.mean(dnn_list), np.mean(gan_list), np.mean(egan_list)))
    ber[0, i_snr] = np.mean(map_list)
    ber[1, i_snr] = np.mean(zf_list)
    ber[2,i_snr] = np.mean(dnn_list)
    ber[3, i_snr] = np.mean(gan_list)
    ber[4, i_snr] = np.mean(egan_list)




# for i_snr in range(SNR_dB.shape[0]):
#     SNR = SNR_dB[i_snr]
#     sym_index_test, X_test, Y_test = af.generate_transmit_data(M, num_test, seed=random.randint(0,1000))
#     mc_data_test = af.gen_data(SNR, sym_index_test)
#     Y_pred = dfp_optimal.test(mc_data_test,sym_index_test,X_test, M, k, n, SNR, location)
#     ber[0,i_snr] = af.BER_my(mc_data_test)
#     Y_pred = dfp_gan.test(mc_data_test,sym_index_test,X_test, M, k, n, SNR, location)
#     ber[1,i_snr] = af.BER(n, M, sym_index_test, Y_pred, num_test)
#     Y_pred = dfp_e_gan.test(mc_data_test,sym_index_test,X_test, M, k, n, SNR, location)
#     ber[2,i_snr] = af.BER(n, M, sym_index_test, Y_pred, num_test)
#     Y_pred = dfp_rl.test(mc_data_test,sym_index_test,X_test, M, k, n, SNR, location)
#     ber[3,i_snr] = af.BER(n, M, sym_index_test, Y_pred, num_test)
#     print('ber at SNR=%d: %0.6f  %0.6f  %0.6f  %0.6f ' %(SNR_dB[i_snr],ber[0,i_snr],ber[1,i_snr],ber[2,i_snr],ber[3,i_snr]))

#嵌入子坐标系

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.semilogy(SNR_dB,ber[0,:],'p:',linewidth=1.5)
ax.semilogy(SNR_dB,ber[1,:],'b^-.',linewidth=1.5)
ax.semilogy(SNR_dB,ber[4,:],'rv-.',linewidth=1.5)
ax.semilogy(SNR_dB,ber[2,:],'k--.',linewidth=1.5)
ax.semilogy(SNR_dB,ber[3,:],'go-.',linewidth=1.5)


axins = ax.inset_axes((0.09, 0.07, 0.3, 0.4), xticks=SNR_dB)
axins.plot(SNR_dB,ber[0,:],'p:',linewidth=1)
axins.plot(SNR_dB,ber[1,:],'b^-.',linewidth=1)
axins.plot(SNR_dB,ber[4,:],'rv-.',linewidth=1)
axins.plot(SNR_dB,ber[2,:],'k--.',linewidth=1)
axins.plot(SNR_dB,ber[3,:],'go-.',linewidth=1)

#

# # 调整子坐标系的显示范围
# 设置放大区间
zone_left = 1
zone_right = 2

# 坐标轴的扩展比例（根据实际数据调整）
x_ratio = 0.05  # x轴显示范围的扩展比例
# y_ratio = 0.8  # y轴显示范围的扩展比例
#
# # X轴的显示范围
xlim0 = SNR_dB[zone_left] - (SNR_dB[zone_right] - SNR_dB[zone_left]) * x_ratio
xlim1 = SNR_dB[zone_right] + (SNR_dB[zone_right] - SNR_dB[zone_left]) * x_ratio
#
# # Y轴的显示范围
# y = np.hstack((ber[0,:][zone_left:zone_right], ber[1,:][zone_left:zone_right], ber[2,:][zone_left:zone_right],ber[3,:][zone_left:zone_right], ber[4,:][zone_left:zone_right]))
# ylim0 = np.min(y) - (np.max(y) - np.min(y)) * y_ratio
# ylim1 = np.max(y) + (np.max(y) - np.min(y)) * y_ratio

axins.set_xlim(xlim0, xlim1)
axins.set_ylim(0.15, 0.3)

plt.axis([5000,50000,0.0,1])
label = [ "MAP training","ZF training", "E-GAN based training","DNN-based training", "GAN-based training"]
plt.legend(label, loc = 1, ncol = 1)
plt.grid(True) ##增加格点
plt.xlabel('Molecules per bit')
plt.ylabel('BER')
plt.savefig("E:\\Desktop\\driftMIMO\\图\\Number",dpi=600)
plt.show()


sio.savemat('./data/bler_awgn', {'SNR_dB': SNR_dB, 'ber': ber})
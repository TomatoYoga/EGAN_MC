#!/usr/bin/env python

# Generative Adversarial Networks (GAN) example in PyTorch.
# See related blog post at https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f#.sch4xgsa9

import random
import function_autoencoder as af
import numpy as np
import Regularization as RZ

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init
import torch.autograd as autograd
import matplotlib.pyplot as plt
import scipy.io

#import robust_loss_pytorch 

# Data params
t_learning_rate = 0.0001  #GANs 0.0001 Resnet-GANS 0.001   
g_learning_rate = 0.0005  #GANs 0.0005 Resnet-GANS 0.0005
                          #GANs xavier_normal
                          #Resnet-GANS None
optim_betas = (0.9, 0.999)
print_interval = 30
weight_gain = 1
bias_gain = 0.1
g_weight_gain = 0.1
g_bias_gain = 0.1
weight_decay=0.000000000000000000000000000000000000000000000001  # regularization parameter Resnet-GANs: 0.01 GANs:0

# ##### MODELS: Generator model and discriminator model
class Transmitter(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Transmitter, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, output_size)
#        torch.nn.init.xavier_normal_(self.map1.weight, weight_gain)
#        torch.nn.init.constant(self.map1.bias, bias_gain)
#        torch.nn.init.xavier_normal_(self.map2.weight, weight_gain)
#        torch.nn.init.constant(self.map2.bias, bias_gain)

    def forward(self, x):
        x = F.relu(self.map1(x))
        return self.map2(x)
    
class Receiver(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Receiver, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, output_size)
#        torch.nn.init.xavier_normal_(self.map1.weight, weight_gain)
#        torch.nn.init.constant(self.map1.bias, bias_gain)
#        torch.nn.init.xavier_normal_(self.map2.weight, weight_gain)
#        torch.nn.init.constant(self.map2.bias, bias_gain)
        
    def forward(self, x):
        x = F.relu(self.map1(x))
        return F.softmax(self.map2(x))
    
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.n = int(input_size/2)
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        
#        torch.nn.init.xavier_normal_(self.map1.weight, g_weight_gain)
#        torch.nn.init.constant(self.map1.bias, g_bias_gain)
#        torch.nn.init.xavier_normal_(self.map2.weight, g_weight_gain)
#        torch.nn.init.constant(self.map2.bias, g_bias_gain)
#        torch.nn.init.xavier_normal_(self.map3.weight, g_weight_gain)
#        torch.nn.init.constant(self.map3.bias, g_bias_gain)

    def forward(self, x):
        x1 = F.elu(self.map1(x))
        x3 = F.tanh(self.map2(x1))#
        return self.map3(x3)

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        #self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
#        torch.nn.init.xavier_normal_(self.map1.weight, weight_gain)
#        torch.nn.init.constant(self.map1.bias, bias_gain*1)
##        torch.nn.init.xavier_normal_(self.map2.weight, weight_gain)
##        torch.nn.init.constant(self.map2.bias, bias_gain*1)
#        torch.nn.init.xavier_normal_(self.map3.weight, weight_gain)
#        torch.nn.init.constant(self.map3.bias, bias_gain*1)

    def forward(self, x):
        x = F.elu(self.map1(x))
        #x = F.elu(self.map2(x))
        return F.sigmoid(self.map3(x))

def extract(v):
    return v.data.storage().tolist()

def stats(d):
    return [np.mean(d), np.std(d)]

def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)

def ini_weights(M,n):
    T = Transmitter(input_size=M, hidden_size=2*M, output_size=2*n)
    R = Receiver(input_size=2*n, hidden_size=4*M, output_size=M)
    G = Generator(input_size=2*n+2*n, hidden_size=128, output_size=2*n)
    D = Discriminator(input_size=2*n, hidden_size=32, output_size=1)
    return T, R, G, D

def Channel(t_data, n, noise, h_real, cuda_gpu):
    num = t_data.shape[0]
    tray_data = torch.empty(num,2*n)
    if(cuda_gpu):
        tray_data = tray_data.cuda()
    tray_data[:,0:2*n] = t_data[:,0:2*n]#*h_real
    r_data = tray_data + noise
    return r_data

def weight_list(model_name_param_dim):
    dim  = model_name_param_dim[1]
    size = len(model_name_param_dim[0])
    weight = torch.empty([1,dim])
    m = 0
    for name, w in model_name_param_dim[0]:
        n = w.shape[0]*w.shape[1]
        weight[0,m:m+n] = torch.reshape(w,[1,n])
        m = m + n
    return weight
    

gi_sampler = get_generator_input_sampler()
criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
c1 = nn.MSELoss()

def train(X, Y, M, k, n, SNR, location, training_epochs=300, batch_size=1000, LR= 0.001, traintestsplit = 0.0, LRdecay=0):
    X_train = np.transpose(X)     # training data
    Y_train = np.transpose(Y)
    num_total = X.shape[1] 
    total_batch = int(num_total / batch_size)
    
    gpus = [0]   
    cuda_gpu = torch.cuda.is_available()  
    cuda_gpu = False
    T, R, G, D = ini_weights(M, n)
    if(cuda_gpu):
        T = torch.nn.DataParallel(T, device_ids=gpus).cuda()  
        R = torch.nn.DataParallel(R, device_ids=gpus).cuda()   
        G = torch.nn.DataParallel(G, device_ids=gpus).cuda()   
        D = torch.nn.DataParallel(D, device_ids=gpus).cuda()   
    t_optimizer = optim.Adam(T.parameters(), lr=t_learning_rate, betas=optim_betas)
    r_optimizer = optim.Adam(R.parameters(), lr=t_learning_rate, betas=optim_betas)
    g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)
    d_optimizer = optim.Adam(D.parameters(), lr=g_learning_rate, betas=optim_betas)
    
    if weight_decay>0:
        reg_t = RZ.Regularization(T, weight_decay, p=2)
        reg_r = RZ.Regularization(R, weight_decay, p=2)
        reg_g = RZ.Regularization(G, weight_decay, p=2)
        reg_d = RZ.Regularization(D, weight_decay, p=2)
    else:
        print("no regularization")
        
#    t_adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims = reg_t.get_weight(T)[1], float_dtype=np.float32, device = 'cpu')
#    r_adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims = reg_r.get_weight(R)[1], float_dtype=np.float32, device = 'cpu')
#    g_adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims = reg_g.get_weight(G)[1], float_dtype=np.float32, device = 'cpu')
#    d_adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims = reg_d.get_weight(D)[1], float_dtype=np.float32, device = 'cpu')

    
    #params = list(T.parameters()) + list(adaptive.parameters())
    #loss = torch.mean(adaptive.lossfun((y_i — y)[:,None]))

    #params = list(T.parameters())
    
    D_fake = []
    D_real = []
    G_error = []
    R_error0 = []
    GR_error0 = []
    R_error = []
    GR_error = []
    Acc = []
    loss_best = 100
    
    for epoch in range(training_epochs):
        for index in range(total_batch):
            
            noise_train = torch.from_numpy(np.random.randn(batch_size,2*n)*np.sqrt(1/(2*SNR*k/n))/np.sqrt(2)).float()
            idx = np.random.randint(num_total,size=batch_size)
            h_real = 1#random.normalvariate(0, 1)
            
            # 1. Train D on real+fake
            T.zero_grad()
            R.zero_grad()
            G.zero_grad()
            D.zero_grad()
            
            t_data = T(torch.from_numpy(X_train[idx,:]))
            norm = torch.empty(1,batch_size)
            fake = torch.zeros(batch_size, 1)
            true = torch.ones(batch_size, 1)
            target = torch.from_numpy(Y_train[idx,:])
            if(cuda_gpu):
                noise_train = noise_train.cuda()
                t_data = t_data.cuda()
                norm = norm.cuda()
                fake = fake.cuda()
                true = true.cuda()
                target = target.cuda()
                
            norm[0,:] = torch.norm(t_data,2,1)
            t_data = t_data/torch.t(norm)*np.sqrt(n)
            r_data = Channel(t_data, n, noise_train, h_real, cuda_gpu)
            #r_data = t_data + noise_train
            
            #  1A: Train D on real
            d_real_decision = D(r_data.detach())
            #d_real_error0 = torch.mean(robust_loss_pytorch.general.lossfun(d_real_decision-Variable(true), alpha=torch.Tensor([2.]), scale=torch.Tensor([0.1])))
            d_real_error0 = criterion(d_real_decision, Variable(true))
            #d_real_reg = torch.sum(d_adaptive.lossfun(weight_list(reg_d.get_weight(D))))
            d_real_error = d_real_error0 + reg_d(D)#d_real_reg # ones = true
            d_real_error.backward() # compute/store gradients, but don't change params
            
            #  1B: Train D on fake
            d_gen_input = Variable(gi_sampler(batch_size, 2*n))
            if(cuda_gpu):
                d_gen_input = d_gen_input.cuda()
            g_data = G(torch.cat((t_data, d_gen_input), 1))  # detach to avoid training G on these labels

            d_fake_data = g_data.detach()
            d_fake_decision = D(d_fake_data)
            #d_fake_error0 = torch.mean(robust_loss_pytorch.general.lossfun(d_fake_decision-Variable(fake), alpha=torch.Tensor([2.]), scale=torch.Tensor([0.1])))
            d_fake_error0 = criterion(d_fake_decision, Variable(fake)) 
            #d_fake_reg = torch.sum(d_adaptive.lossfun(weight_list(reg_d.get_weight(D))))
            d_fake_error = d_fake_error0 + reg_d(D)#d_fake_reg  # zeros = fake
            d_fake_error.backward()
            d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()
            
            # 2. Train G on D's response (but DO NOT train D on these labels)
            G.zero_grad()
            gen_input = Variable(gi_sampler(batch_size, 2*n))
            if(cuda_gpu):
                gen_input = gen_input.cuda()
            g_fake_data = G(torch.cat((t_data, gen_input), 1))
            dg_fake_decision = D(g_fake_data)
            g_error0 = criterion(dg_fake_decision, Variable(true))             
            #g_reg = torch.sum(g_adaptive.lossfun(weight_list(reg_g.get_weight(G))))
            g_error = g_error0 + reg_g(G)#g_reg   # we want to fool, so pretend it's all genuine
            g_error.backward(retain_graph=True)
            g_optimizer.step()  # Only optimizes G's parameters
            
            #r_decision = R(torch.cat((t_data, r_data), 1))
            T.zero_grad()
            R.zero_grad()
            r_decision = R(r_data)
            #r_error0 = torch.mean(robust_loss_pytorch.general.lossfun(r_decision-target, alpha=torch.Tensor([2.]), scale=torch.Tensor([0.1])))
            r_error0 = criterion(r_decision, target)             
            #r_reg = torch.sum(r_adaptive.lossfun(weight_list(reg_r.get_weight(R))))
            r_error = r_error0 + reg_r(R)#r_reg # ones = true
            r_error.backward(retain_graph=True) # compute/store gradients, but don't change params
            r_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()
            
            T.zero_grad()
            r_gen_input = Variable(gi_sampler(batch_size, 2*n))
            if(cuda_gpu):
                r_gen_input = r_gen_input.cuda()
            g_data = G(torch.cat((t_data, r_gen_input), 1))
            gr_decision = R(g_data)
            #gr_error0 = torch.mean(robust_loss_pytorch.general.lossfun(gr_decision-target, alpha=torch.Tensor([2.]), scale=torch.Tensor([0.1])))
            gr_error0 = criterion(gr_decision, target) 
            #gr_reg = torch.sum(t_adaptive.lossfun(weight_list(reg_t.get_weight(T))))
            gr_error = gr_error0 + reg_t(T)#gr_reg
            gr_error.backward()
            t_optimizer.step()
            
            D_fake.append(extract(d_fake_error0))
            D_real.append(extract(d_real_error0))
            G_error.append(extract(g_error0))
            R_error0.append(extract(r_error0))
            GR_error0.append(extract(gr_error0))
            R_error.append(extract(r_error))
            GR_error.append(extract(gr_error))                                              
            if loss_best>r_error.detach().cpu().numpy():
                torch.save(T.state_dict(), location+"t_gan.pkl")  
                torch.save(R.state_dict(), location+"r_gan.pkl")  
                torch.save(G.state_dict(), location+"g_gan.pkl") 
                torch.save(D.state_dict(), location+"d_gan.pkl")
                loss_best = r_error.detach().cpu().numpy()
    #---------------------------------------------------------------------------------------------------------
            
        if epoch % print_interval == 0:
            
            t_data = torch.autograd.Variable(t_data)
            r_data = torch.autograd.Variable(r_data)
            g_data = torch.autograd.Variable(d_fake_data)
            
            #print(c1(t_data,r_data)," ",c1(t_data,g_data))
            print("GAN: epoch: %s: d_real_error: %0.5f d_fake_error: %0.5f g_error: %0.5f" % (epoch, d_real_error.detach().cpu().numpy(), d_fake_error.detach().cpu().numpy(), g_error.detach().cpu().numpy()))
            print("     epoch: %s: r_error: %0.5f gr_error: %0.5f" % (epoch, r_error.detach().cpu().numpy(), gr_error.detach().cpu().numpy()))
            #print("%s: D:(Real:, Fake:  ) " % (epoch))   
                
            num_vali = 10000
            SNR_vali = np.array([-10,-4,0,4,7,10])
            ber = np.zeros(SNR_vali.shape)
            for i_snr in range(SNR_vali.shape[0]):
                SNR = 10**(SNR_vali[i_snr]/10)
                sym_index_test, X_test, Y_test = af.generate_transmit_data(M, num_vali, seed=random.randint(0,1000))
                Y_pred = test(X_test, M, k, n, SNR, location)
                ber[i_snr] = af.BER(n, M, sym_index_test, Y_pred, num_vali)
                print('      The BER at SNR=%d is %0.6f' %(SNR_vali[i_snr], ber[i_snr]))  
            Acc.append(ber[i_snr])
            
#    plt.plot(D_fake)
#    plt.show()
#    plt.plot(D_real)
#    plt.show()
#    plt.plot(G_error)
#    plt.show()      
#    plt.plot(R_error)
#    plt.show()
#    plt.plot(R_error0)
#    plt.show()
#    plt.plot(GR_error)
#    plt.show()
#    plt.plot(Acc)
#    plt.show()   
    
#    scipy.io.savemat('R_error0_res_awgn.mat', mdict={'R_error0': R_error0})
#    scipy.io.savemat('GR_error0_res_awgn.mat', mdict={'GR_error0': GR_error0})
    scipy.io.savemat('./data/R_error_gan_awgn.mat', mdict={'R_error': R_error})
    scipy.io.savemat('./data/GR_error_gan_awgn.mat', mdict={'GR_error': GR_error})
#    scipy.io.savemat('D_fake_res_awgn.mat', mdict={'D_fake': D_fake})
#    scipy.io.savemat('D_real_res_awgn.mat', mdict={'D_real': D_real})
#    scipy.io.savemat('G_error_res_awgn.mat', mdict={'G_error': G_error})
#    scipy.io.savemat('Acc_res_awgn.mat', mdict={'Acc': Acc})

            
def test(X, M, k, n, SNR, location):
    gpus = [0]  
    cuda_gpu = torch.cuda.is_available()   
    cuda_gpu = False
    T, R, G, D = ini_weights(M, n)
    if(cuda_gpu):
        T = torch.nn.DataParallel(T, device_ids=gpus).cuda()  
        R = torch.nn.DataParallel(R, device_ids=gpus).cuda()  
        G = torch.nn.DataParallel(G, device_ids=gpus).cuda()  
        D = torch.nn.DataParallel(D, device_ids=gpus).cuda()  
    T.load_state_dict(torch.load(location+'t_gan.pkl'))
    R.load_state_dict(torch.load(location+'r_gan.pkl'))
    G.load_state_dict(torch.load(location+'g_gan.pkl'))
    D.load_state_dict(torch.load(location+'d_gan.pkl'))
    
    T.zero_grad()
    R.zero_grad()
    
    X_test = np.transpose(X) 
    num_test = X.shape[1]
    noise_test = torch.from_numpy(np.random.randn(num_test,2*n)*np.sqrt(1/(2*SNR*k/n))/np.sqrt(2)).float()
    h_real = np.random.randn(num_test,1)
    H_real = torch.from_numpy(np.ones((num_test,n))*h_real).float()
    
    t_data = T(torch.from_numpy(X_test))
    norm = torch.empty(1,num_test)
    if(cuda_gpu):
        noise_test = noise_test.cuda()
        t_data = t_data.cuda()
        norm = norm.cuda()
        H_real = H_real.cuda()
        
    norm[0,:] = torch.norm(t_data,2,1)
    t_data = t_data/torch.t(norm)*np.sqrt(n)
    r_data = Channel(t_data, n, noise_test, H_real, cuda_gpu)
    #r_data = t_data + noise_test
    
    r_gen_input = Variable(gi_sampler(num_test, 2*n))
    if(cuda_gpu):
        r_gen_input = r_gen_input.cuda()
    g_data = G(torch.cat((t_data, r_gen_input), 1))
        
    t_data = torch.autograd.Variable(t_data)
    r_data = torch.autograd.Variable(r_data)
    g_data = torch.autograd.Variable(g_data)
            
    r_decision = R(r_data)
    pred = r_decision.detach().cpu().numpy()
    return np.transpose(pred)
   
    
    
import math
import torch
import torch.nn as nn
import numpy as np
#from torch.distributions import Gumbel,Bernoulli,Normal

class DSVM(nn.Module):
    def __init__(self,y_dim,h_dim,z_dim,n_layers,device,bias=False):
        super(DSVM, self).__init__()   
        
        self.y_dim = y_dim  
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.device = device
        
        # z trainsition
        self.ztrainsition = nn.Sequential(
            nn.Linear(z_dim,2*z_dim),
#             nn.ReLU(),
#             nn.Linear(2*z_dim, 2*z_dim),
            nn.ReLU())
        self.ztrainsition_mean = nn.Linear(2*z_dim, z_dim)
        self.ztrainsition_std = nn.Sequential(
            nn.Linear(2*z_dim, z_dim),
            nn.Softplus())   
        
        # y emission
        self.yemission = nn.Sequential(
            nn.Linear(h_dim ,2*y_dim),
#             nn.ReLU(),
#             nn.Linear(2*y_dim, 2*y_dim),
            nn.ReLU())
        #self.yemission_mean = nn.Linear(y_dim, y_dim)
        self.yemission_std = nn.Sequential(
            nn.Linear(2*y_dim, y_dim),
            nn.Softplus()) 
        
        # z posterior
        self.zposterior = nn.Sequential(
            nn.Linear(z_dim + h_dim, 2*z_dim),
            nn.ReLU(),
            nn.Linear(2*z_dim,2*z_dim),
            nn.ReLU())
        self.zposterior_mean = nn.Linear(2*z_dim, z_dim)
        self.zposterior_std = nn.Sequential(
            nn.Linear(2*z_dim, z_dim),
            nn.Softplus())   

        #recurrence
        self.rnn_forward = nn.GRU(2*y_dim + z_dim, h_dim, n_layers, bidirectional=False)
        self.rnn_backward = nn.GRU(y_dim, h_dim, n_layers, bidirectional=False)


    def forward(self,y):

        all_z_posterior_mean, all_z_posterior_std = [], []
        all_z_prior_mean, all_z_prior_std = [], []
        all_z_t_sampled = []
        
        all_h = []
        
        all_y_emission_mean, all_y_emission_std = [], []
        
        kld_gaussian_loss = 0
        nll_loss = 0
        
        z0 = torch.zeros((y.size(1),self.z_dim),device = self.device)
        z0_std = torch.zeros((y.size(1),self.z_dim),device = self.device)
        
        h_forward = torch.zeros((self.n_layers, y.size(1), self.h_dim),device = self.device)
        A0 = torch.zeros((self.n_layers, y.size(1), self.h_dim),device = self.device)
        
        y_emission_mean = torch.zeros((y.size(1),self.y_dim),device = self.device)
        y_emission_std = torch.zeros((y.size(1),self.y_dim),device = self.device)

        all_z_posterior_mean.append(z0)
        all_z_posterior_std.append(z0_std)
            
        all_z_prior_mean.append(z0)
        all_z_prior_std.append(z0_std)
            
        all_z_t_sampled.append(z0)
            
        all_h.append(h_forward)
        
        all_y_emission_std.append(y_emission_std)
        all_y_emission_mean.append(y_emission_mean)
    
        ## Backward Rnn
        ### Reverse of copy of numpy array of given tensor
        yinverse = torch.flip(y,[0])
        output_backward, h_backward = self.rnn_backward(yinverse,A0)
        
        for t in range(y.size(0)):
            
            #print(t)
            
            # z prior
            z_prior = self.ztrainsition(all_z_t_sampled[-1])
            z_prior_mean = self.ztrainsition_mean(z_prior)
            z_prior_std = self.ztrainsition_std(z_prior)
            all_z_prior_std.append(z_prior_std)
            all_z_prior_mean.append(z_prior_mean)
                
            # z posterior
            z_posterior = self.zposterior(torch.cat([output_backward[y.size(0)-t-1],all_z_t_sampled[-1]],1))
            z_posterior_mean =  self.zposterior_mean(z_posterior)  #Res parameterization
            z_posterior_std =  self.zposterior_std(z_posterior)         
            all_z_posterior_std.append(z_posterior_std)
            all_z_posterior_mean.append(z_posterior_mean)
            
            #sampling and reparameterization for the continuous variable
            z_t = self._reparameterized_normal_sample(z_posterior_mean, z_posterior_std)
            all_z_t_sampled.append(z_t)
            #print("z_t_sampled:",z_t,z_t.size())
            
            ## Forward RNN
            #print(torch.unsqueeze(y[t],axis=0).shape,torch.unsqueeze(all_z_t_sampled[t+1],axis=0).shape,all_sigma[t].shape) #print(torch.unsqueeze(torch.cat([y[t],all_z_t_sampled[t+1],all_y_emission_std[t]],1),axis=0).size(),torch.cat([y[t],all_z_t_sampled[t+1],all_y_emission_std[t]],1).size())
            if t == 0 :
                output_forward, h_forward = self.rnn_forward(torch.unsqueeze(torch.cat([torch.zeros_like(y[t]),all_z_t_sampled[-1],all_y_emission_std[-1]],1),0), h_forward)
            else:
                output_forward, h_forward = self.rnn_forward(torch.unsqueeze(torch.cat([y[t-1],all_z_t_sampled[-1],all_y_emission_std[-1]],1),0), h_forward)
            all_h.append(h_forward)
            #print(output_forward.size(),h_forward.size())
            
            # y emission
            y_emission = self.yemission(torch.squeeze(all_h[-1], 0))
            #y_emission_mean = self.yemission_mean(y_emission)
            y_emission_std = self.yemission_std(y_emission) 
            all_y_emission_mean.append(y_emission_mean)
            all_y_emission_std.append(y_emission_std)
            #print(z_posterior_std.shape,all_h[].shape,y_emission_std.shape)
            
            #computing losses
            kld_gaussian_loss += self._kld_gauss(z_posterior_mean, z_posterior_std, z_prior_mean, z_prior_std)
            nll_loss += self._nll_gauss(y_emission_mean, y_emission_std, y[t])
        
        return kld_gaussian_loss,nll_loss,(all_z_posterior_mean, all_z_posterior_std),(all_z_prior_mean, all_z_prior_std),all_z_t_sampled,all_y_emission_mean,all_y_emission_std,all_h

    def _forecastingMultiStep(self,y,S=10,step=1):
        
        with torch.no_grad():
            
            forecast_y_MC = []
            forecast_vol_MC = []
            forecast_z_mean_MC = []
            forecast_z_std_MC = []
            forecast_z_MC = []
            
            for s in range(S):
                all_y  = []
                all_y.append(y[-1]) # the first elements of the all_y is the last training sample
                
                forecast_y_steps = [] # forecast_y_steps contains all the forecasting samples
                forecast_vol_steps = []
                forecast_z_mean_steps = []
                forecast_z_std_steps = []
                forecast_z_steps = []
                
                kld_gaussian_loss,nll_loss,(all_z_posterior_mean, all_z_posterior_std),(all_z_prior_mean, all_z_prior_std),all_z_t_sampled,all_y_emission_mean, all_y_emission_std, all_h = self.forward(y)
                
                for t in range(step):

                    #print(h_forward.squeeze().size(),output_forward[t].size())
                    z_prior = self.ztrainsition(all_z_t_sampled[-1])
                    z_prior_mean =  self.ztrainsition_mean(z_prior)
                    z_prior_std =  self.ztrainsition_std(z_prior)

                    z_t = torch.distributions.Normal(z_prior_mean,z_prior_std).sample()
                    all_z_t_sampled.append(z_t)

                    #print(all_y[-1].size(),all_z_t_sampled[-1].size(),all_z_t_sampled[-1].size(),all_y_emission_std[-1].size(),all_h[-1].size())
                    output_forward, h_forward = self.rnn_forward(torch.unsqueeze(torch.cat([all_y[-1],all_z_t_sampled[-1],all_y_emission_std[-1]],1),0), all_h[-1])
                    all_h.append(h_forward)

                    #print("z_t_sampled:",z_t_sampled[t])
    #                 all_z_posterior_mean.append(z_prior_mean)
    #                 all_z_posterior_std.append(z_prior_std)

                    ## y emission
                    y_emission = self.yemission(torch.squeeze(all_h[-1], 0))
                    y_emission_mean = torch.zeros((y.size(1),self.y_dim),device = self.device)
                    #y_emission_mean = self.yemission_mean(y_emission)
                    y_emission_std = self.yemission_std(y_emission)

                    all_y_emission_mean.append(y_emission_mean)
                    all_y_emission_std.append(y_emission_std)

                    y_t = torch.distributions.Normal(y_emission_mean,y_emission_std).sample()
                    
                    all_y.append(y_t)

                    forecast_y_steps.append(y_t.cpu().numpy())
                    forecast_vol_steps.append(y_emission_std.cpu().numpy())
                    forecast_z_mean_steps.append(z_prior_mean.cpu().numpy())
                    forecast_z_std_steps.append(z_prior_std.cpu().numpy())
                    forecast_z_steps.append(z_t.cpu().numpy())

                forecast_y_MC.append(forecast_y_steps)
                forecast_vol_MC.append(forecast_vol_steps)
                forecast_z_mean_MC.append(forecast_z_mean_steps)
                forecast_z_std_MC.append(forecast_z_std_steps)
                forecast_z_MC.append(forecast_z_steps)
            
            
        return np.array(forecast_y_MC), np.array(forecast_vol_MC),np.array(forecast_z_mean_MC),np.array(forecast_z_std_MC),np.array(forecast_z_MC)
    #forecast_x,all_y_emission_mean,all_y_emission_std,d_t_sampled_plot,z_t_sampled,all_d_posterior,all_z_posterior_mean,all_z_posterior_std

#     def reset_parameters(self, stdv=1e-1):
#         for weight in self.parameters():
#             weight.data.normal_(0, stdv)


#     def _init_weights(self, stdv):
#         pass


    def _reparameterized_normal_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_()
        eps = eps.to(self.device)
        return eps.mul(std).add_(mean)

    def _reparameterized_category_gumbel_softmax_sample(self,logits):
        """using std to sample"""
        #print(self.temperature)
        if self.temperature >0.01:
            self.temperature = self.temperature/1.001
        else:
            self.temperature = 0.01
        #print(logits.size())
        #print(torch.distributions.Gumbel(torch.tensor([0.0]), torch.tensor([1.0])).sample(logits.size()).squeeze().size())
        y = torch.log(logits) + torch.distributions.Gumbel(torch.tensor([0.0],device = self.device), torch.tensor([1.0],device = self.device)).sample(logits.size()).squeeze()
        return torch.nn.functional.softmax((y / self.temperature),dim=1)


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element =  2 * torch.log(std_2) - 2 * torch.log(std_1) + (std_1.pow(2) + (mean_1 - mean_2).pow(2))/std_2.pow(2) - 1
        return 0.5 * torch.sum(kld_element)
    
    
    def _kld_category(self,d_posterior, d_prior):
        #print(d_posterior,d_prior)
        #print(torch.div(d_posterior,d_prior))
        return torch.sum(torch.mul(torch.log(torch.div(d_posterior,d_prior)),d_posterior))
        
    
    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x*torch.log(theta) + (1-x)*torch.log(1-theta))


    def _nll_gauss(self, mean, std, x):
        return  torch.sum(0.5*torch.log(torch.tensor(2*math.pi,device = self.device))+torch.log(std) + (x-mean).pow(2)/(2*std.pow(2)))
    
    def _one_hot_encode(self,x, n_classes):
        """One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
        x: List of sample Labels
        return: Numpy array of one-hot encoded labels
        """
        return torch.eye(n_classes,device = self.device)[x]
    
    
def train(model,optimizer,trainY,epoch,batch_size,status="train"):
    model.train()
    #forward + backward + optimize
    for batch in range(0,trainY.size(1),batch_size):
    #for batch in np.random.choice(range(trainX.size(1)),trainX.size(1), replace=False):
        optimizer.zero_grad()
        kld_gaussian_loss,nll_loss,(all_z_posterior_mean, all_z_posterior_std),(all_z_prior_mean, all_z_prior_std), all_z_t_sampled,all_y_emission_std,all_y_emission_std,all_h = model(trainY[:,batch:(batch+batch_size),:])
            
        kld_loss = kld_gaussian_loss
        loss = kld_loss + nll_loss
     
        loss.backward()
        
        #grad norm clipping, only in pytorch version >= 1.10
        #nn.utils.clip_grad_norm_(model.parameters(), clip)        
        optimizer.step()
        
    average_loss,(all_z_posterior_mean,all_z_posterior_std), (all_z_prior_mean, all_z_prior_std), all_z_t_sampled,all_y_emission_mean, all_y_emission_std, all_h = test(model,trainY,epoch,"train")
    
    return average_loss,(all_z_posterior_mean,all_z_posterior_std),(all_z_prior_mean, all_z_prior_std), all_z_t_sampled, all_y_emission_mean, all_y_emission_std, all_h

def test(model,testY,epoch,status="test"):
    """uses test data to evaluate 
    likelihood of the model"""
    model.eval()
    with torch.no_grad():
        size = testY.size(1)
        kld_gaussian_loss,nll_loss,(all_z_posterior_mean, all_z_posterior_std), (all_z_prior_mean, all_z_prior_std), all_z_t_sampled, all_y_emission_mean, all_y_emission_std, all_h = model(testY)
        average_kld_gaussian_loss = kld_gaussian_loss.item()/size 
        average_nll_loss = nll_loss.item()/size
        average_loss = (average_kld_gaussian_loss + average_nll_loss)
        print('{} Epoch:{}\t KLD_Gaussian Loss: {:.6f}, NLL Loss: {:.6f}, Loss: {:.4f}'.format(status,epoch,average_kld_gaussian_loss,average_nll_loss,average_loss))
    return average_loss,(all_z_posterior_mean,all_z_posterior_std), (all_z_prior_mean, all_z_prior_std), all_z_t_sampled,all_y_emission_mean,all_y_emission_std, all_h







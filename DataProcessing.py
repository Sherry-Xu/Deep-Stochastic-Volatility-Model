import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

## Normalize dataset
def normalize_moments(dataset):
    moments = np.zeros(2)
    moments[0] = dataset.mean()
    moments[1] = dataset.std()
    print("Moments:",moments[0],moments[1])
    return moments

def normalize_fit(dataset,moments):
    dataset = (dataset-moments[0])/moments[1]
    return dataset

def normalize_invert(dataset,moments):
    dataset = dataset*moments[1]+moments[0]
    return dataset

def min_max(dataset):
    moments = np.zeros(2)
    moments[0] = dataset.min()
    moments[1] = dataset.max()
    print(moments[0],moments[1])
    return moments

def minmax_fit(dataset,moments):
    dataset = (dataset-moments[0])/(moments[1] - moments[0])
    return dataset

def minmax_invert(dataset,moments):
    dataset = dataset*(moments[1] - moments[0])+moments[0]
    return dataset

# convert an array of values into a dataset matrix from 1-d time series
def create_dataset(dataset, look_back=1,predict_len = 1):
    dataX, dataY = [], []
    for i in range(0,len(dataset)-look_back,predict_len):
            a = dataset[i:(i+look_back), 0]
            b = dataset[(i + predict_len):(i+look_back+predict_len), 0]
            dataX.append(a)
            dataY.append(b)
    return np.array(dataX), np.array(dataY) 

# convert an array of values into a dataset matrix
def create_dataset_supervised(dataset, look_back=1,predict_len=1):
    dataX, dataY = [], []
    for i in range(0,len(dataset)-look_back,predict_len):
            a = dataset[i:(i+look_back), :]
            b = dataset[(i+1):(i+look_back+1),:]
            dataX.append(a)
            dataY.append(b)
    return np.array(dataX), np.array(dataY) 

def evaluation(predict,original):
    ae = np.abs(predict - original)
    se = ae**2
    #index = (original!=0)
    ape = np.abs(ae/original)
    ape[ape == -np.inf] = np.NaN
    ape[ape == np.inf] = np.NaN
    #print(ape)

    mae = np.nanmean(ae)
    rmse = np.sqrt(np.nanmean(se))
    mape = np.nanmean(ape)
    
    plt.figure(figsize=(10,3))
    plt.subplot(121)
    plt.plot(np.mean(ae,axis = 0),label ='MAE varying according to days')
    plt.legend()
    #plt.show();
    plt.subplot(122)
    plt.plot(np.mean(ae,axis = 1),label ='MAE varying according to hours')
    plt.legend()
    plt.show();   

    plt.figure(figsize=(10,3))
    plt.subplot(121)
    plt.plot(np.sqrt(np.mean(se,axis = 0)),label = 'RMSE varying according to days')
    plt.legend()
    #plt.show();

    plt.subplot(122)
    plt.plot(np.sqrt(np.mean(se,axis = 1)),label = 'RMSE varying according to hours')
    plt.legend()
    plt.show(); 

    plt.figure(figsize=(10,3))
    plt.subplot(121)
    plt.plot(np.nanmean(ape,axis = 0),label = 'MAPE varying according to days')
    plt.legend()
    #plt.show(); 

    plt.subplot(122)
    plt.plot(np.nanmean(ape,axis = 1),label = 'MAPE varying according to hours')
    plt.legend()
    #plt.tight_layout()
    plt.show();
    print("mae","rmse","mape")

    return mae,rmse,mape
    
def hour_evaluation(predict,original):
    ae = np.abs(predict - original)
    se = ae**2
    #index = (original!=0)
    ape = np.abs(ae/original)
    ape[ape == -np.inf] = np.NaN
    ape[ape == np.inf] = np.NaN
    #print(ape)

    mae = np.nanmean(ae)
    rmse = np.sqrt(np.nanmean(se))
    mape = np.nanmean(ape)
    
    hour_mae = np.mean(ae,axis = 1)
    hour_rmse = np.sqrt(np.mean(se,axis = 1))
    hour_mape = np.nanmean(ape,axis = 1)
    hour_result = pd.DataFrame({'hour_mae':hour_mae,'hour_rmse': hour_rmse,'hour_mape':hour_mape})
    return hour_result


def prediction(model,testX):
    testForecast,testForecast_std,all_d,all_z,all_d_prior,all_z_prior_mean = model.forecasting(testX)
    all_testForecast = np.array(testForecast.detach().cpu().numpy().squeeze())
    all_testForecast_std = np.array(testForecast_std.detach().cpu().numpy().squeeze())
    all_d = np.array([all_d[i].detach().numpy() for i in range(len(all_d))]).transpose(1,0)
    all_z = np.array([all_z[i].detach().cpu().numpy() for i in range(len(all_z))]).transpose(1,0,2)
    all_d_prior = np.array([all_d_prior[i].detach().cpu().numpy() for i in range(len(all_d_prior))]).transpose(1,0)
    all_z_prior_mean  = np.array([all_z_prior_mean[i].detach().cpu().numpy() for i in range(len(all_z_prior_mean))]).transpose(1,0,2)
    return all_testForecast,all_testForecast_std,all_d,all_z,all_d_prior,all_z_prior_mean  


def PosteriorPlot(i,all_d_t_sampled,all_z_t_sampled,all_d_posterior,all_z_posterior_mean,plot_data,colormap,timestep,predict_dim,colorbar = True):
    
    ## Transform the d and z to be the size of (batch,timestep,dim)
    all_d_t_sampled = np.array([all_d_t_sampled[i].detach().cpu().numpy() for i in range(len(all_d_t_sampled))]).transpose(1,0)
    all_z_t_sampled = np.array([all_z_t_sampled[i].detach().cpu().numpy() for i in range(len(all_z_t_sampled))]).transpose(1,0,2)
    all_d_posterior = np.array([all_d_posterior[i].detach().cpu().numpy() for i in range(len(all_d_posterior))]).transpose(1,0)
    all_z_posterior_mean = np.array([all_z_posterior_mean[i].cpu().detach().numpy() for i in range(len(all_z_posterior_mean))]).transpose(1,0,2)
    
    ## Transform the dataset to be the size of (batch,timestep*inputdim)
    plot_data = plot_data.numpy().transpose((1, 0, 2)).reshape(plot_data.size(1),-1)
    
    ## Make plot
    fig,(ax1,ax3) = plt.subplots(2,1,figsize=(20, 6))
    
    ax1.plot(np.arange((timestep)*predict_dim)/predict_dim+1+1/predict_dim/2,plot_data[i,:],color='#1f77b4',label = "y")
    ax2 = ax1.twinx()
    ax2 = sns.heatmap(all_d_t_sampled[i,:].reshape(1,-1), linewidth=0.5, cbar=colorbar,alpha=0.2, cmap=colormap,vmin=0, vmax=1)
    ax1.legend()
    #ax1.set_title("y")
    
    ax3.plot(np.arange(len(all_z_t_sampled[i,:]))+1/2,all_z_t_sampled[i,:],alpha=1,color='#ff7f0e',label = "Posterior sampled z")
    ax4 = ax3.twinx()
    ax4 = sns.heatmap(all_d_t_sampled[i,:].reshape(1,-1), linewidth=0.5, cbar=colorbar,alpha=0.2, cmap=colormap,vmin=0, vmax=1)
    ax3.legend()
    
    fig.tight_layout()
    
    #plt.suptitle("Posterior samples")
    plt.show();
    ############################
    
    fig,(ax1,ax3) = plt.subplots(2,1,figsize=(20, 6))
    
    ax1.plot(np.arange((timestep)*predict_dim)/predict_dim+1+1/predict_dim/2,plot_data[i,:],color='#1f77b4',label = "y")
    ax2 = ax1.twinx()
    ax2 = sns.heatmap(all_d_posterior[i,:].reshape(1,-1), linewidth=0.5, cbar=colorbar,alpha=0.2, cmap=colormap,vmin=0, vmax=1)
    ax1.legend()
    #ax1.set_title("y")
    
    ax3.plot(np.arange(len(all_z_posterior_mean[i,:]))+1/2,all_z_posterior_mean[i,:],alpha=1,color='#ff7f0e',label="Posterior sampled z")
    ax4 = ax3.twinx()
    ax4 = sns.heatmap(all_d_posterior[i,:].reshape(1,-1), linewidth=0.5, cbar=colorbar,alpha=0.2, cmap=colormap,vmin=0, vmax=1)
    ax3.legend()
    #ax3.set_title("Posterior mean z")
    fig.tight_layout()
    
    #plt.suptitle("Posterior mean")
    plt.show();
    
    print(all_d_t_sampled[i,:],all_d_posterior[i,:])
    
    
def OriginalDataPlot(i,z_original,d_original,y_original2,colormap,timestep,predict_dim):
    fig,ax =plt.subplots(figsize=(20, 3))
    ax.plot(np.arange((timestep)*predict_dim)/predict_dim+1+1/predict_dim/2,y_original2[i:(i+timestep)],label="y",color="#1f77b4")
    ax.plot(np.arange(len(z_original[i:(i+timestep+1)]))+1/2,z_original[i:(i+timestep+1)],label="z",color='#ff7f0e')
    ax2 = ax.twinx()
    ax2 = sns.heatmap(d_original[i:(i+timestep+1)].reshape(1,-1), linewidth=0.5, cbar=False,alpha=0.2, cmap=colormap,vmin=0, vmax=1)
    ax.legend()
    plt.suptitle("Original Data")

def PredictionPriorPlot(i,all_testForecast,all_testForecast_std,all_d_prior_sampled,all_z_prior_sampled,all_d_prior,all_z_prior_mean,testY,colormap,timestep,predict_dim,colorbar = True):

    y_original = testY[:,i,:]
    y_predict = all_testForecast[i]
    y_std = all_testForecast_std[i]
    
    d_t_sampled = all_d_prior_sampled[i,:]
    z_t_sampled = all_z_prior_sampled[i,:]
    
    fig,(ax1,ax3) = plt.subplots(2,1,figsize=(20, 6))
    
    ax1.plot(np.arange(timestep*predict_dim)/predict_dim+1+1/predict_dim/2,y_original.reshape(-1,1))
    ax1.plot(np.arange((timestep-1)*predict_dim,timestep*predict_dim)/predict_dim+1+1/predict_dim/2,y_predict.reshape(-1,1))
    ax1.plot(np.arange((timestep-1)*predict_dim,timestep*predict_dim)/predict_dim+1+1/predict_dim/2,(y_predict-y_std).reshape(-1,1),color='grey',alpha=0.5)
    ax1.plot(np.arange((timestep-1)*predict_dim,timestep*predict_dim)/predict_dim+1+1/predict_dim/2,(y_predict+y_std).reshape(-1,1),color='grey',alpha=0.5)
    ax1.fill_between(np.arange((timestep-1)*predict_dim,timestep*predict_dim)/predict_dim+1+1/predict_dim/2,(y_predict-y_std).reshape(-1),(y_predict+y_std).reshape(-1),color='grey',alpha=0.1)
    ax2 = ax1.twinx()
    ax2 = sns.heatmap(all_d_prior_sampled[i,:].reshape(1,-1), linewidth=0.5, cbar = colorbar,alpha=0.2, cmap = colormap,vmin=0, vmax=1)
    ax1.set_title("Original Time Series and Prediction with Confidence Interval")

    ax3.plot(np.arange(len(all_z_prior_sampled[i,:]))+1/2,all_z_prior_sampled[i,:],alpha=1,label = "Latent Factor z Samples",color='#ff7f0e')
    ax4 = ax3.twinx()
    ax4 = sns.heatmap(all_d_prior_sampled[i,:].reshape(1,-1), linewidth=0.5, cbar = colorbar,alpha=0.2, cmap = colormap,vmin=0, vmax=1)
    ax3.legend()
    
    fig.tight_layout()
    plt.show();
    
    #############

    fig,(ax1,ax3) = plt.subplots(2,1,figsize=(20, 6))
    
    ax1.plot(np.arange(timestep*predict_dim)/predict_dim+1+1/predict_dim/2,y_original.reshape(-1,1))
    ax1.plot(np.arange((timestep-1)*predict_dim,timestep*predict_dim)/predict_dim+1+1/predict_dim/2,y_predict.reshape(-1,1))
    ax1.plot(np.arange((timestep-1)*predict_dim,timestep*predict_dim)/predict_dim+1+1/predict_dim/2,(y_predict-y_std).reshape(-1,1),color='grey',alpha=0.5)
    ax1.plot(np.arange((timestep-1)*predict_dim,timestep*predict_dim)/predict_dim+1+1/predict_dim/2,(y_predict+y_std).reshape(-1,1),color='grey',alpha=0.5)
    ax1.fill_between(np.arange((timestep-1)*predict_dim,timestep*predict_dim)/predict_dim+1+1/predict_dim/2,(y_predict-y_std).reshape(-1),(y_predict+y_std).reshape(-1),color='grey',alpha=0.1)
    ax1.set_title("Original Time Series and Prediction with Confidence Interval")
    ax2 = ax1.twinx()
    ax2 = sns.heatmap(all_d_prior_sampled[i,:].reshape(1,-1), linewidth=0.5, cbar = colorbar,alpha=0.2, cmap = colormap,vmin=0, vmax=1)

    ax3.plot(np.arange(len(all_z_prior_mean[i,:]))+1/2,all_z_prior_mean[i,:],alpha=1,label = "Latent Factor z Prior mean",color='#ff7f0e')
    ax4 = ax3.twinx()
    ax4 = sns.heatmap(all_d_prior[i,:].reshape(1,-1), linewidth=0.5, cbar = colorbar,alpha=0.2, cmap = colormap,vmin=0, vmax=1)
    ax3.legend()
    
    fig.tight_layout()
    plt.show();
    
    print(all_d_prior_sampled[i,:].reshape(1,-1))
    print(all_d_prior[i,:].reshape(1,-1))
    
    
def PosteriorPredictionPlot(i,plot_data,all_d_t_sampled,all_z_t_sampled,all_d_posterior,all_z_posterior_mean,all_y_emission_mean,all_y_emission_std,predict_dim,timestep,forecaststep,colorbar = True,colormap = "Blues"):
    
    y_original  = plot_data[i+forecaststep,-(forecaststep)*predict_dim:]
    y_predict = all_y_emission_mean[i,timestep:timestep+forecaststep,:]
    y_std = all_y_emission_std[i,timestep:timestep+forecaststep,:]
    ## Make plot
    fig,(ax1,ax3) = plt.subplots(2,1,figsize=(20, 6))

    ax1.plot(np.arange((timestep+forecaststep)*predict_dim)/predict_dim+1+1/predict_dim/2,np.concatenate((plot_data[i,:],y_original)),color='#1f77b4')
    ax1.plot(np.arange((timestep)*predict_dim,(timestep+forecaststep)*predict_dim)/predict_dim+1+1/predict_dim/2,y_predict.reshape(-1,1),color = '#ff7f0e')
    ax1.plot(np.arange((timestep)*predict_dim,(timestep+forecaststep)*predict_dim)/predict_dim+1+1/predict_dim/2,(y_predict-y_std).reshape(-1,1),color='grey',alpha=0.5)
    ax1.plot(np.arange((timestep)*predict_dim,(timestep+forecaststep)*predict_dim)/predict_dim+1+1/predict_dim/2,(y_predict+y_std).reshape(-1,1),color='grey',alpha=0.5)
    ax1.fill_between(np.arange((timestep)*predict_dim,(timestep+forecaststep)*predict_dim)/predict_dim+1+1/predict_dim/2,(y_predict-y_std).reshape(-1),(y_predict+y_std).reshape(-1),color='grey',alpha=0.1)
    ax2 = ax1.twinx()
    ax2 = sns.heatmap(all_d_t_sampled[i,:].reshape(1,-1), linewidth=0.5, cbar=colorbar,alpha=0.2, cmap=colormap,vmin=0, vmax=1)
    ax2.vlines(timestep+1,0,1)
    ax1.set_title("Original Data")

    ax3.plot(np.arange(len(all_z_t_sampled[i,:]))+1/2,all_z_t_sampled[i,:],alpha=1,color='#2ca02c',label = "Z Sampled")
    ax4 = ax3.twinx()
    ax4 = sns.heatmap(all_d_t_sampled[i,:].reshape(1,-1), linewidth=0.5, cbar=colorbar,alpha=0.2, cmap=colormap,vmin=0, vmax=1)
    ax4.vlines(timestep+1,0,1)
    ax4.set_title("Z Posterior|Z Prior")

    fig.tight_layout()
    plt.show();

    ############################

    fig,(ax1,ax3) = plt.subplots(2,1,figsize=(20, 6))
    ax1.plot(np.arange((timestep+forecaststep)*predict_dim)/predict_dim+1+1/predict_dim/2,np.concatenate((plot_data[i,:],y_original)),color='#1f77b4')
    ax1.plot(np.arange((timestep)*predict_dim,(timestep+forecaststep)*predict_dim)/predict_dim+1+1/predict_dim/2,y_predict.reshape(-1,1),color = '#ff7f0e')
    ax1.plot(np.arange((timestep)*predict_dim,(timestep+forecaststep)*predict_dim)/predict_dim+1+1/predict_dim/2,(y_predict-y_std).reshape(-1,1),color='grey',alpha=0.5)
    ax1.plot(np.arange((timestep)*predict_dim,(timestep+forecaststep)*predict_dim)/predict_dim+1+1/predict_dim/2,(y_predict+y_std).reshape(-1,1),color='grey',alpha=0.5)
    ax1.fill_between(np.arange((timestep)*predict_dim,(timestep+forecaststep)*predict_dim)/predict_dim+1+1/predict_dim/2,(y_predict-y_std).reshape(-1),(y_predict+y_std).reshape(-1),color='grey',alpha=0.1)
    ax2 = ax1.twinx()
    ax2 = sns.heatmap(all_d_posterior[i,:].reshape(1,-1), linewidth=0.5, cbar=colorbar,alpha=0.2, cmap=colormap,vmin=0, vmax=1)
    ax2.vlines(timestep+1,0,1)
    ax1.set_title("Original Data")

    ax3.plot(np.arange(len(all_z_posterior_mean[i,:]))+1/2,all_z_posterior_mean[i,:],alpha=1,color='#2ca02c')
    ax4 = ax3.twinx()
    ax4 = sns.heatmap(all_d_posterior[i,:].reshape(1,-1), linewidth=0.5, cbar=colorbar,alpha=0.2, cmap=colormap,vmin=0, vmax=1)
    ax4.vlines(timestep+1,0,1)
    ax4.set_title("Z Posterior|Z Prior")

    fig.tight_layout()
    plt.show();

    #print(all_d_t_sampled[i,:],all_d_posterior[i,:])
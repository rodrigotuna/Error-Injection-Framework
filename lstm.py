#@title Load Python libraries
#pip install alpha_vantage -q
#pip install torch
#pip install numpy
#Download M4 dataset. Path should be ./NAB-master/data

import numpy as np
# pip install torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from alpha_vantage.timeseries import TimeSeries 
import numpy as np
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error
from ESRNN.m4_data import *
import json
from error import normal_dist,introduce_errors

torch.manual_seed(0)
np.random.seed(0)
print("All libraries loaded")

torch.manual_seed(0)
config = {
    "Stock": {
        "key": "99UE5LPF59QDCGSY", # Claim your free API key here: https://www.alphavantage.co/support/#api-key
        "symbol": "AMZN",
        "outputsize": "full",
        "key_adjusted_close": "5. adjusted close",
    },
    "data": {
        "window_size": 1,
        "train_split_size": 0.88,
    }, 
    "plots": {
        "show_plots": True,
        "xticks_interval": 90,
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_error": "#FF0000",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 1, # since we are only using 1 feature, close price
        "num_lstm_layers": 1,
        "lstm_size": 12,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu", # "cuda" or "cpu"
        "batch_size": 1,
        "num_epoch": 50,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}

def prepare_data_x(x, window_size):
    # perform windowing
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row,window_size), strides=(x.strides[0],x.strides[0]))
    return output[:-1], output[-1]

def prepare_data_y(x, window_size):
    # # perform simple moving average
    # output = np.convolve(x, np.ones(window_size), 'valid') / window_size

    # use the next day as label
    output = x[window_size:]
    return output

def prepare_data(normalized_data_close_price, config, plot=False):
    data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size=config["data"]["window_size"])
    data_y = prepare_data_y(normalized_data_close_price, window_size=config["data"]["window_size"])

    # split dataset
    
    split_index = int(data_y.shape[0]*config["data"]["train_split_size"])
    data_x_train = data_x[:split_index]
    data_x_val = data_x[split_index:]
    data_y_train = data_y[:split_index]
    data_y_val = data_y[split_index:]


    return split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        x = np.expand_dims(x, 2) # in our case, we have only 1 feature, so we need to convert `x` into [batch, sequence, features] for LSTM
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu

class LSTMModel(nn.Module):

        def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
            super().__init__()
            self.hidden_layer_size = hidden_layer_size
            self.linear_1 = nn.Linear(input_size, hidden_layer_size)
            self.relu = nn.ReLU()
            self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.linear_2 = nn.Linear(num_layers*hidden_layer_size, output_size)
            self.init_weights()
            self.historycorrectorlstm = []
            self.historycorrection = []
      
        def init_weights(self):
            for name, param in self.lstm.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight_ih' in name:
                    nn.init.kaiming_normal_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
        def forward(self, x):
            batchsize = x.shape[0]
            # layer 1
            x = self.linear_1(x)
            x = self.relu(x)

            # LSTM layer
            lstm_out, (h_n, c_n) = self.lstm(x)     

            # reshape output from hidden cell into [batch, features] for `linear_2`
            x = h_n.permute(1, 0, 2).reshape(batchsize, -1) 

            # layer 2
            x = self.dropout(x)
            predictions = self.linear_2(x)
            return predictions[:,-1]

def run_epoch(dataloader, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        x = x.to(config["training"]["device"])
        y = y.to(config["training"]["device"])

        out = model(x)
        
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / batchsize)

    lr = scheduler.get_last_lr()[0]

    return epoch_loss, lr

def skip(id, paramstring, metric):
    if id not in metric.keys():
        return False
    
    if paramstring not in metric[id].keys():
        return False
    
    return set(metric[id][paramstring].keys()) == set(['y_pred', 'y_true', 'y_train'])

# select from Hourly, Daily, Weekly ... etc
X_train_df, y_train_df, X_test_df, y_test_df = prepare_m4_data(
dataset_name="Monthly", 
directory= "./data", 
num_obs=200)


# unique_id is the selected timeseries from dataset Monthly
with open(f"pred_plots_exp/metric_lstm.json", 'r') as infile:
    metric = json.load(infile)


for size in range(1,6,2):
    for num in [0.01, 0.05, 0.1, 0.15, 0.20, 0.25]:
        for mag_p in [0.1, 0.2, 0.5]:
            for loc_p in [0.1, 0.2, 0.5]:
                for unique_id in range(1,21):
                    str_id = str(unique_id) 
                    if not os.path.exists(f"pred_plots_exp/lstm{unique_id}"):
                        # if the demo_folder directory is not present 
                        # then create it.
                        os.makedirs(f"pred_plots_exp/lstm{unique_id}")

                    if skip(str_id, f"{size}-{mag_p}-{num}-{loc_p}", metric):
                        print("Skipped")
                        continue

                    if str_id not in metric.keys():
                        metric[str_id] = {}
                    
                    metric[str_id][f"{size}-{mag_p}-{num}-{loc_p}"] = {}

                    print(unique_id)
                    print("LSTM")
                    data_close_price = y_train_df.loc[y_train_df.unique_id=="M"+str(unique_id), "y"][-500:].values

                    split_index = int(len(data_close_price)*config["data"]["train_split_size"])
                    train_series = data_close_price[:split_index]

                    mag_mean = size*np.std(train_series)
                    mag_std = mag_p*mag_mean
                    loc_mean = 1/num
                    loc_std = loc_p*loc_mean
                    
                    data_close_price[:split_index] = introduce_errors(train_series, normal_dist(mag_mean, mag_std), normal_dist(loc_mean, loc_std))
                    # normalize
                    scaler = Normalizer()
                    normalized_data_close_price = scaler.fit_transform(data_close_price)
                    split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen = prepare_data(normalized_data_close_price, config, plot=config["plots"]["show_plots"])

                    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
                    dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

                    print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
                    print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

                    model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"], num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
                    model = model.to(config["training"]["device"])

                    # create `DataLoader`
                    train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
                    val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

                    # define optimizer, scheduler and loss function
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
                    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

                    # begin training
                    for epoch in range(config["training"]["num_epoch"]):
                        loss_train, lr_train = run_epoch(train_dataloader, is_training=True)
                        loss_val, lr_val = run_epoch(val_dataloader)
                        scheduler.step()

                        print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
                                    .format(epoch+1, config["training"]["num_epoch"], loss_train, loss_val, lr_train))

                    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
                    torch.manual_seed(0)
                    train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
                    val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

                    model.eval()

                    # predict on the training data, to see how well the model managed to learn and memorize

                    predicted_train = np.array([])

                    for idx, (x, y) in enumerate(train_dataloader):
                        x = x.to(config["training"]["device"])
                        out = model(x)
                        out = out.cpu().detach().numpy()
                        predicted_train = np.concatenate((predicted_train, out))

                    # predict on the validation data, to see how the model does

                    predicted_val = np.array([])

                    for idx, (x, y) in enumerate(val_dataloader):
                        x = x.to(config["training"]["device"])
                        out = model(x)
                        out = out.cpu().detach().numpy()
                        predicted_val = np.concatenate((predicted_val, out))
                    
                    print(len(predicted_val))

                    if True:
                        # prepare data for plotting, show predicted prices
                        data_date = np.array(y_train_df.loc[y_train_df.unique_id=="M"+str(unique_id), 'ds'][-500:].values)

                        num_data_points = len(data_date)

                        to_plot_data_y_train_pred = np.zeros(num_data_points)
                        to_plot_data_y_val_pred = np.zeros(num_data_points)

                        to_plot_data_y_train_pred[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = scaler.inverse_transform(predicted_train)
                        to_plot_data_y_val_pred[split_index+config["data"]["window_size"]:] = scaler.inverse_transform(predicted_val)

                        to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
                        to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)

                        # plots

                        fig = figure(figsize=(25, 5), dpi=80)
                        fig.patch.set_facecolor((1.0, 1.0, 1.0))
                        plt.plot(data_date, data_close_price, label="Errored Data", color=config["plots"]["color_error"])
                        plt.plot(data_date, y_train_df.loc[y_train_df.unique_id=="M"+str(unique_id), "y"][-500:].values, label="Original Data", color=config["plots"]["color_actual"])
                        plt.plot(data_date, to_plot_data_y_train_pred, label="Predicted prices (train)", color=config["plots"]["color_pred_train"])
                        plt.plot(data_date, to_plot_data_y_val_pred, label="LSTM Prediction", color=config["plots"]["color_pred_val"])
                        plt.legend()
                        plt.savefig(f"pred_plots_exp/lstm{unique_id}/normal-{size}-{mag_p}-{num}-{loc_p}.pdf")

                        # prepare data for plotting, zoom in validation

                        to_plot_data_y_val_subset = scaler.inverse_transform(data_y_val)
                        to_plot_predicted_val = scaler.inverse_transform(predicted_val)
                        to_plot_data_date = data_date[split_index+config["data"]["window_size"]:]

                        # plots
                        
                        fig = figure(figsize=(25, 5), dpi=80)
                        fig.patch.set_facecolor((1.0, 1.0, 1.0))
                        plt.plot(to_plot_data_date, to_plot_data_y_val_subset, label="Actual prices", color=config["plots"]["color_actual"])
                        plt.plot(to_plot_data_date, to_plot_predicted_val, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
                        plt.title("Zoom in to examine predicted price on validation data portion")
                        plt.grid(b=None, which='major', axis='y', linestyle='--')

                        plt.legend()
                        plt.savefig(f"pred_plots_exp/lstm{unique_id}/scale-{size}-{mag_p}-{num}-{loc_p}.pdf")
                        metric[str_id][f"{size}-{mag_p}-{num}-{loc_p}"]['y_true'] = to_plot_data_y_val_subset[:len(to_plot_data_y_val_subset)-1].tolist()
                        metric[str_id][f"{size}-{mag_p}-{num}-{loc_p}"]['y_pred'] = to_plot_predicted_val[1:].tolist()
                        metric[str_id][f"{size}-{mag_p}-{num}-{loc_p}"]['y_train'] = data_close_price[:split_index].tolist()

                        with open(f"pred_plots_exp/metric_lstm.json", 'w') as outfile:
                            json.dump(metric, outfile, indent=2)
                        
                        print(metric.keys())
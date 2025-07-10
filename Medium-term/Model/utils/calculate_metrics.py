import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
np.seterr(over='ignore')
def plot_metric(dfhistory, metric, column=None):
    train_metrics = dfhistory["train_" + metric]
    val_metrics = dfhistory['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    if column is not None: 
        column = " " + column
    else:
        column = ""
    plt.title('Training and validation '+ metric + column)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.show()

def cal_tda(y_real, y_pred):

    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    
    tda_count = 0
    for i in range(len(y_real) - 1):
        actual_change = y_real[i + 1] - y_real[i]
        predicted_change = y_pred[i + 1] - y_real[i]

        if actual_change * predicted_change >= 0:
            tda_count += 1
    
    # calculate the tda value
    tda = tda_count / (len(y_real) - 1)
    
    return tda

def cal_smape(y_real, y_pred):

    N = len(y_real)
    smape = 1/N * np.sum(np.abs(y_real - y_pred) / ((np.abs(y_real) + np.abs(y_pred))/2))
    return smape

# 定义MAPE函数
def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 1e-10  
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# 计算点预测的评估指标
def cal_eval(y_real, y_pred):
    
    y_real, y_pred = np.array(y_real).ravel(), np.array(y_pred).ravel()

    r2 = r2_score(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred, squared=True)
    rmse = mean_squared_error(y_real, y_pred, squared=False)  # RMSE and MAE are various on different scales
    mae = mean_absolute_error(y_real, y_pred)
    medae = median_absolute_error(y_real, y_pred)
    mape = mean_absolute_percentage_error(y_real, y_pred) * 100  # Note that dataset cannot have any 0 value.
    epsilon = np.finfo(np.float64).eps  # to avoid division by zero
    mdape = np.median(np.abs(y_real - y_pred) / np.maximum(np.abs(y_real), epsilon)) * 100  # Note that dataset cannot have any 0 value.
    tda = cal_tda(y_real, y_pred) * 100
    smape = cal_smape(y_real, y_pred) * 100
    
    df_eval = pd.DataFrame({'R2': r2, 
                            'MSE':mse, 'RMSE': rmse, 
                            'MAE': mae, 'MedAE':medae, 
                            'MAPE': mape, 'MdAPE':mdape,
                            'SMAPE': smape,
                            'TDA': tda}, index=['Eval'])

    return df_eval



# 计算区间预测的评估指标
def cal_interval_eval(y_real, pre_low, pre_up, mu=95, eta=50):

    n_samples = len(y_real)
    
    n_in_intervals = 0
    for i in range(n_samples):
        if pre_low[i]<= y_real[i] <= pre_up[i]:
            n_in_intervals = n_in_intervals+1
       
    
    PIAW = np.mean(pre_up - pre_low)
    PINAW = PIAW/(np.max(y_real)-np.min(y_real))
    
    PICP = n_in_intervals / n_samples * 100

    pre_mid = (pre_low + pre_up)/2
    MPICD = np.mean(np.abs(pre_mid - y_real))
    
    
    A_t_values = np.where(y_real < pre_low, 
                          (pre_low - y_real) / (pre_up - pre_low),
                          np.where(y_real > pre_up, (y_real - pre_up) / (pre_up - pre_low), 0)
                          )
    AWD = np.sum(A_t_values)

    CWC = cal_CWC(PINAW, PICP, mu, eta)
    
    df_interval_eval = pd.DataFrame({
        'PIAW': PIAW, 
        'PINAW': PINAW, 
        'PICP': PICP, 
        'MPICD': MPICD, 
        'AWD': AWD,
        'CWC': CWC
        }, 
        index=[mu]
        )

    return df_interval_eval.T


def cal_CWC(PINAW, PICP, mu, eta):
    gamma = 1 if PICP < mu else 0
    CWC = PINAW * (1 + gamma * np.exp(-eta * (PICP - mu)))
    return CWC


def cal_multi_quantile_eval(levels, df_pred_true, mu=95, eta=50):

    result_dict = {}
    for quantile in levels:
        y_real, pre_low, pre_up = df_pred_true['Real'], df_pred_true[f'Predict-lo-{quantile}'], df_pred_true[f'Predict-hi-{quantile}']
        df_interval_eval = cal_interval_eval(y_real, pre_low, pre_up, mu=mu, eta=eta)
        result_dict[f'Quantile-{quantile}'] = df_interval_eval

    return pd.concat(result_dict.values(), axis=1, keys=result_dict.keys())
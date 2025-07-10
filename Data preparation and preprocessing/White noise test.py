import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.stats.diagnostic import acorr_ljungbox
import os

def analyze_time_series_minimal(df, start_col_index=2, n_rows=182600, lags_list=[90], save_image=True):
    for col_index in range(start_col_index, len(df.columns)):
        col_name = df.columns[col_index]
        
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        df_clean = df.dropna(subset=[col_name])
        time_series = df_clean[col_name].iloc[:n_rows]
        
        plt.figure(figsize=(12, 6))
        autocorrelation_plot(time_series)
        
        plt.xlabel('Lag', fontsize=24, family='Times New Roman') 
        plt.ylabel('Autocorrelation', fontsize=24, family='Times New Roman')  
        if save_image:
            save_path = r'C:\Users\16432\Desktop\ACF'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'ACF_{col_name}.png'), dpi=600)
        
        lb_bp_results = acorr_ljungbox(time_series, lags=lags_list, boxpierce=True, return_df=True)
        
        lb_bp_results['lb_pvalue'] = lb_bp_results['lb_pvalue'].apply(lambda x: f'{x:.10f}')
        lb_bp_results['bp_pvalue'] = lb_bp_results['bp_pvalue'].apply(lambda x: f'{x:.10f}')
        

df = pd.read_parquet("data_modified.parquet")
analyze_time_series_minimal(df, start_col_index=2, n_rows=182600, lags_list=[18260])
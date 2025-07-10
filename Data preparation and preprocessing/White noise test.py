import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.stats.diagnostic import acorr_ljungbox
import os

def analyze_time_series_minimal(df, start_col_index=2, n_rows=182600, lags_list=[90], save_image=True):
    # 遍历从第4列到最后一列的所有列
    for col_index in range(start_col_index, len(df.columns)):
        col_name = df.columns[col_index]
        print(f"\n📌 正在分析列: {col_name}")
        
        # 转换为数值型 + 去除缺失
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        df_clean = df.dropna(subset=[col_name])
        
        # 取前 n_rows 行
        time_series = df_clean[col_name].iloc[:n_rows]
        
        print(f"数据长度: {len(time_series)}")
        print(f"缺失值比例: {1 - len(time_series)/n_rows:.2%}")

        # 绘制自相关图（ACF）
        plt.figure(figsize=(12, 6))
        autocorrelation_plot(time_series)
        
        # 设置标题和轴标签的字体大小
        plt.xlabel('Lag', fontsize=24, family='Times New Roman')  # x轴字体大小
        plt.ylabel('Autocorrelation', fontsize=24, family='Times New Roman')  # y轴字体大小

        # 保存 ACF 图，设置为 600 DPI
        if save_image:
            # 确保路径存在
            save_path = r'C:\Users\16432\Desktop\博士研究方向\博士小论文\文章1\绘图\白噪声图'
            if not os.path.exists(save_path):
                os.makedirs(save_path)  # 如果目录不存在，则创建
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'ACF_{col_name}.png'), dpi=600)
        
        # plt.show()

        # 白噪声检验（Ljung-Box & Box-Pierce）
        print(f"\n🔍 白噪声检验（Ljung-Box & Box-Pierce）:")
        lb_bp_results = acorr_ljungbox(time_series, lags=lags_list, boxpierce=True, return_df=True)
        
        # 显示 p-values 小数点后 10 位
        lb_bp_results['lb_pvalue'] = lb_bp_results['lb_pvalue'].apply(lambda x: f'{x:.10f}')
        lb_bp_results['bp_pvalue'] = lb_bp_results['bp_pvalue'].apply(lambda x: f'{x:.10f}')
        
        print(lb_bp_results[['lb_stat', 'lb_pvalue', 'bp_stat', 'bp_pvalue']])

# 示例用法
df = pd.read_parquet("data_modified.parquet")
analyze_time_series_minimal(df, start_col_index=2, n_rows=182600, lags_list=[18260])
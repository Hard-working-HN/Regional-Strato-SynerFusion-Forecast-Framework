import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error

# ======= 用户自定义 =======
input_dir = r"C:\Users\16432\Desktop\博士研究方向\博士小论文\文章1\附表\Table_S7_S8\Result_statistics\Long\parquet5"  # 合并后的 CSV 所在文件夹
output_dir = r"C:\Users\16432\Desktop\博士研究方向\博士小论文\文章1\附表\Table_S7_S8\Evaluation\Long\parquet5"  # 计算结果输出路径
os.makedirs(output_dir, exist_ok=True)

# ======= 异常值索引 =======
exclude_ids = [55, 91, 103, 147, 174, 188, 199, 262, 332, 363, 433, 439, 492]

# ======= 指标计算函数 =======
def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_safe = np.where(y_true == 0, 1e-6, y_true)
    
    metrics = {}
    metrics['R2'] = r2_score(y_true, y_pred)
    metrics['MSE'] = mean_squared_error(y_true, y_pred)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['MedAE'] = median_absolute_error(y_true, y_pred)
    metrics['MAPE'] = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    metrics['MdAPE'] = np.median(np.abs((y_true - y_pred) / y_true_safe)) * 100
    metrics['TDA'] = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100
    metrics['SMAPE'] = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6))
    metrics['WMAPE'] = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true_safe)) * 100
    metrics['MRE'] = np.mean((y_pred - y_true) / y_true_safe) * 100
    metrics['SI'] = metrics['RMSE'] / (np.mean(y_true) + 1e-6)
    
    return metrics

# ======= 单个文件处理函数 =======
def process_file(input_csv, output_dir):
    file_name = os.path.splitext(os.path.basename(input_csv))[0]
    df = pd.read_csv(input_csv)

    # 判断是否需要剔除异常
    if file_name.startswith('CO') or file_name.startswith('NO2'):
        df = df[~df.iloc[:, 1].isin(exclude_ids)]
    
    y_true = df.iloc[:, 2].values
    y_pred = df.iloc[:, 3].values

    # === 平均 ===
    avg_metrics = calculate_metrics(y_true, y_pred)
    avg_df = pd.DataFrame([avg_metrics])

    # === 单一 ===
    single_results = []
    for unique_val in sorted(df.iloc[:, 1].unique()):
        df_sub = df[df.iloc[:, 1] == unique_val]
        if len(df_sub) < 2:
            continue
        metrics = calculate_metrics(df_sub.iloc[:, 2].values, df_sub.iloc[:, 3].values)
        metrics['ID'] = unique_val
        single_results.append(metrics)

    single_df = pd.DataFrame(single_results).set_index('ID')

    # === 保存 ===
    save_folder = os.path.join(output_dir, file_name)
    os.makedirs(save_folder, exist_ok=True)

    avg_output = os.path.join(save_folder, f"Average_{file_name}.csv")
    single_output = os.path.join(save_folder, f"Single_{file_name}.csv")

    avg_df.to_csv(avg_output, index=False, encoding='utf-8-sig')
    single_df.to_csv(single_output, encoding='utf-8-sig')

    print(f"✅ 已处理: {file_name}")

# ======= 批量执行 =======
def batch_process(input_dir, output_dir):
    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
    if not csv_files:
        print("❌ 没有找到CSV文件")
        return

    for file in csv_files:
        input_csv = os.path.join(input_dir, file)
        process_file(input_csv, output_dir)

    print("\n🎉 全部文件处理完成。")

# ======= 运行 =======
batch_process(input_dir, output_dir)

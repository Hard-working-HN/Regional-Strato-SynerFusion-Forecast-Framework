import pandas as pd
import numpy as np
import random
import time
import os
import glob
import gc
import torch
import torch.nn as nn
from utils.timefeatures import time_features
from utils.calculate_metrics import *
from utils.data_process import *
from utils.tools import *
import warnings
warnings.filterwarnings("ignore")

start_total = time.perf_counter()

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机种子
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 定义参数：这里采用 365 天的输入预测未来 90 天（直接多步预测）
window = 90          # 输入 365 天的数据
length_size = 30      # 输出预测 90 天
batch_size = 1024
loss_function = "MSE"
if loss_function == 'MSE':
    loss_func = nn.MSELoss(reduction='mean')

learning_rate = 0.001
epochs = 50          # 总训练 epoch 数

samples_per_county = 3652  # 每个县的数据样本数

# 定义 3 层 GRU 模型
class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim, window, num_layers=3):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size * window, output_dim)

    def forward(self, x, x_mark):
        combined = torch.cat((x, x_mark), dim=2)
        out, _ = self.gru(combined)
        out = out.contiguous().view(out.size(0), -1)
        out = self.fc(out)
        return out

# 搜索存放 parquet 数据的文件夹
parquet_files = glob.glob(r"E:\VScode\Article\Com_Model\data\othermodel\*.parquet")
print(f"共找到 {len(parquet_files)} 个 parquet 文件用于训练。")

for parquet_file in parquet_files:
    file_start_time = time.perf_counter()
    file_name = os.path.basename(parquet_file).split('.')[0]
    print(f"\n------------------------\n开始处理文件：{parquet_file}")

    df = pd.read_parquet(parquet_file)
    if df.shape[0] % samples_per_county != 0:
        print(f"警告：数据行数不是 {samples_per_county} 的整数倍，总行数：{df.shape[0]}")

    # 特征和标签
    data_target = df.iloc[:, -1]
    data_features = df.iloc[:, 1:-1]
    date_col = df.iloc[:, 0]

    df_stamp = pd.DataFrame({'date': pd.to_datetime(date_col)})
    data_stamp = time_features(df_stamp, timeenc=1, freq='D')

    data_all = data_target.to_frame() if data_features.shape[1]==0 else pd.concat([data_features, data_target], axis=1)
    data_all_np = data_all.values
    data_all_np = data_all_np.reshape(-1,1) if data_all_np.ndim==1 else data_all_np

    # 标准化
    scaler = MinMaxScaler()
    data_inverse = scaler.fit_transform(data_all_np)

    total_samples = len(data_inverse)
    total_counties = total_samples // samples_per_county

    # 划分：30 训练，6 验证，其余测试
    train_counties, val_counties = 30, 6
    train_size = train_counties * samples_per_county
    val_size = (train_counties + val_counties) * samples_per_county

    data_train = data_inverse[:train_size]
    data_val = data_inverse[train_size:val_size]
    data_test = data_inverse[val_size:]
    stamp_train = data_stamp[:train_size]
    stamp_val = data_stamp[train_size:val_size]
    stamp_test = data_stamp[val_size:]

    # 构造数据加载器
    train_loader, x_train, y_train, x_train_mark, y_train_mark, _, _ = tslib_data_loader(
        window, length_size, batch_size, data_train, stamp_train,
        samples_per_county=samples_per_county, shuffle=True, device=device, start_offset=0)
    val_loader, x_val, y_val, x_val_mark, y_val_mark, _, _ = tslib_data_loader(
        window, length_size, batch_size, data_val, stamp_val,
        samples_per_county=samples_per_county, shuffle=False, device=device, start_offset=train_size)

    input_size = x_train.shape[2] + x_train_mark.shape[2]
    hidden_size = 128

    # 实例化 GRU 模型
    gru_model = SimpleGRU(input_size, hidden_size, output_dim=length_size, window=window).to(device)
    optimizer = torch.optim.Adam(gru_model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    print("—— GRU 训练开始 ——")
    for epoch in range(epochs):
        gru_model.train()
        train_loss = 0
        for feature_, _, feature_mark_, _, label_, _ in train_loader:
            optimizer.zero_grad()
            pred = gru_model(feature_, feature_mark_)
            loss = loss_func(pred, label_)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        gru_model.eval()
        val_loss = 0
        with torch.no_grad():
            for feature_, _, feature_mark_, _, label_, _ in val_loader:
                pred = gru_model(feature_, feature_mark_)
                val_loss += loss_func(pred, label_).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dir = f"./ShortGRUWeights/{file_name}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(gru_model.state_dict(), os.path.join(save_dir, f'{file_name}_best_GRU.pth'))
        print(f"[{file_name}] Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    print("—— GRU 训练结束 ——")

    # 测试集预测
    gru_model.load_state_dict(torch.load(os.path.join(save_dir, f'{file_name}_best_GRU.pth')))
    gru_model.eval()
    records = []
    step = length_size
    for i in range(total_counties - train_counties - val_counties):
        offset = val_size + i * samples_per_county
        county_data = data_test[i*samples_per_county:(i+1)*samples_per_county]
        county_stamp = stamp_test[i*samples_per_county:(i+1)*samples_per_county]
        for start in range(0, samples_per_county - window + 1, step):
            rem = samples_per_county - (start + window)
            if rem <= 0: break
            L = step if rem >= step else rem
            x_seq = county_data[start:start+window]
            x_mark_seq = county_stamp[start:start+window]
            label_seq = county_data[start+window:start+window+L, -1]
            x_t = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0).to(device)
            mark_t = torch.tensor(x_mark_seq, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad(): pred_full = gru_model(x_t, mark_t).cpu().numpy().squeeze(0)
            pred_seq = pred_full[:L]
            for j in range(L):
                idx = offset + start + window + j
                date_val = df_stamp.iloc[idx]['date']
                county_idx = idx // samples_per_county + 1
                records.append({
                    'date': date_val,
                    'county_index': county_idx,
                    'true_norm': label_seq[j],
                    'pred_norm': pred_seq[j]
                })

    df_pred = pd.DataFrame(records)
    n = df_pred.shape[0]
    temp_pred = np.zeros((n, data_all_np.shape[1]))
    temp_label = np.zeros((n, data_all_np.shape[1]))
    temp_pred[:, -1] = df_pred['pred_norm'].values
    temp_label[:, -1] = df_pred['true_norm'].values
    df_pred['pred'] = scaler.inverse_transform(temp_pred)[:, -1]
    df_pred['true'] = scaler.inverse_transform(temp_label)[:, -1]
    df_pred.drop(['pred_norm','true_norm'], axis=1, inplace=True)

    # 保存结果
    save_pred_dir = f"./ShortPrediction/{file_name}"
    os.makedirs(save_pred_dir, exist_ok=True)
    df_pred.to_csv(os.path.join(save_pred_dir, f'{file_name}_predictions.csv'), index=False)
    print(f"预测结果保存至 {save_pred_dir}")

    # 计算评估指标
    df_eval = cal_eval(df_pred['true'].values, df_pred['pred'].values)
    pd.DataFrame(df_eval, index=[0]).to_csv(os.path.join(save_pred_dir, f'{file_name}_metrics.csv'), index=False)
    print("评估指标计算并保存完毕。")

    # 清理
    del df, data_train, data_val, data_test, stamp_train, stamp_val, stamp_test
    del x_train, x_val
    del gru_model, train_loader, val_loader, df_pred
    torch.cuda.empty_cache()
    gc.collect()
    print(f"文件 {file_name} 完成，总耗时 {time.perf_counter() - file_start_time:.2f} 秒。")

print(f"全部完成，总耗时 {time.perf_counter() - start_total:.2f} 秒。")

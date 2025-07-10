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
window = 7          # 输入 365 天的数据
length_size = 1     # 输出预测 30 天
batch_size = 1024
loss_function = "MSE"
if loss_function == 'MSE':
    loss_func = nn.MSELoss(reduction='mean')

learning_rate = 0.001
epochs = 999          # 总训练 epoch 数

samples_per_county = 3652  # 每个县的数据样本数

# 搜索存放 parquet 数据的文件夹
parquet_files = glob.glob(r"E:\VScode\Article\Com_Model\data\othermodel\*.parquet")
print("共找到 {} 个 parquet 文件用于训练。".format(len(parquet_files)))

for parquet_file in parquet_files:
    file_start_time = time.perf_counter()

    file_name = os.path.basename(parquet_file).split('.')[0]
    print(f"\n------------------------\n开始处理文件：{parquet_file}")
    print(f"文件名解析为：{file_name}")

    df = pd.read_parquet(parquet_file)
    print(f"文件 {file_name} 数据总行数为：{len(df)}。")
    if df.shape[0] % samples_per_county != 0:
        print(f"警告：数据行数不是 {samples_per_county} 的整数倍，可能数据有问题。总行数：{df.shape[0]}")

    # 数据分为：时间戳、原始特征和目标变量
    data_target = df.iloc[:, -1]
    data_features = df.iloc[:, 1:-1]
    date_col = df.iloc[:, 0]

    df_stamp = pd.DataFrame({'date': pd.to_datetime(date_col)})
    data_stamp = time_features(df_stamp, timeenc=1, freq='D')

    # 若 data_features 为空，则仅保留 target，否则拼接
    if data_features.shape[1] == 0:
        data_all = data_target.to_frame()
    else:
        data_all = pd.concat([data_features, data_target], axis=1)
    data_all_np = np.array(data_all)
    # 如果转换后的数组为1维，则强制转换为二维
    if data_all_np.ndim == 1:
        data_all_np = data_all_np.reshape(-1, 1)

    # 标准化处理
    scaler = MinMaxScaler()
    data_inverse = scaler.fit_transform(data_all_np)

    total_samples = len(data_inverse)
    total_counties = total_samples // samples_per_county

    # 划分训练集、验证集、测试集（示例：训练 30 个县，验证 6 个县，其余作为测试）
    train_counties = 30
    val_counties = 6
    test_counties = total_counties - train_counties - val_counties

    train_size = train_counties * samples_per_county
    val_size = (train_counties + val_counties) * samples_per_county
    test_size = total_samples - val_size

    print(f"总共有 {total_counties} 个县的数据，每县 {samples_per_county} 条记录。")
    print(f"训练集使用前 {train_counties} 个县，总行数为：{train_size}")
    print(f"验证集使用接下来的 {val_counties} 个县，总行数为：{val_size - train_size}")
    print(f"测试集使用最后 {test_counties} 个县，总行数为：{test_size}")

    data_train = data_inverse[:train_size, :]
    data_val = data_inverse[train_size:val_size, :]
    data_test = data_inverse[val_size:, :]
    data_stamp_train = data_stamp[:train_size, :]
    data_stamp_val = data_stamp[train_size:val_size, :]
    data_stamp_test = data_stamp[val_size:, :]

    # 通过现有的 ts_lib 数据加载器来构造训练和验证样本
    train_loader, x_train, y_train, x_train_mark, y_train_mark, label_train, label_indices_train = tslib_data_loader(
        window, length_size, batch_size, data_train, data_stamp_train, 
        samples_per_county=samples_per_county, shuffle=True, device=device, start_offset=0)

    val_loader, x_val, y_val, x_val_mark, y_val_mark, label_val, label_indices_val = tslib_data_loader(
        window, length_size, batch_size, data_val, data_stamp_val, 
        samples_per_county=samples_per_county, shuffle=False, device=device, start_offset=train_size)

    print("训练集 x 形状:", x_train.shape, "验证集 x 形状:", x_val.shape)
    print(f"训练集批次数：{len(train_loader)}, 验证集批次数：{len(val_loader)}")

    # 定义 3 层 CNN 模型（输出尺寸为 length_size，即预测未来 30 天）
    class SimpleCNN(nn.Module):
        def __init__(self, input_channels, hidden_channels, output_dim, window):
            super(SimpleCNN, self).__init__()
            # 第一层卷积
            self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
            # 第二层卷积
            self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
            # 第三层卷积
            self.conv3 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
            self.relu = nn.ReLU()
            # 全连接层将卷积输出映射到最终预测值，卷积输出尺寸为 (hidden_channels, window)
            self.fc = nn.Linear(hidden_channels * window, output_dim)
        def forward(self, x, x_mark):
            # x: (batch, window, feature_dim)
            # x_mark: (batch, window, time_feature_dim)
            combined = torch.cat((x, x_mark), dim=2)  # 拼接后形状: (batch, window, feature_dim+time_feature_dim)
            # 调整为 Conv1d 要求的输入形状 (batch, channels, window)
            combined = combined.transpose(1, 2)
            out = self.relu(self.conv1(combined))
            out = self.relu(self.conv2(out))
            out = self.relu(self.conv3(out))
            # 展平后映射到输出
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out

    # 根据训练数据确定输入通道数
    sample_feature = x_train[:1]         # shape: (1, window, feature_dim)
    sample_feature_mark = x_train_mark[:1] # shape: (1, window, time_feature_dim)
    input_channels = sample_feature.shape[2] + sample_feature_mark.shape[2]
    hidden_channels = 128
    cnn_model = SimpleCNN(input_channels, hidden_channels, output_dim=length_size, window=window).to(device)
    print("模型初始化完成：")
    print(cnn_model)

    # 创建权重保存文件夹（保存路径中用 ShortCNNWeights 表示中期模型）
    weights_dir = './ShortCNNWeights'
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    file_weights_dir = os.path.join(weights_dir, file_name)
    if not os.path.exists(file_weights_dir):
        os.makedirs(file_weights_dir)

    # 训练 3 层 CNN 模型
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    print("——————————————————————3层CNN训练开始——————————————————————")
    for epoch in range(epochs):
        cnn_model.train()
        train_loss_sum = 0
        for step, (feature_, y_train_batch, feature_mark_, y_train_mark_batch, label_, _) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = cnn_model(feature_, feature_mark_)
            loss = loss_func(pred, label_)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        cnn_model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for step, (feature_, y_val_batch, feature_mark_, y_val_mark_batch, label_, _) in enumerate(val_loader):
                pred = cnn_model(feature_, feature_mark_)
                loss = loss_func(pred, label_)
                val_loss_sum += loss.item()

        if val_loss_sum < best_val_loss:
            best_val_loss = val_loss_sum
            torch.save(cnn_model.state_dict(), os.path.join(file_weights_dir, f'{file_name}_best_CNN_model.pth'))
        print(f"[{file_name}] Epoch {epoch+1}/{epochs} - CNN 训练损失: {train_loss_sum:.8f}, 验证损失: {val_loss_sum:.8f}")
    print("——————————————————————3层CNN训练结束——————————————————————")

    # 测试集预测：对测试集每个县单独滑动构造预测序列
    print("——————————————————————测试集预测开始——————————————————————")
    cnn_model.load_state_dict(torch.load(os.path.join(file_weights_dir, f'{file_name}_best_CNN_model.pth')))
    cnn_model.eval()
    pred_records = []  # 存储每个预测记录，包含日期、县号、真实值和预测值

    # 对测试集每个县进行预测
    step = length_size  # 采用预测步长与预测窗口一致，这样每次预测的区间不重叠
    for i in range(test_counties):
        # 全局偏移：测试集在原始数据中的起始位置（相对于整个数据）
        county_offset = val_size + i * samples_per_county  
        # 提取本县的全部数据和时间特征（均为标准化后的结果）
        start_idx = i * samples_per_county
        end_idx = (i+1) * samples_per_county
        county_data = data_test[start_idx:end_idx, :]            # shape: (samples_per_county, d)
        county_stamp = data_stamp_test[start_idx:end_idx, :]       # shape: (samples_per_county, time_feature_dim)
        n = samples_per_county
        # 按步长滑动生成样本
        for start in range(0, n - window + 1, step):
            remaining = n - (start + window)
            if remaining <= 0:
                break
            # L 为本次预测标签长度：如果剩余天数超过预测窗口，则预测整个窗口，否则只预测剩余部分
            L = step if remaining >= step else remaining
            # 构造输入序列（x）和对应的时间特征（x_mark）
            x_seq = county_data[start : start+window, :]            # shape: (window, d)
            x_mark_seq = county_stamp[start : start+window, :]        # shape: (window, time_feature_dim)
            # 真正标签：取当前预测区间内目标列（最后一列）的值
            label_seq = county_data[start+window : start+window+L, -1]  # shape: (L,)
            
            # 转换为 tensor，并增加 batch 维度（1, window, ...）
            x_tensor = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0).to(device)
            x_mark_tensor = torch.tensor(x_mark_seq, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_full = cnn_model(x_tensor, x_mark_tensor)  # 输出 shape: (1, length_size)
            pred_full = pred_full.cpu().numpy().squeeze(0)      # shape: (length_size,)
            # 对于当前样本，只取前 L 个预测结果
            pred_seq = pred_full[:L]
            
            # 存储每一天的预测结果及对应信息
            for j in range(L):
                global_idx = county_offset + start + window + j  # 全局索引对应原始数据中的行号
                date_val = df_stamp.iloc[global_idx]['date']
                county_index = (global_idx // samples_per_county) + 1
                pred_records.append({
                    'date': date_val,
                    'county_index': county_index,
                    'true_value_norm': label_seq[j],
                    'pred_value_norm': pred_seq[j]
                })
    # 将预测结果列表转换为 DataFrame
    df_result = pd.DataFrame(pred_records)

    # 反标准化：由于 scaler 对整个数据（多列）进行标准化，这里构造辅助数组，仅将最后一列赋值为预测值或真实值
    n_rec = df_result.shape[0]
    temp_pred = np.zeros((n_rec, data_all_np.shape[1]))
    temp_label = np.zeros((n_rec, data_all_np.shape[1]))
    temp_pred[:, -1] = df_result['pred_value_norm'].values
    temp_label[:, -1] = df_result['true_value_norm'].values
    pred_inverse = scaler.inverse_transform(temp_pred)[:, -1]
    label_inverse = scaler.inverse_transform(temp_label)[:, -1]
    # 更新 DataFrame 中的预测和真实值
    df_result['pred_value'] = pred_inverse
    df_result['true_value'] = label_inverse
    # 删除归一化结果列
    df_result.drop(['pred_value_norm', 'true_value_norm'], axis=1, inplace=True)

    # 保存预测结果至 ShortPrediction 目录
    pre_dir = './ShortPrediction'
    if not os.path.exists(pre_dir):
        os.makedirs(pre_dir)
    file_pre_dir = os.path.join(pre_dir, file_name)
    if not os.path.exists(file_pre_dir):
        os.makedirs(file_pre_dir)
    df_result.to_csv(os.path.join(file_pre_dir, f'{file_name}_test_predictions.csv'), index=False)
    print(f"测试集预测结果已保存至 {file_pre_dir}/{file_name}_test_predictions.csv")
    print("——————————————————————测试集预测结束——————————————————————")

    # 计算并保存评估指标
    print("——————————————————————计算评估指标开始——————————————————————")
    df_eval = cal_eval(df_result['true_value'].values, df_result['pred_value'].values)
    print(f"评估指标:  {df_eval}")
    metrics_df = pd.DataFrame({
        'Metric': ['df_eval'],
        'Value': [df_eval]
    })
    metrics_df.to_csv(os.path.join(file_pre_dir, f'{file_name}_metrics.csv'), index=False)
    print(f"评估指标已保存至 {file_pre_dir}/{file_name}_metrics.csv")
    print("——————————————————————计算评估指标结束——————————————————————")

    # 释放内存
    del df, data_train, data_val, data_test, data_stamp_train, data_stamp_val, data_stamp_test
    del x_train, y_train, x_train_mark, y_train_mark, label_train, label_indices_train
    del x_val, y_val, x_val_mark, y_val_mark, label_val, label_indices_val
    del cnn_model, train_loader, val_loader
    del pred_records, df_result, metrics_df
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"文件 {file_name} 的训练与测试全部完成。总用时：{time.perf_counter() - file_start_time:.2f} 秒")

end_total = time.perf_counter()
print(f"全部文件的处理和训练完成，总用时: {end_total - start_total:.4f} 秒")

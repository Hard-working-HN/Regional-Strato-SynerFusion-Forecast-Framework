import pandas as pd
import numpy as np
import random
import time
import os
import glob
import shap
import gc 
import torch
import torch.nn as nn
from models import STAR
from models.LSTM import *
from models.MLP import *
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

# 定义参数
window = 7  # 输入7天
length_size = 1  # 预测1天
batch_size = 1024
dim_lstm = 128
num_blocks_lstm = 3
loss_function = "MSE"
if loss_function == 'MSE':
    loss_func = nn.MSELoss(reduction='mean')

learning_rate_base = 0.001
epochs_phase1 = 999  # 第一阶段 epochs
epochs_phase2 = 333   # 第二阶段 epochs

samples_per_county = 3652  # 每个县的数据样本数

class Config:
    def __init__(self):
        self.seq_len = window
        self.label_len = int(window / 2)
        self.pred_len = length_size
        self.freq = 'D'
        self.batch_size = batch_size
        self.dec_in = None
        self.enc_in = None
        self.c_out = 1
        self.d_model = 128
        self.n_heads = 8
        self.dropout = 0.0
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 128
        self.factor = 5
        self.activation = 'gelu'
        self.channel_independence = 0
        self.top_k = 5
        self.num_kernels = 6
        self.distil = 1
        self.use_norm = True
        self.d_core = 512
        self.embed = 'timeF'
        self.output_attention = 0
        self.task_name = 'short_term_forecast'

parquet_files = glob.glob("data/parquet/*.parquet")
print("共找到 {} 个parquet文件用于训练。".format(len(parquet_files)))

for parquet_file in parquet_files:
    file_start_time = time.perf_counter()

    file_name = os.path.basename(parquet_file).split('.')[0]
    print(f"\n------------------------\n开始处理文件：{parquet_file}")
    print(f"文件名解析为：{file_name}")

    df = pd.read_parquet(parquet_file)
    print(f"文件 {file_name} 数据总行数为：{len(df)}。")

    if df.shape[0] % 3652 != 0:
        print(f"警告：数据行数不是3652的整数倍，可能数据有问题。总行数：{df.shape[0]}")

    data_target = df.iloc[:, -1]
    data_features = df.iloc[:, 1:-1]
    date_col = df.iloc[:, 0]

    df_stamp = pd.DataFrame({'date': pd.to_datetime(date_col)})
    data_stamp = time_features(df_stamp, timeenc=1, freq='D')

    data_dim = df.shape[1] - 1
    data_all = pd.concat([data_features, data_target], axis=1)
    data_all_np = np.array(data_all)

    scaler = MinMaxScaler()
    data_inverse = scaler.fit_transform(data_all_np)

    total_samples = len(data_inverse)
    total_counties = total_samples // samples_per_county

    # 按7:1.5:1.5比例划分县
    train_counties = int(total_counties * 0.7)
    val_counties = int(total_counties * 0.85)
    test_counties = total_counties - val_counties

    train_size = train_counties * samples_per_county
    val_size = val_counties * samples_per_county
    test_size = total_samples - val_size

    print(f"总共有 {total_counties} 个县的数据，每县 {samples_per_county} 条记录。")
    print(f"训练集使用前 {train_counties} 个县，总行数为：{train_size}")
    print(f"验证集使用 {val_counties - train_counties} 个县，总行数为：{val_size - train_size}")
    print(f"测试集使用 {test_counties} 个县，总行数为：{test_size}")

    data_train = data_inverse[:train_size, :]
    data_val = data_inverse[train_size:val_size, :]
    data_test = data_inverse[val_size:, :]

    data_stamp_train = data_stamp[:train_size, :]
    data_stamp_val = data_stamp[train_size:val_size, :]
    data_stamp_test = data_stamp[val_size:, :]

    train_loader, x_train, y_train, x_train_mark, y_train_mark, label_train_lstm, label_indices_train = tslib_data_loader(
        window, length_size, batch_size, data_train, data_stamp_train, samples_per_county=samples_per_county, shuffle=True, device=device, start_offset=0)

    val_loader, x_val, y_val, x_val_mark, y_val_mark, label_val_lstm, label_indices_val = tslib_data_loader(
        window, length_size, batch_size, data_val, data_stamp_val, samples_per_county=samples_per_county, shuffle=False, device=device, start_offset=train_size)

    test_loader, x_test, y_test, x_test_mark, y_test_mark, label_test_lstm, label_indices_test = tslib_data_loader(
        window, length_size, batch_size, data_test, data_stamp_test, samples_per_county=samples_per_county, shuffle=False, device=device, start_offset=val_size)

    print("训练集x形状:", x_train.shape, "验证集x形状:", x_val.shape, "测试集x形状:", x_test.shape)
    print(f"训练集批次数：{len(train_loader)}, 验证集批次数：{len(val_loader)}, 测试集批次数：{len(test_loader)}")

    LSTMMain_model = LSTMMain(input_size=data_dim + data_stamp.shape[1],
                              output_len=length_size,
                              lstm_hidden=dim_lstm,
                              lstm_layers=num_blocks_lstm,
                              batch_size=batch_size,
                              device=device)
    LSTMMain_model.to(device)

    config = Config()
    config.dec_in = data_dim
    config.enc_in = data_dim
    net = STAR.Model(config).to(device)

    mlp_input_dim = 2
    mlp_hidden_dim = 64
    mlp_model = MLP(input_dim=mlp_input_dim, hidden_dim=mlp_hidden_dim).to(device)

    print("模型初始化完成：")
    print(LSTMMain_model)
    print(net)
    print(mlp_model)

    # 创建权重保存文件夹
    weights_dir = './short_weights'
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    file_weights_dir = os.path.join(weights_dir, file_name)
    if not os.path.exists(file_weights_dir):
        os.makedirs(file_weights_dir)

    # 第一阶段训练
    phase1_start_time = time.perf_counter()
    print("——————————————————————第一阶段：训练基础模型开始——————————————————————")
    optimizer_LSTM = torch.optim.Adam(LSTMMain_model.parameters(), lr=learning_rate_base)
    optimizer_STAR = torch.optim.Adam(net.parameters(), lr=learning_rate_base)

    best_val_loss_LSTM = float('inf')
    best_val_loss_STAR = float('inf')

    for epoch in range(epochs_phase1):
        LSTMMain_model.train()
        train_loss_sum_LSTM = 0
        for step, (feature_, _, feature_mark_, _, label_, _) in enumerate(train_loader):
            optimizer_LSTM.zero_grad()
            pred_lstm = LSTMMain_model(feature_, feature_mark_)
            loss_lstm = loss_func(pred_lstm, label_)
            loss_lstm.backward()
            optimizer_LSTM.step()
            train_loss_sum_LSTM += loss_lstm.item()

        LSTMMain_model.eval()
        val_loss_sum_LSTM = 0
        with torch.no_grad():
            for step, (feature_, _, feature_mark_, _, label_, _) in enumerate(val_loader):
                pred_lstm = LSTMMain_model(feature_, feature_mark_)
                loss_lstm = loss_func(pred_lstm, label_)
                val_loss_sum_LSTM += loss_lstm.item()

        if val_loss_sum_LSTM < best_val_loss_LSTM:
            best_val_loss_LSTM = val_loss_sum_LSTM
            torch.save(LSTMMain_model.state_dict(), os.path.join(file_weights_dir, f'{file_name}_best_LSTM_model.pth'))

        print(f"[{file_name}] Epoch {epoch+1}/{epochs_phase1} - LSTM 训练损失: {train_loss_sum_LSTM:.8f}, 验证损失: {val_loss_sum_LSTM:.8f}")

        net.train()
        train_loss_sum_STAR = 0
        for step, (feature_, y_train_batch, feature_mark_, y_train_mark_batch, label_, _) in enumerate(train_loader):
            optimizer_STAR.zero_grad()
            pred_star = net(feature_, feature_mark_, y_train_batch, y_train_mark_batch, None)
            pred_star = pred_star.squeeze(1)
            pred_star = pred_star[:, -1].unsqueeze(1)
            loss_star = loss_func(pred_star, label_)
            loss_star.backward()
            optimizer_STAR.step()
            train_loss_sum_STAR += loss_star.item()

        net.eval()
        val_loss_sum_STAR = 0
        with torch.no_grad():
            for step, (feature_, y_val_batch, feature_mark_, y_val_mark_batch, label_, _) in enumerate(val_loader):
                pred_star = net(feature_, feature_mark_, y_val_batch, y_val_mark_batch, None)
                pred_star = pred_star.squeeze(1)
                pred_star = pred_star[:, -1].unsqueeze(1)
                loss_star = loss_func(pred_star, label_)
                val_loss_sum_STAR += loss_star.item()

        if val_loss_sum_STAR < best_val_loss_STAR:
            best_val_loss_STAR = val_loss_sum_STAR
            torch.save(net.state_dict(), os.path.join(file_weights_dir, f'{file_name}_best_STAR_model.pth'))

        print(f"[{file_name}] Epoch {epoch+1}/{epochs_phase1} - STAR 训练损失: {train_loss_sum_STAR:.8f}, 验证损失: {val_loss_sum_STAR:.8f}")

    phase1_end_time = time.perf_counter()
    print("——————————————————————第一阶段：训练基础模型结束——————————————————————")
    print(f"[{file_name}] 第一阶段训练用时: {phase1_end_time - phase1_start_time:.2f} 秒")

    LSTMMain_model.load_state_dict(torch.load(os.path.join(file_weights_dir, f'{file_name}_best_LSTM_model.pth')))
    net.load_state_dict(torch.load(os.path.join(file_weights_dir, f'{file_name}_best_STAR_model.pth')))

    # 第二阶段训练
    phase2_start_time = time.perf_counter()
    print("——————————————————————第二阶段：冻结基础模型，训练 MLP 开始——————————————————————")
    for param in LSTMMain_model.parameters():
        param.requires_grad = False
    for param in net.parameters():
        param.requires_grad = False

    optimizer_MLP = torch.optim.Adam(mlp_model.parameters(), lr=learning_rate_base)
    best_val_loss_MLP = float('inf')

    for epoch in range(epochs_phase2):
        mlp_model.train()
        train_loss_sum_MLP = 0
        for step, (feature_, y_train_batch, feature_mark_, y_train_mark_batch, label_, _) in enumerate(train_loader):
            optimizer_MLP.zero_grad()
            with torch.no_grad():
                pred_lstm = LSTMMain_model(feature_, feature_mark_)
                pred_star = net(feature_, feature_mark_, y_train_batch, y_train_mark_batch, None)
                pred_star = pred_star.squeeze(1)
                pred_star = pred_star[:, -1].unsqueeze(1)
            combined_pred = torch.cat((pred_lstm, pred_star), dim=1)
            pred_mlp = mlp_model(combined_pred)
            loss_mlp = loss_func(pred_mlp, label_)
            loss_mlp.backward()
            optimizer_MLP.step()
            train_loss_sum_MLP += loss_mlp.item()

        mlp_model.eval()
        val_loss_sum_MLP = 0
        with torch.no_grad():
            for step, (feature_, y_val_batch, feature_mark_, y_val_mark_batch, label_, _) in enumerate(val_loader):
                pred_lstm = LSTMMain_model(feature_, feature_mark_)
                pred_star = net(feature_, feature_mark_, y_val_batch, y_val_mark_batch, None)
                pred_star = pred_star.squeeze(1)
                pred_star = pred_star[:, -1].unsqueeze(1)
                combined_pred = torch.cat((pred_lstm, pred_star), dim=1)
                pred_mlp = mlp_model(combined_pred)
                loss_mlp = loss_func(pred_mlp, label_)
                val_loss_sum_MLP += loss_mlp.item()

        if val_loss_sum_MLP < best_val_loss_MLP:
            best_val_loss_MLP = val_loss_sum_MLP
            torch.save(mlp_model.state_dict(), os.path.join(file_weights_dir, f'{file_name}_best_MLP_model.pth'))

        print(f"[{file_name}] Epoch {epoch+1}/{epochs_phase2} - MLP 训练损失: {train_loss_sum_MLP:.8f}, 验证损失: {val_loss_sum_MLP:.8f}")

    phase2_end_time = time.perf_counter()
    print("——————————————————————第二阶段：冻结基础模型，训练 MLP 结束——————————————————————")
    print(f"[{file_name}] 第二阶段训练用时: {phase2_end_time - phase2_start_time:.2f} 秒")

    # 测试集预测
    print("——————————————————————测试集预测开始——————————————————————")
    LSTMMain_model.load_state_dict(torch.load(os.path.join(file_weights_dir, f'{file_name}_best_LSTM_model.pth')))
    net.load_state_dict(torch.load(os.path.join(file_weights_dir, f'{file_name}_best_STAR_model.pth')))
    mlp_model.load_state_dict(torch.load(os.path.join(file_weights_dir, f'{file_name}_best_MLP_model.pth')))

    LSTMMain_model.eval()
    net.eval()
    mlp_model.eval()

    predictions = []
    labels = []
    global_indices = []

    with torch.no_grad():
        for step, (feature_, y_test_batch, feature_mark_, y_test_mark_batch, label_, label_idx) in enumerate(test_loader):
            pred_lstm = LSTMMain_model(feature_, feature_mark_)
            pred_star = net(feature_, feature_mark_, y_test_batch, y_test_mark_batch, None)
            pred_star = pred_star.squeeze(1)
            pred_star = pred_star[:, -1].unsqueeze(1)
            combined_pred = torch.cat((pred_lstm, pred_star), dim=1)
            pred_mlp = mlp_model(combined_pred)

            predictions.append(pred_mlp.cpu().numpy())
            labels.append(label_.cpu().numpy())
            global_indices.append(label_idx.cpu().numpy())

    predictions = np.vstack(predictions)
    labels = np.vstack(labels)
    global_indices = np.hstack(global_indices)

    # 反标准化
    temp_pred = np.zeros((predictions.shape[0], data_all_np.shape[1]))
    temp_label = np.zeros((labels.shape[0], data_all_np.shape[1]))
    temp_pred[:, -1] = predictions[:, 0]
    temp_label[:, -1] = labels[:, 0]

    pred_inverse = scaler.inverse_transform(temp_pred)[:, -1]
    label_inverse = scaler.inverse_transform(temp_label)[:, -1]

    county_indices = (global_indices // samples_per_county) + 1
    dates = df_stamp['date'].iloc[global_indices].values

    df_result = pd.DataFrame({
        'date': dates,
        'county_index': county_indices,
        'true_value': label_inverse,
        'pred_value': pred_inverse
    })

    # 创建/data/pre目录并保存预测结果
    pre_dir = './data/short_pre'
    if not os.path.exists(pre_dir):
        os.makedirs(pre_dir)
    file_pre_dir = os.path.join(pre_dir, file_name)
    if not os.path.exists(file_pre_dir):
        os.makedirs(file_pre_dir)

    df_result.to_csv(os.path.join(file_pre_dir, f'{file_name}_test_predictions.csv'), index=False)
    print(f"测试集预测结果已保存至 {file_pre_dir}/{file_name}_test_predictions.csv")

    # ===================== 计算并保存评估指标 =====================
    print("——————————————————————计算评估指标开始——————————————————————")

    # 计算评估指标
    df_eval = cal_eval(label_inverse, pred_inverse)

    # 打印评估指标
    print(f"评估指标:  {df_eval}")

    # 保存评估指标到 CSV 文件
    metrics_df = pd.DataFrame({
        'Metric': ['df_eval'],
        'Value': [df_eval]
    })
    metrics_df.to_csv(os.path.join(file_pre_dir, f'{file_name}_metrics.csv'), index=False)
    print(f"评估指标已保存至 {file_pre_dir}/{file_name}_metrics.csv")

    print("——————————————————————计算评估指标结束——————————————————————")
    
    # 释放内存
    del df, data_train, data_val, data_test, data_stamp_train, data_stamp_val, data_stamp_test
    del x_train, y_train, x_train_mark, y_train_mark, label_train_lstm, label_indices_train
    del x_val, y_val, x_val_mark, y_val_mark, label_val_lstm, label_indices_val
    del x_test, y_test, x_test_mark, y_test_mark, label_test_lstm, label_indices_test
    del LSTMMain_model, net, mlp_model, train_loader, val_loader, test_loader
    del predictions, labels, global_indices, df_result, metrics_df
    # del shap_values, shap_df, explainer, shap_input, shap_x, shap_x_mark, shap_indices
    torch.cuda.empty_cache()
    gc.collect()

    print(f"文件 {file_name} 的训练、测试与SHAP计算全部完成。总用时：{time.perf_counter() - file_start_time:.2f} 秒")

end_total = time.perf_counter()
print(f"全部文件的处理和训练完成，总用时: {end_total - start_total:.4f} 秒")
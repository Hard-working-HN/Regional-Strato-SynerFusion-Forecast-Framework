import torch
import torch.nn as nn
import torch.fft
from models.FECAM import dct_channel_block
from models.GraphBlock import GraphBlock

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device="cpu"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        output, (h, c) = self.lstm(input_seq, (h_0, c_0))
        return output, h


class LSTMMain(nn.Module):
    def __init__(self, input_size, output_len, lstm_hidden, lstm_layers,  batch_size, time_feature_dim=3, device="cpu"):
        super(LSTMMain, self).__init__()
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.input_size = input_size  # 输入特征的数量
        self.time_feature_dim = time_feature_dim  # 时间特征的数量
        self.total_input_size = self.input_size + self.time_feature_dim  # LSTM 的总输入维度
        self.lstmunit = LSTM(input_size, lstm_hidden, lstm_layers, batch_size, device)
        self.linear = nn.Linear(lstm_hidden, output_len)

    def forward(self, input_seq, input_time_features):
        bs, seq_len, nf = input_seq.shape
        dct_model = dct_channel_block(nf-1).to(device)
        input_seq_only_dct = dct_model(input_seq[:, :, :-1]).to(device)
        graph_block = GraphBlock(c_out=nf-1, d_model=nf-1, seq_len=seq_len).to(device)
        input_seq_dct_graph = graph_block(input_seq_only_dct).to(device)
        input_seq_combined = torch.cat((input_seq_dct_graph, input_seq[:, :, -1:]), dim=2)
        input_seq_with_time = torch.cat((input_time_features, input_seq_combined), dim=2)
        ula, h_out = self.lstmunit(input_seq_with_time)
        out = ula.contiguous().view(ula.shape[0] * ula.shape[1], self.lstm_hidden)
        out = self.linear(out)
        out = out.view(ula.shape[0], ula.shape[1], -1)
        out = out[:, -1, :]
        return out
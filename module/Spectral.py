import torch
from torch import nn
from torch.nn import functional as F


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x).squeeze()

        # assert x.size() == orig_q_size
        return x


class transformer(nn.Module):
    def __init__(self, hidden_size, ffn_size, out_size, dropout_rate, attention_dropout_rate, num_heads):
        super(transformer, self).__init__()
        self.relu = nn.ReLU()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, out_size)
        self.p = nn.Linear(out_size, out_size)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        output = self.fc(x)
        out = self.p(self.relu(output))

        return F.normalize(output, dim=-1), F.normalize(out, dim=-1)


class LSTM(nn.Module):
    def __init__(self, band, out_fea_num, b):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=b // 3,  # 每行的像素点个数
            hidden_size=256,
            num_layers=1,  # 层数
            batch_first=True,  # input和output会以batch_size为第一维度
        )

        self.final = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_fea_num)
        )
        self.band = band

    def forward(self, x):
        n, c = x.shape
        pixel = x
        pad1 = torch.zeros((pixel.shape[0], self.band, c // self.band)).type_as(x)

        for j in range(0, self.band):
            pad1[:, j, :] = pixel[:, j:j + (c // self.band - 1) * self.band + 1:self.band]

        # print(pixel.shape)
        x0, (h_n, c_n) = self.lstm(pad1)
        x = x0[:, -1, :]
        x1 = self.final(x)
        return F.normalize(x, dim=-1), F.normalize(x1, dim=-1)

# models.py
import torch.nn as nn
import torch

# 定义模型类
class SlowVariableAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob=0.3):
        super(SlowVariableAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_prob)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=input_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        encoded = self.dropout(encoded)
        decoded, _ = self.decoder(encoded)
        return encoded, decoded

class FastVariableAutoencoder(nn.Module):
    def __init__(self, input_channels, conv_out_channels, lstm_hidden_size, num_layers, dropout_prob=0.3):
        super(FastVariableAutoencoder, self).__init__()
        self.conv3 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=5, padding=2)
        self.encoder = nn.LSTM(input_size=conv_out_channels,
                               hidden_size=lstm_hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout_prob)
        self.decoder = nn.LSTM(input_size=lstm_hidden_size,
                               hidden_size=input_channels,
                               num_layers=num_layers,
                               batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整为 (batch_size, channels, seq_length)
        conv_out_3 = torch.relu(self.conv3(x))
        conv_out_5 = torch.relu(self.conv5(x))
        conv_out = torch.cat((conv_out_3, conv_out_5), dim=1)  # (batch_size, 32, seq_length)
        conv_out = conv_out.permute(0, 2, 1)  # 调整为 (batch_size, seq_length, 32)
        encoded, _ = self.encoder(conv_out)
        encoded = self.dropout(encoded)
        decoded, _ = self.decoder(encoded)
        return encoded, decoded

class FeatureFusionClassifier(nn.Module):
    def __init__(self, slow_feature_size, fast_feature_size, num_classes=4):
        super(FeatureFusionClassifier, self).__init__()
        self.total_feature_size = (slow_feature_size + fast_feature_size) * 5  # 5 是 time_steps 的数量
        self.fc1 = nn.Linear(self.total_feature_size, 64)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, slow_features, fast_features):
        slow_features = slow_features.reshape(slow_features.size(0), -1)  # 展平时间维度
        fast_features = fast_features.reshape(fast_features.size(0), -1)  # 展平时间维度
        concatenated_features = torch.cat((slow_features, fast_features), dim=-1)
        out = self.fc1(concatenated_features)
        out = self.dropout(out)
        out = torch.relu(out)
        logits = self.fc2(out)
        probabilities = torch.softmax(logits, dim=-1)  # 输出后验概率
        return probabilities

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)  # 移除 batch_first=True

    def forward(self, alkylation_probs, transalkylation_probs):
        # 将两个概率张量堆叠成一个序列
        combined = torch.stack((alkylation_probs, transalkylation_probs), dim=0)  # (2, batch_size, embed_dim)

        # 通过多头注意力层
        attn_output, _ = self.multihead_attn(combined, combined, combined)  # (2, batch_size, embed_dim)

        # 对序列长度维度取平均
        fused_output = attn_output.mean(dim=0)  # (batch_size, embed_dim)
        return fused_output

class FullProcessAttentionClassifier(nn.Module):
    def __init__(self, num_classes=4, num_heads=4, embed_dim=4):
        super(FullProcessAttentionClassifier, self).__init__()
        self.multi_head_attention = MultiHeadAttentionLayer(embed_dim, num_heads)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, alkylation_probs, transalkylation_probs):
        fused_features = self.multi_head_attention(alkylation_probs, transalkylation_probs)
        logits = self.fc(fused_features)
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities
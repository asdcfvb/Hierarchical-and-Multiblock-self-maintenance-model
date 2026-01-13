# train_and_save.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from model import SlowVariableAutoencoder, FastVariableAutoencoder, FeatureFusionClassifier, FullProcessAttentionClassifier

# 定义特征归一化器
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # 防止除以零
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        X_scaled = (X - self.mean_) / self.scale_
        return X_scaled

    def inverse_transform(self, X):
        return (X * self.scale_) + self.mean_

# 定义模块化的划分快慢变量函数
def calculate_msv_and_split_variables(x_train_alkylation_msv, x_train_transalkylation_msv, threshold_alkylation=0.95, threshold_transalkylation=0.95):
    # 对用于计算MSV的数据进行归一化
    scaler_alkylation_msv = StandardScaler()
    x_train_alkylation_msv = scaler_alkylation_msv.fit(x_train_alkylation_msv).transform(x_train_alkylation_msv)

    scaler_transalkylation_msv = StandardScaler()
    x_train_transalkylation_msv = scaler_transalkylation_msv.fit(x_train_transalkylation_msv).transform(x_train_transalkylation_msv)

    # 计算烷基化的平方速度(MSV)
    velocity_changes_alkylation = np.diff(x_train_alkylation_msv, axis=0)
    squared_velocity_alkylation = np.square(velocity_changes_alkylation)
    MSV_values_alkylation = np.mean(squared_velocity_alkylation, axis=0)

    # 计算烷基转移的平方速度(MSV)
    velocity_changes_transalkylation = np.diff(x_train_transalkylation_msv, axis=0)
    squared_velocity_transalkylation = np.square(velocity_changes_transalkylation)
    MSV_values_transalkylation = np.mean(squared_velocity_transalkylation, axis=0)

    # 计算烷基化的累积慢速贡献率
    inverse_MSV_alkylation = 1 / MSV_values_alkylation
    sorted_indices_alkylation = np.argsort(-inverse_MSV_alkylation)  # 按倒数从大到小排序
    sorted_inverse_MSV_alkylation = inverse_MSV_alkylation[sorted_indices_alkylation]
    cumulative_slow_contribution_rate_alkylation = np.cumsum(sorted_inverse_MSV_alkylation) / np.sum(sorted_inverse_MSV_alkylation)

    # 找出烷基化累积慢速贡献率达到阈值的慢变量
    slow_variable_indices_alkylation = sorted_indices_alkylation[cumulative_slow_contribution_rate_alkylation <= threshold_alkylation]
    faster_variable_indices_alkylation = sorted_indices_alkylation[cumulative_slow_contribution_rate_alkylation > threshold_alkylation]

    # 计算烷基转移的累积慢速贡献率
    inverse_MSV_transalkylation = 1 / MSV_values_transalkylation
    sorted_indices_transalkylation = np.argsort(-inverse_MSV_transalkylation)  # 按倒数从大到小排序
    sorted_inverse_MSV_transalkylation = inverse_MSV_transalkylation[sorted_indices_transalkylation]
    cumulative_slow_contribution_rate_transalkylation = np.cumsum(sorted_inverse_MSV_transalkylation) / np.sum(sorted_inverse_MSV_transalkylation)

    # 找出烷基转移累积慢速贡献率达到阈值的慢变量
    slow_variable_indices_transalkylation = sorted_indices_transalkylation[cumulative_slow_contribution_rate_transalkylation <= threshold_transalkylation]
    faster_variable_indices_transalkylation = sorted_indices_transalkylation[cumulative_slow_contribution_rate_transalkylation > threshold_transalkylation]

    return slow_variable_indices_alkylation, faster_variable_indices_alkylation, slow_variable_indices_transalkylation, faster_variable_indices_transalkylation

# 读取烷基化数据用于计算MSV
alkylation_df = pd.read_excel(r'datasets\Alkylation.xlsx', engine='openpyxl')
x_train_alkylation_msv = alkylation_df.iloc[:, 0:11].values  # 使用1-11列作为输入特征

# 读取烷基转移数据用于计算MSV
transalkylation_df = pd.read_excel(r'datasets\Transalkylation.xlsx', engine='openpyxl')
x_train_transalkylation_msv = transalkylation_df.iloc[:, 0:6].values  # 使用1-6列作为输入特征

# 调用模块化函数进行划分快慢变量
slow_indices_alkylation, faster_indices_alkylation, slow_indices_transalkylation, faster_indices_transalkylation = calculate_msv_and_split_variables(
    x_train_alkylation_msv, x_train_transalkylation_msv)

# 数据加载
train_df = pd.read_excel(r'datasets\Alkylation - train.xlsx', engine='openpyxl')

train_df_2 = pd.read_excel(r'datasets\Transalkylation - train.xlsx', engine='openpyxl')

# 标签编码
x_train_alkylation = train_df.iloc[:, 0:11].values  # 使用1-11列作为输入特征
y_train_alkylation = train_df.iloc[:, 11].values    # 使用第12列为输出标签
x_train_transalkylation = train_df_2.iloc[:, 0:6].values  # 使用1-6列作为输入特征
y_train_transalkylation = train_df_2.iloc[:, 6].values    # 使用第7列为输出标签


encode_label = LabelEncoder()
encode_label.fit(y_train_alkylation)
y_train_alkylation = encode_label.transform(y_train_alkylation)

encode_label.fit(y_train_transalkylation)
y_train_transalkylation = encode_label.transform(y_train_transalkylation)

# 对训练、验证和测试数据进行归一化
scaler_alkylation = StandardScaler()
scaler_transalkylation = StandardScaler()

x_train_alkylation = scaler_alkylation.fit(x_train_alkylation).transform(x_train_alkylation)

x_train_transalkylation = scaler_transalkylation.fit(x_train_transalkylation).transform(x_train_transalkylation)

# 分别处理较慢和快变量
x_train_slow_alkylation = x_train_alkylation[:, slow_indices_alkylation]
x_train_faster_alkylation = x_train_alkylation[:, faster_indices_alkylation]

x_train_slow_transalkylation = x_train_transalkylation[:, slow_indices_transalkylation]
x_train_faster_transalkylation = x_train_transalkylation[:, faster_indices_transalkylation]


# 时间序列滑动窗口处理
def create_sliding_window_data(data, labels, window_size):
    num_samples = data.shape[0] - window_size + 1
    x = np.array([data[i:i + window_size] for i in range(num_samples)])
    y = labels[window_size - 1:]
    return x, y

window_size = 5
x_train_alkylation, y_train_alkylation = create_sliding_window_data(x_train_alkylation, y_train_alkylation, window_size)

x_train_transalkylation, y_train_transalkylation = create_sliding_window_data(x_train_transalkylation, y_train_transalkylation, window_size)



# 处理慢变量和快变量的滑动窗
x_train_slow_alkylation, _ = create_sliding_window_data(x_train_slow_alkylation, y_train_alkylation, window_size)
x_train_faster_alkylation, _ = create_sliding_window_data(x_train_faster_alkylation, y_train_alkylation, window_size)


x_train_slow_transalkylation, _ = create_sliding_window_data(x_train_slow_transalkylation, y_train_transalkylation, window_size)
x_train_faster_transalkylation, _ = create_sliding_window_data(x_train_faster_transalkylation, y_train_transalkylation, window_size)


# 将数据转换为 PyTorch 张量
X_train_slow_alkylation = torch.tensor(x_train_slow_alkylation, dtype=torch.float32)
X_train_faster_alkylation = torch.tensor(x_train_faster_alkylation, dtype=torch.float32)


X_train_slow_transalkylation = torch.tensor(x_train_slow_transalkylation, dtype=torch.float32)
X_train_faster_transalkylation = torch.tensor(x_train_faster_transalkylation, dtype=torch.float32)



y_train_tensor_alkylation = torch.tensor(y_train_alkylation, dtype=torch.long)
y_train_tensor_transalkylation = torch.tensor(y_train_transalkylation, dtype=torch.long)


# 实例化模型
input_size_slow_alkylation = len(slow_indices_alkylation)
input_size_faster_alkylation = len(faster_indices_alkylation)
input_size_slow_transalkylation = len(slow_indices_transalkylation)
input_size_faster_transalkylation = len(faster_indices_transalkylation)

hidden_size = 32  # 隐藏层大小
num_layers = 2  # LSTM层数
conv_out_channels = 32  # 卷积层输出通道数
lstm_hidden_size = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化烷基化和烷基转移的慢变量和快变量自编码器
slow_ae_alkylation = SlowVariableAutoencoder(input_size=input_size_slow_alkylation, hidden_size=hidden_size, num_layers=num_layers).to(device)
slow_ae_transalkylation = SlowVariableAutoencoder(input_size=input_size_slow_transalkylation, hidden_size=hidden_size, num_layers=num_layers).to(device)

fast_ae_alkylation = FastVariableAutoencoder(input_channels=input_size_faster_alkylation, conv_out_channels=conv_out_channels, lstm_hidden_size=lstm_hidden_size, num_layers=num_layers).to(device)
fast_ae_transalkylation = FastVariableAutoencoder(input_channels=input_size_faster_transalkylation, conv_out_channels=conv_out_channels, lstm_hidden_size=lstm_hidden_size, num_layers=num_layers).to(device)

# 定义优化器
optimizer_slow_alkylation = optim.Adam(slow_ae_alkylation.parameters(), lr=0.00035, weight_decay=1e-5)
optimizer_slow_transalkylation = optim.Adam(slow_ae_transalkylation.parameters(), lr=0.0001, weight_decay=1e-5)
optimizer_fast_alkylation = optim.Adam(fast_ae_alkylation.parameters(), lr=0.00035, weight_decay=1e-5)
optimizer_fast_transalkylation = optim.Adam(fast_ae_transalkylation.parameters(), lr=0.0001, weight_decay=1e-5)

# 自定义损失函数：包括重建误差、SFA 约束和协方差约束
def slow_variable_loss(y_true, y_pred, encoded_features, alpha=3.5, beta=3.5):
    # 重建误差 (MSE)
    mse_loss = nn.MSELoss()(y_pred, y_true)

    # SFA 约束：计算特征在时间上的一阶差分
    diff = encoded_features[:, 1:, :] - encoded_features[:, :-1, :]
    sfa_loss = torch.mean(diff ** 2)

    # 协方差约束：使协方差矩阵接近单位矩阵
    batch_size, time_steps, feature_dim = encoded_features.shape
    cov_matrix = torch.bmm(encoded_features.permute(0, 2, 1), encoded_features) / time_steps
    identity = torch.eye(feature_dim).to(cov_matrix.device)
    cov_loss = torch.mean((cov_matrix - identity) ** 2)

    # 总损失
    total_loss = mse_loss + alpha * sfa_loss + beta * cov_loss
    return total_loss

num_epochs_feature_extraction = 200
batch_size = 16
num_batches_alkylation = len(X_train_slow_alkylation) // batch_size
num_batches_transalkylation = len(X_train_slow_transalkylation) // batch_size

train_losses_slow_alkylation = []
train_losses_fast_alkylation = []
train_losses_slow_transalkylation = []
train_losses_fast_transalkylation = []

for epoch in range(num_epochs_feature_extraction):
    # 设置模型为训练模式
    slow_ae_alkylation.train()
    fast_ae_alkylation.train()
    slow_ae_transalkylation.train()
    fast_ae_transalkylation.train()

    epoch_loss_slow_alkylation = 0.0
    epoch_loss_fast_alkylation = 0.0
    epoch_loss_slow_transalkylation = 0.0
    epoch_loss_fast_transalkylation = 0.0

    # 同时训练烷基化和烷基转移
    for i in range(max(num_batches_alkylation, num_batches_transalkylation)):
        # 训练烷基化模型
        if i < num_batches_alkylation:
            inputs_slow_alkylation = X_train_slow_alkylation[i * batch_size:(i + 1) * batch_size].to(device)
            inputs_faster_alkylation = X_train_faster_alkylation[i * batch_size:(i + 1) * batch_size].to(device)

            # 烷基化慢变量训练
            optimizer_slow_alkylation.zero_grad()
            encoded_slow, decoded_slow = slow_ae_alkylation(inputs_slow_alkylation)
            loss_slow_alkylation = slow_variable_loss(inputs_slow_alkylation, decoded_slow, encoded_slow)
            loss_slow_alkylation.backward()
            optimizer_slow_alkylation.step()
            epoch_loss_slow_alkylation += loss_slow_alkylation.item()

            # 烷基化快变量训练
            optimizer_fast_alkylation.zero_grad()
            encoded_fast, decoded_fast = fast_ae_alkylation(inputs_faster_alkylation)
            loss_fast_alkylation = nn.MSELoss()(inputs_faster_alkylation, decoded_fast)
            loss_fast_alkylation.backward()
            optimizer_fast_alkylation.step()
            epoch_loss_fast_alkylation += loss_fast_alkylation.item()

        # 训练烷基转移模型
        if i < num_batches_transalkylation:
            inputs_slow_transalkylation = X_train_slow_transalkylation[i * batch_size:(i + 1) * batch_size].to(device)
            inputs_faster_transalkylation = X_train_faster_transalkylation[i * batch_size:(i + 1) * batch_size].to(device)

            # 烷基转移慢变量训练
            optimizer_slow_transalkylation.zero_grad()
            encoded_slow, decoded_slow = slow_ae_transalkylation(inputs_slow_transalkylation)
            loss_slow_transalkylation = slow_variable_loss(inputs_slow_transalkylation, decoded_slow, encoded_slow)
            loss_slow_transalkylation.backward()
            optimizer_slow_transalkylation.step()
            epoch_loss_slow_transalkylation += loss_slow_transalkylation.item()

            # 烷基转移快变量训练
            optimizer_fast_transalkylation.zero_grad()
            encoded_fast, decoded_fast = fast_ae_transalkylation(inputs_faster_transalkylation)
            loss_fast_transalkylation = nn.MSELoss()(inputs_faster_transalkylation, decoded_fast)
            loss_fast_transalkylation.backward()
            optimizer_fast_transalkylation.step()
            epoch_loss_fast_transalkylation += loss_fast_transalkylation.item()

    # 记录训练损失
    train_losses_slow_alkylation.append(epoch_loss_slow_alkylation / num_batches_alkylation)
    train_losses_fast_alkylation.append(epoch_loss_fast_alkylation / num_batches_alkylation)
    train_losses_slow_transalkylation.append(epoch_loss_slow_transalkylation / num_batches_transalkylation)
    train_losses_fast_transalkylation.append(epoch_loss_fast_transalkylation / num_batches_transalkylation)

    # 打印每个epoch的损失
    print(f'Epoch {epoch + 1}/{num_epochs_feature_extraction}, '
          f'Alkylation Slow Train Loss: {train_losses_slow_alkylation[-1]:.4f}, '
          f'Alkylation Fast Train Loss: {train_losses_fast_alkylation[-1]:.4f}, '
          f'Transalkylation Slow Train Loss: {train_losses_slow_transalkylation[-1]:.4f}, '
          f'Transalkylation Fast Train Loss: {train_losses_fast_transalkylation[-1]:.4f}')

# 实例化子工序分类器
alkylation_classifier = FeatureFusionClassifier(slow_feature_size=hidden_size, fast_feature_size=lstm_hidden_size).to(device)
transalkylation_classifier = FeatureFusionClassifier(slow_feature_size=hidden_size, fast_feature_size=lstm_hidden_size).to(device)

# 定义子工序分类器的优化器和损失函数
optimizer_alkylation_classifier = optim.Adam(alkylation_classifier.parameters(), lr=0.0001)
optimizer_transalkylation_classifier = optim.Adam(transalkylation_classifier.parameters(), lr=0.0001)
classification_loss_fn = nn.CrossEntropyLoss()

# 实例化全流程层多头注意力分类器
full_process_classifier = FullProcessAttentionClassifier(num_classes=4, num_heads=4, embed_dim=4).to(device)
optimizer_full_process_classifier = optim.Adam(full_process_classifier.parameters(), lr=0.0001)

# 准备训练数据批次
num_batches_alkylation = len(X_train_slow_alkylation) // batch_size
num_batches_transalkylation = len(X_train_slow_transalkylation) // batch_size

# 初始化准确率和损失记录列表
accuracy_alkylation = []
accuracy_transalkylation = []
accuracy_full_process = []

loss_alkylation = []
loss_transalkylation = []
loss_full_process = []

num_epochs = 300

for epoch in range(num_epochs):
    alkylation_classifier.train()
    transalkylation_classifier.train()

    correct_alkylation = 0
    total_alkylation = 0
    correct_transalkylation = 0
    total_transalkylation = 0

    epoch_classification_loss_alkylation = 0.0
    epoch_classification_loss_transalkylation = 0.0

    # 烷基化分类器训练
    for i in range(num_batches_alkylation):
        slow_features_alkylation, _ = slow_ae_alkylation(
            X_train_slow_alkylation[i * batch_size:(i + 1) * batch_size].to(device))
        fast_features_alkylation, _ = fast_ae_alkylation(
            X_train_faster_alkylation[i * batch_size:(i + 1) * batch_size].to(device))

        optimizer_alkylation_classifier.zero_grad()
        alkylation_output = alkylation_classifier(slow_features_alkylation, fast_features_alkylation)
        alkylation_labels = y_train_tensor_alkylation[i * batch_size:(i + 1) * batch_size].to(device)

        # 计算分类损失
        loss_alkylation_batch = classification_loss_fn(alkylation_output, alkylation_labels)
        loss_alkylation_batch.backward()
        optimizer_alkylation_classifier.step()

        epoch_classification_loss_alkylation += loss_alkylation_batch.item()

        # 计算精度
        _, predicted_alkylation = torch.max(alkylation_output, 1)
        correct_alkylation += (predicted_alkylation == alkylation_labels).sum().item()
        total_alkylation += alkylation_labels.size(0)

    # 烷基转移分类器训练
    for i in range(num_batches_transalkylation):
        slow_features_transalkylation, _ = slow_ae_transalkylation(
            X_train_slow_transalkylation[i * batch_size:(i + 1) * batch_size].to(device))
        fast_features_transalkylation, _ = fast_ae_transalkylation(
            X_train_faster_transalkylation[i * batch_size:(i + 1) * batch_size].to(device))

        optimizer_transalkylation_classifier.zero_grad()
        transalkylation_output = transalkylation_classifier(slow_features_transalkylation,
                                                            fast_features_transalkylation)
        transalkylation_labels = y_train_tensor_transalkylation[i * batch_size:(i + 1) * batch_size].to(device)

        # 计算分类损失
        loss_transalkylation_batch = classification_loss_fn(transalkylation_output, transalkylation_labels)
        loss_transalkylation_batch.backward()
        optimizer_transalkylation_classifier.step()

        epoch_classification_loss_transalkylation += loss_transalkylation_batch.item()

        # 计算精度
        _, predicted_transalkylation = torch.max(transalkylation_output, 1)
        correct_transalkylation += (predicted_transalkylation == transalkylation_labels).sum().item()
        total_transalkylation += transalkylation_labels.size(0)

    # 计算子工序分类器的准确率
    alkylation_accuracy = 100 * correct_alkylation / total_alkylation
    transalkylation_accuracy = 100 * correct_transalkylation / total_transalkylation
    accuracy_alkylation.append(alkylation_accuracy)
    accuracy_transalkylation.append(transalkylation_accuracy)

    # 计算并记录烷基化和烷基转移的平均损失
    avg_loss_alkylation = epoch_classification_loss_alkylation / num_batches_alkylation
    avg_loss_transalkylation = epoch_classification_loss_transalkylation / num_batches_transalkylation
    loss_alkylation.append(avg_loss_alkylation)
    loss_transalkylation.append(avg_loss_transalkylation)

    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Alkylation Accuracy: {alkylation_accuracy:.2f}%, '
          f'Alkylation Loss: {avg_loss_alkylation:.4f}, '
          f'Transalkylation Accuracy: {transalkylation_accuracy:.2f}%, '
          f'Transalkylation Loss: {avg_loss_transalkylation:.4f}')

    # 全流程层自适应分类器训练
    full_process_classifier.train()

    correct_full_process = 0
    total_full_process = 0
    epoch_classification_loss_full_process = 0.0

    for i in range(num_batches_alkylation):
        slow_features_alkylation, _ = slow_ae_alkylation(
            X_train_slow_alkylation[i * batch_size:(i + 1) * batch_size].to(device))
        fast_features_alkylation, _ = fast_ae_alkylation(
            X_train_faster_alkylation[i * batch_size:(i + 1) * batch_size].to(device))
        slow_features_transalkylation, _ = slow_ae_transalkylation(
            X_train_slow_transalkylation[i * batch_size:(i + 1) * batch_size].to(device))
        fast_features_transalkylation, _ = fast_ae_transalkylation(
            X_train_faster_transalkylation[i * batch_size:(i + 1) * batch_size].to(device))

        alkylation_probs = alkylation_classifier(slow_features_alkylation, fast_features_alkylation)
        transalkylation_probs = transalkylation_classifier(slow_features_transalkylation, fast_features_transalkylation)

        optimizer_full_process_classifier.zero_grad()
        full_process_output = full_process_classifier(alkylation_probs, transalkylation_probs)
        full_process_labels = y_train_tensor_alkylation[i * batch_size:(i + 1) * batch_size].to(device)

        loss_full_process_batch = classification_loss_fn(full_process_output, full_process_labels)
        loss_full_process_batch.backward()
        optimizer_full_process_classifier.step()

        epoch_classification_loss_full_process += loss_full_process_batch.item()

        # 计算全流程分类器的精度
        _, predicted_full_process = torch.max(full_process_output, 1)
        correct_full_process += (predicted_full_process == full_process_labels).sum().item()
        total_full_process += full_process_labels.size(0)

    # 计算并记录全流程分类器的准确率和平均损失
    full_process_accuracy = 100 * correct_full_process / total_full_process
    accuracy_full_process.append(full_process_accuracy)

    avg_loss_full_process = epoch_classification_loss_full_process / num_batches_alkylation
    loss_full_process.append(avg_loss_full_process)

    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Full Process Accuracy: {full_process_accuracy:.2f}%, '
          f'Full Process Loss: {avg_loss_full_process:.4f}')

# 保存路径
save_path = r'save_model\semi_supervised_HMASCRAE.pth'
scaler_save_path = r'save_model\scaler_params.pth'

# 将所有模型保存在一起，并添加快慢变量索引
model_checkpoint = {
    'slow_ae_alkylation': slow_ae_alkylation.state_dict(),
    'fast_ae_alkylation': fast_ae_alkylation.state_dict(),
    'slow_ae_transalkylation': slow_ae_transalkylation.state_dict(),
    'fast_ae_transalkylation': fast_ae_transalkylation.state_dict(),
    'alkylation_classifier': alkylation_classifier.state_dict(),
    'transalkylation_classifier': transalkylation_classifier.state_dict(),
    'full_process_classifier': full_process_classifier.state_dict(),
    # 保存快慢变量索引
    'slow_indices_alkylation': slow_indices_alkylation,
    'faster_indices_alkylation': faster_indices_alkylation,
    'slow_indices_transalkylation': slow_indices_transalkylation,
    'faster_indices_transalkylation': faster_indices_transalkylation
}

torch.save(model_checkpoint, save_path)
print(f"Model saved to {save_path}")

# 保存标准化参数
scaler_checkpoint = {
    'scaler_alkylation_mean': scaler_alkylation.mean_,
    'scaler_alkylation_scale': scaler_alkylation.scale_,
    'scaler_transalkylation_mean': scaler_transalkylation.mean_,
    'scaler_transalkylation_scale': scaler_transalkylation.scale_
}

torch.save(scaler_checkpoint, scaler_save_path)
print(f"Scaler parameters saved to {scaler_save_path}")

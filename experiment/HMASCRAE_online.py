# adaptive_training.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import itertools
from model import SlowVariableAutoencoder, FastVariableAutoencoder, FeatureFusionClassifier, FullProcessAttentionClassifier


# 设置全局字体属性
english_font = FontProperties(fname=r'C:\Windows\Fonts\times.ttf', size=20)  # Times New Roman

# 更新matplotlib的rcParams，使其使用指定的字体
plt.rcParams['font.family'] = english_font.get_name()  # 设置全局字体
plt.rcParams['axes.labelsize'] = 20  # x轴和y轴标签字体大小
plt.rcParams['xtick.labelsize'] = 20  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 20  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 20  # 图例字体大小

# 特征归一化
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

# 加载新数据并进行预处理
new_data_alkylation = pd.read_excel('datasets\Alkylation - test.xlsx', engine='openpyxl')
new_data_transalkylation = pd.read_excel('datasets\Transalkylation - test.xlsx', engine='openpyxl')

# 特征和标签提取
x_new_alkylation = new_data_alkylation.iloc[:, 0:11].values  # 使用1-11列作为输入特征
y_new_alkylation = new_data_alkylation.iloc[:, 11].values  # 使用第12列为输出标签
x_new_transalkylation = new_data_transalkylation.iloc[:, 0:6].values  # 使用1-6列作为输入特征
y_new_transalkylation = new_data_transalkylation.iloc[:, 6].values  # 使用第7列为输出标签

# 标签编码
encode_label = LabelEncoder()
encode_label.fit(y_new_alkylation)
y_new_alkylation = encode_label.transform(y_new_alkylation)
encode_label.fit(y_new_transalkylation)
y_new_transalkylation = encode_label.transform(y_new_transalkylation)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载归一化参数
scaler_checkpoint = torch.load('save_model\scaler_params_最终.pth', map_location=device)

scaler_alkylation = StandardScaler()
scaler_alkylation.mean_ = scaler_checkpoint['scaler_alkylation_mean']
scaler_alkylation.scale_ = scaler_checkpoint['scaler_alkylation_scale']

scaler_transalkylation = StandardScaler()
scaler_transalkylation.mean_ = scaler_checkpoint['scaler_transalkylation_mean']
scaler_transalkylation.scale_ = scaler_checkpoint['scaler_transalkylation_scale']

# 对新数据进行归一化
x_new_alkylation = scaler_alkylation.transform(x_new_alkylation)
x_new_transalkylation = scaler_transalkylation.transform(x_new_transalkylation)

# 加载模型
load_path = 'save_model\semi_supervised_HMASCRAE_最终.pth'
checkpoint = torch.load(load_path, map_location=device)

# 加载快慢变量索引
slow_indices_alkylation = checkpoint['slow_indices_alkylation']
faster_indices_alkylation = checkpoint['faster_indices_alkylation']
slow_indices_transalkylation = checkpoint['slow_indices_transalkylation']
faster_indices_transalkylation = checkpoint['faster_indices_transalkylation']

# 确认索引长度
print(f"Slow indices alkylation: {len(slow_indices_alkylation)}")
print(f"Faster indices alkylation: {len(faster_indices_alkylation)}")
print(f"Slow indices transalkylation: {len(slow_indices_transalkylation)}")
print(f"Faster indices transalkylation: {len(faster_indices_transalkylation)}")

input_size_slow_alkylation = len(slow_indices_alkylation)
input_size_faster_alkylation = len(faster_indices_alkylation)
input_size_slow_transalkylation = len(slow_indices_transalkylation)
input_size_faster_transalkylation = len(faster_indices_transalkylation)

hidden_size = 32  # LSTM的隐藏层大小
num_layers = 2  # LSTM层数
conv_out_channels = 32  # 卷积层输出通道数
lstm_hidden_size = 32  # 快变量自编码器LSTM隐藏层大小

# 定义模型实例
slow_ae_alkylation = SlowVariableAutoencoder(input_size=input_size_slow_alkylation, hidden_size=hidden_size, num_layers=num_layers).to(device)
fast_ae_alkylation = FastVariableAutoencoder(input_channels=input_size_faster_alkylation, conv_out_channels=conv_out_channels, lstm_hidden_size=lstm_hidden_size, num_layers=num_layers).to(device)
alkylation_classifier = FeatureFusionClassifier(slow_feature_size=hidden_size, fast_feature_size=lstm_hidden_size).to(device)

slow_ae_transalkylation = SlowVariableAutoencoder(input_size=input_size_slow_transalkylation, hidden_size=hidden_size, num_layers=num_layers).to(device)
fast_ae_transalkylation = FastVariableAutoencoder(input_channels=input_size_faster_transalkylation, conv_out_channels=conv_out_channels, lstm_hidden_size=lstm_hidden_size, num_layers=num_layers).to(device)
transalkylation_classifier = FeatureFusionClassifier(slow_feature_size=hidden_size, fast_feature_size=lstm_hidden_size).to(device)

# 定义全流程分类器实例
full_process_classifier = FullProcessAttentionClassifier(num_classes=4, num_heads=4, embed_dim=4).to(device)

# 加载模型权重
slow_ae_alkylation.load_state_dict(checkpoint['slow_ae_alkylation'])
fast_ae_alkylation.load_state_dict(checkpoint['fast_ae_alkylation'])
alkylation_classifier.load_state_dict(checkpoint['alkylation_classifier'])

slow_ae_transalkylation.load_state_dict(checkpoint['slow_ae_transalkylation'])
fast_ae_transalkylation.load_state_dict(checkpoint['fast_ae_transalkylation'])
transalkylation_classifier.load_state_dict(checkpoint['transalkylation_classifier'])

full_process_classifier.load_state_dict(checkpoint['full_process_classifier'])  # 确保保存了全流程分类器的权重

# 将自编码器模型设置为评估模式并冻结参数
slow_ae_alkylation.eval()
fast_ae_alkylation.eval()
slow_ae_transalkylation.eval()
fast_ae_transalkylation.eval()

for param in slow_ae_alkylation.parameters():
    param.requires_grad = False
for param in fast_ae_alkylation.parameters():
    param.requires_grad = False
for param in slow_ae_transalkylation.parameters():
    param.requires_grad = False
for param in fast_ae_transalkylation.parameters():
    param.requires_grad = False

# 将新数据划分为慢变量和快变量
x_new_slow_alkylation = x_new_alkylation[:, slow_indices_alkylation]
x_new_faster_alkylation = x_new_alkylation[:, faster_indices_alkylation]
x_new_slow_transalkylation = x_new_transalkylation[:, slow_indices_transalkylation]
x_new_faster_transalkylation = x_new_transalkylation[:, faster_indices_transalkylation]

# 标签转换为张量
y_new_tensor_alkylation = torch.tensor(y_new_alkylation, dtype=torch.long).to(device)
y_new_tensor_transalkylation = torch.tensor(y_new_transalkylation, dtype=torch.long).to(device)

# 滑动窗口处理
window_size = 5
def create_sliding_window_data(data, labels, window_size):
    num_samples = data.shape[0] - window_size + 1
    x = np.array([data[i:i + window_size] for i in range(num_samples)])
    y = labels[window_size - 1:]
    return x, y

x_new_slow_alkylation, y_new_alkylation = create_sliding_window_data(x_new_slow_alkylation, y_new_alkylation, window_size)
x_new_faster_alkylation, _ = create_sliding_window_data(x_new_faster_alkylation, y_new_alkylation, window_size)
x_new_slow_transalkylation, y_new_transalkylation = create_sliding_window_data(x_new_slow_transalkylation, y_new_transalkylation, window_size)
x_new_faster_transalkylation, _ = create_sliding_window_data(x_new_faster_transalkylation, y_new_transalkylation, window_size)

# 转换为 PyTorch 张量
X_new_slow_alkylation = torch.tensor(x_new_slow_alkylation, dtype=torch.float32).to(device)
X_new_faster_alkylation = torch.tensor(x_new_faster_alkylation, dtype=torch.float32).to(device)
y_new_tensor_alkylation = torch.tensor(y_new_alkylation, dtype=torch.long).to(device)

X_new_slow_transalkylation = torch.tensor(x_new_slow_transalkylation, dtype=torch.float32).to(device)
X_new_faster_transalkylation = torch.tensor(x_new_faster_transalkylation, dtype=torch.float32).to(device)
y_new_tensor_transalkylation = torch.tensor(y_new_transalkylation, dtype=torch.long).to(device)

# 定义组合损失函数
def combined_loss(mse_targets, mse_predictions, classification_targets, classification_outputs, alpha=0.1):
    mse_loss = nn.MSELoss()(mse_predictions, mse_targets)
    ce_loss = nn.CrossEntropyLoss()(classification_outputs, classification_targets)
    return alpha * mse_loss + (1 - alpha) * ce_loss

# 自适应训练，只训练分类器
optimizer_alkylation_classifier = optim.Adam(alkylation_classifier.parameters(), lr=0.0055)
optimizer_transalkylation_classifier = optim.Adam(transalkylation_classifier.parameters(), lr=0.0018)

# 使用学习率调度器（自适应学习率）
scheduler_alkylation = optim.lr_scheduler.ReduceLROnPlateau(optimizer_alkylation_classifier, mode='min', factor=0.5, patience=2, verbose=True)
scheduler_transalkylation = optim.lr_scheduler.ReduceLROnPlateau(optimizer_transalkylation_classifier, mode='min', factor=0.5, patience=2, verbose=True)

# 自适应训练参数
batch_size = 300
num_batches_new_data_alkylation = len(X_new_slow_alkylation) // batch_size
num_batches_new_data_transalkylation = len(X_new_slow_transalkylation) // batch_size

# 阈值设置
threshold_f1_alkylation = 0.98  # 烷基化的F1分数阈值
threshold_f1_transalkylation = 0.98  # 烷基转移的F1分数阈值

# 记录哪些批次触发了自适应训练
adaptation_trigger_batches_alkylation = []
adaptation_trigger_batches_transalkylation = []

# 记录每个批次的F1分数（自适应前和自适应后）
f1_scores_alkylation_before = []
f1_scores_alkylation_after = []
f1_scores_transalkylation_before = []
f1_scores_transalkylation_after = []

def evaluate_classifier(classifier, slow_ae, fast_ae, X_slow, X_faster, y_true):
    """
    评估分类器的F1分数。

    参数：
        classifier: 分类器（例如 alkylation_classifier 或 transalkylation_classifier）
        slow_ae: 慢通道自编码器（例如 slow_ae_alkylation 或 slow_ae_transalkylation）
        fast_ae: 快通道自编码器（例如 fast_ae_alkylation 或 fast_ae_transalkylation）
        X_slow: 慢通道输入数据
        X_faster: 快通道输入数据
        y_true: 真实标签

    返回：
        f1: 分类器的F1分数
        y_pred: 预测结果
    """
    classifier.eval()
    with torch.no_grad():
        # 获取特征
        slow_features, _ = slow_ae(X_slow)
        fast_features, _ = fast_ae(X_faster)

        # 使用分类器进行预测
        output = classifier(slow_features, fast_features)

        # 获取预测类别
        _, predicted = torch.max(output, dim=1)

    # 计算F1分数
    y_pred = predicted.cpu().numpy()
    f1 = f1_score(y_true.cpu().numpy(), y_pred, average='weighted')

    return f1, y_pred

# 决策层：融合两个子工序的结果
def decision_layer(alkylation_classifier, transalkylation_classifier,
                   slow_ae_alkylation, fast_ae_alkylation,
                   slow_ae_transalkylation, fast_ae_transalkylation,
                   X_slow_alkylation, X_faster_alkylation,
                   X_slow_transalkylation, X_faster_transalkylation,
                   classifier):
    """
    使用多头注意力机制融合两个子工序的后验概率，进行全流程分类。
    参数：
        alkylation_classifier: 烷基化分类器
        transalkylation_classifier: 烷基转移分类器
        slow_ae_alkylation, fast_ae_alkylation: 烷基化自编码器
        slow_ae_transalkylation, fast_ae_transalkylation: 烷基转移自编码器
        X_slow_alkylation, X_faster_alkylation: 烷基化输入特征（慢和快通道）
        X_slow_transalkylation, X_faster_transalkylation: 烷基转移输入特征（慢和快通道）
        classifier: 全流程分类器（包含多头注意力机制）
    返回：
        final_pred: 融合后的最终预测类别
    """
    alkylation_classifier.eval()
    transalkylation_classifier.eval()
    classifier.eval()
    with torch.no_grad():
        # 烷基化分类
        slow_features_alkylation, _ = slow_ae_alkylation(X_slow_alkylation)
        fast_features_alkylation, _ = fast_ae_alkylation(X_faster_alkylation)
        output_alkylation = alkylation_classifier(slow_features_alkylation, fast_features_alkylation)

        # 烷基转移分类
        slow_features_transalkylation, _ = slow_ae_transalkylation(X_slow_transalkylation)
        fast_features_transalkylation, _ = fast_ae_transalkylation(X_faster_transalkylation)
        output_transalkylation = transalkylation_classifier(slow_features_transalkylation, fast_features_transalkylation)

        # 使用全流程分类器融合两个子工序的后验概率
        final_output = classifier(output_alkylation, output_transalkylation)

        # 获取最终预测类别
        _, final_pred = torch.max(final_output, dim=1)

    return final_pred

# 获取全流程分类器的预测
def get_full_process_predictions(full_process_classifier, alkylation_classifier, transalkylation_classifier,
                                 slow_ae_alkylation, fast_ae_alkylation,
                                 slow_ae_transalkylation, fast_ae_transalkylation,
                                 X_slow_alkylation, X_faster_alkylation,
                                 X_slow_transalkylation, X_faster_transalkylation):
    """
    获取全流程分类器的预测结果。

    参数：
        全部模型和特征数据。

    返回：
        final_pred: 全流程的预测结果
    """
    full_process_classifier.eval()
    alkylation_classifier.eval()
    transalkylation_classifier.eval()
    with torch.no_grad():
        # 烷基化分类
        slow_features_alkylation, _ = slow_ae_alkylation(X_slow_alkylation)
        fast_features_alkylation, _ = fast_ae_alkylation(X_faster_alkylation)
        output_alkylation = alkylation_classifier(slow_features_alkylation, fast_features_alkylation)

        # 烷基转移分类
        slow_features_transalkylation, _ = slow_ae_transalkylation(X_slow_transalkylation)
        fast_features_transalkylation, _ = fast_ae_transalkylation(X_faster_transalkylation)
        output_transalkylation = transalkylation_classifier(slow_features_transalkylation, fast_features_transalkylation)

        # 全流程分类
        final_output = full_process_classifier(output_alkylation, output_transalkylation)
        _, final_pred = torch.max(final_output, dim=1)

    return final_pred.cpu().numpy()

# 自适应训练过程 - 逐批次评估和训练（包括决策层融合）
for i in range(max(num_batches_new_data_alkylation, num_batches_new_data_transalkylation)):
    print(f"\nProcessing Batch {i + 1}")

    # 烷基化部分
    if i < num_batches_new_data_alkylation:
        # 选择批次数据
        batch_inputs_slow = X_new_slow_alkylation[i * batch_size:(i + 1) * batch_size]
        batch_inputs_faster = X_new_faster_alkylation[i * batch_size:(i + 1) * batch_size]
        batch_labels = y_new_tensor_alkylation[i * batch_size:(i + 1) * batch_size]

        # 评估批次 - 自适应前
        f1_alkylation, _ = evaluate_classifier(alkylation_classifier, slow_ae_alkylation, fast_ae_alkylation,
                                               batch_inputs_slow, batch_inputs_faster, batch_labels)
        f1_scores_alkylation_before.append(f1_alkylation)
        f1_scores_alkylation_after.append(f1_alkylation)  # 初始时自适应前后F1分数相同
        print(f'[Alkylation] F1 Score Before Adaptation: {f1_alkylation:.4f}')

        if f1_alkylation < threshold_f1_alkylation:
            print("[Alkylation] F1 score below threshold. Performing adaptive training.")
            adaptation_trigger_batches_alkylation.append(i)  # 记录触发自适应的批次

            # 零梯度，进行自适应训练
            optimizer_alkylation_classifier.zero_grad()

            # 获取特征
            slow_features, slow_decoded = slow_ae_alkylation(batch_inputs_slow)
            fast_features, fast_decoded = fast_ae_alkylation(batch_inputs_faster)

            # 使用分类器进行预测
            outputs = alkylation_classifier(slow_features, fast_features)
            loss = combined_loss(mse_targets=batch_inputs_slow, mse_predictions=slow_decoded,
                                 classification_targets=batch_labels, classification_outputs=outputs)
            loss.backward()
            optimizer_alkylation_classifier.step()
            scheduler_alkylation.step(loss.item())

            # 更新当前批次的F1分数为自适应后的F1分数
            f1_alkylation, _ = evaluate_classifier(alkylation_classifier, slow_ae_alkylation, fast_ae_alkylation,
                                                   batch_inputs_slow, batch_inputs_faster, batch_labels)
            f1_scores_alkylation_after[-1] = f1_alkylation
            print(f'[Alkylation] F1 Score After Adaptation: {f1_alkylation:.4f}')

    # 烷基转移部分
    if i < num_batches_new_data_transalkylation:
        # 选择批次数据
        batch_inputs_slow = X_new_slow_transalkylation[i * batch_size:(i + 1) * batch_size]
        batch_inputs_faster = X_new_faster_transalkylation[i * batch_size:(i + 1) * batch_size]
        batch_labels = y_new_tensor_transalkylation[i * batch_size:(i + 1) * batch_size]

        # 评估批次 - 自适应前
        f1_transalkylation, _ = evaluate_classifier(transalkylation_classifier, slow_ae_transalkylation,
                                                    fast_ae_transalkylation, batch_inputs_slow, batch_inputs_faster,
                                                    batch_labels)
        f1_scores_transalkylation_before.append(f1_transalkylation)
        f1_scores_transalkylation_after.append(f1_transalkylation)  # 初始时自适应前后F1分数相同
        print(f'[Transalkylation] F1 Score Before Adaptation: {f1_transalkylation:.4f}')

        if f1_transalkylation < threshold_f1_transalkylation:
            print("[Transalkylation] F1 score below threshold. Performing adaptive training.")
            adaptation_trigger_batches_transalkylation.append(i)  # 记录触发自适应的批次

            # 零梯度，进行自适应训练
            optimizer_transalkylation_classifier.zero_grad()

            # 获取特征
            slow_features, slow_decoded = slow_ae_transalkylation(batch_inputs_slow)
            fast_features, fast_decoded = fast_ae_transalkylation(batch_inputs_faster)

            # 使用分类器进行预测
            outputs = transalkylation_classifier(slow_features, fast_features)
            loss = combined_loss(mse_targets=batch_inputs_slow, mse_predictions=slow_decoded,
                                 classification_targets=batch_labels, classification_outputs=outputs)
            loss.backward()
            optimizer_transalkylation_classifier.step()
            scheduler_transalkylation.step(loss.item())

            # 更新当前批次的F1分数为自适应后的F1分数
            f1_transalkylation, _ = evaluate_classifier(transalkylation_classifier, slow_ae_transalkylation,
                                                        fast_ae_transalkylation, batch_inputs_slow, batch_inputs_faster,
                                                        batch_labels)
            f1_scores_transalkylation_after[-1] = f1_transalkylation
            print(f'[Transalkylation] F1 Score After Adaptation: {f1_transalkylation:.4f}')

    # 融合两个子工序的结果（决策层）
    if i < num_batches_new_data_alkylation and i < num_batches_new_data_transalkylation:
        # 获取当前批次的预测结果
        final_pred = decision_layer(alkylation_classifier, transalkylation_classifier,
                                    slow_ae_alkylation, fast_ae_alkylation,
                                    slow_ae_transalkylation, fast_ae_transalkylation,
                                    X_new_slow_alkylation[i * batch_size:(i + 1) * batch_size],
                                    X_new_faster_alkylation[i * batch_size:(i + 1) * batch_size],
                                    X_new_slow_transalkylation[i * batch_size:(i + 1) * batch_size],
                                    X_new_faster_transalkylation[i * batch_size:(i + 1) * batch_size],
                                    full_process_classifier)
        # 计算全流程的F1分数
        f1_full_process = f1_score(y_new_tensor_alkylation[i * batch_size:(i + 1) * batch_size].cpu().numpy(),
                                   final_pred.cpu().numpy(), average='weighted')
        print(f'[Full Process] F1 Score: {f1_full_process:.4f}')

# 绘制混淆矩阵的函数
def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues, filename=None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Confusion matrix")
    print(cm)

    plt.figure(figsize=(8, 6))  # 根据需要调整图像大小
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=20)

    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, format='svg', dpi=300)
        print(f"Confusion matrix saved as {filename}")
    else:
        plt.show()
    plt.close()

def get_all_predictions(classifier, slow_ae, fast_ae, X_slow, X_faster):
    classifier.eval()
    with torch.no_grad():
        slow_features, _ = slow_ae(X_slow)
        fast_features, _ = fast_ae(X_faster)
        outputs = classifier(slow_features, fast_features)
        _, predicted = torch.max(outputs, dim=1)
    return predicted.cpu().numpy()

# 获取所有烷基化和烷基转移的预测结果
all_pred_alkylation = get_all_predictions(
    alkylation_classifier, slow_ae_alkylation, fast_ae_alkylation,
    X_new_slow_alkylation, X_new_faster_alkylation
)
all_pred_transalkylation = get_all_predictions(
    transalkylation_classifier, slow_ae_transalkylation, fast_ae_transalkylation,
    X_new_slow_transalkylation, X_new_faster_transalkylation
)

# 获取全流程分类器的预测结果
all_pred_full_process = get_full_process_predictions(
    full_process_classifier, alkylation_classifier, transalkylation_classifier,
    slow_ae_alkylation, fast_ae_alkylation,
    slow_ae_transalkylation, fast_ae_transalkylation,
    X_new_slow_alkylation, X_new_faster_alkylation,
    X_new_slow_transalkylation, X_new_faster_transalkylation
)

# 计算混淆矩阵
cm_alkylation = confusion_matrix(y_new_tensor_alkylation.cpu().numpy(), all_pred_alkylation)
cm_transalkylation = confusion_matrix(y_new_tensor_transalkylation.cpu().numpy(), all_pred_transalkylation)
cm_full_process = confusion_matrix(y_new_tensor_alkylation.cpu().numpy(), all_pred_full_process)

# 计算每个类别的F1分数
f1_alkylation_per_class = f1_score(
    y_new_tensor_alkylation.cpu().numpy(),
    all_pred_alkylation,
    average=None
)
f1_transalkylation_per_class = f1_score(
    y_new_tensor_transalkylation.cpu().numpy(),
    all_pred_transalkylation,
    average=None
)
f1_full_process_per_class = f1_score(
    y_new_tensor_alkylation.cpu().numpy(),  # 假设这里是full_process的标签
    all_pred_full_process,
    average=None
)

# 计算宏平均F1分数
f1_alkylation_macro = f1_score(
    y_new_tensor_alkylation.cpu().numpy(),
    all_pred_alkylation,
    average='macro'
)
f1_transalkylation_macro = f1_score(
    y_new_tensor_transalkylation.cpu().numpy(),
    all_pred_transalkylation,
    average='macro'
)
f1_full_process_macro = f1_score(
    y_new_tensor_alkylation.cpu().numpy(),  # 假设这里是full_process的标签
    all_pred_full_process,
    average='macro'
)

# 计算加权平均F1分数
f1_alkylation_weighted = f1_score(
    y_new_tensor_alkylation.cpu().numpy(),
    all_pred_alkylation,
    average='weighted'
)
f1_transalkylation_weighted = f1_score(
    y_new_tensor_transalkylation.cpu().numpy(),
    all_pred_transalkylation,
    average='weighted'
)
f1_full_process_weighted = f1_score(
    y_new_tensor_alkylation.cpu().numpy(),  # 假设这里是full_process的标签
    all_pred_full_process,
    average='weighted'
)

# 输出结果
print("Alkylation F1 Scores per class:", f1_alkylation_per_class)
print("Alkylation Macro F1 Score:", f1_alkylation_macro)
print("Alkylation Weighted F1 Score:", f1_alkylation_weighted)

print("Transalkylation F1 Scores per class:", f1_transalkylation_per_class)
print("Transalkylation Macro F1 Score:", f1_transalkylation_macro)
print("Transalkylation Weighted F1 Score:", f1_transalkylation_weighted)

print("Full Process F1 Scores per class:", f1_full_process_per_class)
print("Full Process Macro F1 Score:", f1_full_process_macro)
print("Full Process Weighted F1 Score:", f1_full_process_weighted)

# 输出每个类别的F1分数
classes = ["0", "1", "2", "3"]

print("\n=== F1 Scores per Class ===")
print("Alkylation F1 Scores:")
for cls, score in zip(classes, f1_alkylation_per_class):
    print(f"  Class {cls}: {score:.4f}")

print("\nTransalkylation F1 Scores:")
for cls, score in zip(classes, f1_transalkylation_per_class):
    print(f"  Class {cls}: {score:.4f}")

print("\nFull Process F1 Scores:")
for cls, score in zip(classes, f1_full_process_per_class):
    print(f"  Class {cls}: {score:.4f}")

# 绘制并保存混淆矩阵
plot_confusion_matrix(
    cm_alkylation, classes=classes, normalize=True, filename='cm_alkylation.svg'
)
plot_confusion_matrix(
    cm_transalkylation, classes=classes, normalize=True, filename='cm_transalkylation.svg'
)
plot_confusion_matrix(
    cm_full_process, classes=classes, normalize=True, filename='cm_full_process.svg'
)

# 获取所有预测和真实值为NumPy数组
true_values_alkylation = y_new_tensor_alkylation.cpu().numpy()
predicted_values_alkylation = all_pred_alkylation

true_values_transalkylation = y_new_tensor_transalkylation.cpu().numpy()
predicted_values_transalkylation = all_pred_transalkylation

true_values_full_process = y_new_tensor_alkylation.cpu().numpy()  # 假设全流程标签与烷基化标签相同
predicted_values_full_process = all_pred_full_process

# 定义绘制在线评估结果的函数
def plot_evaluation_results(true_values, predicted_values, yticks, xlabel='Samples', ylabel='Level', filename=None):
    """
    绘制在线评估结果图并保存为SVG格式
    true_values: 真实值
    predicted_values: 预测值
    yticks: y轴刻度
    xlabel: x轴标签 (默认: 'Samples')
    ylabel: y轴标签 (默认: 'Level')
    filename: 保存文件的路径 (可选)
    """
    # 创建图形，并设置图形尺寸（宽度: 14英寸，高度: 4英寸）
    fig, ax = plt.subplots(figsize=(14, 4))

    # 绘制真实值和预测值的散点图
    ax.scatter(range(len(true_values)), true_values, label='True', color='red', marker='^', s=40, alpha=0.6)
    ax.scatter(range(len(true_values)), predicted_values, label='Assessment', color='blue', marker='o', s=20, alpha=0.6)
    # 设置x轴、y轴标签
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 设置图例位置并调整大小参数
    legend = ax.legend(
        loc='upper right',          # 图例锚点位置
        bbox_to_anchor=(1, 0.95),   # 图例相对于锚点的坐标 (x, y)
        fontsize=16,                # 保持字体大小不变
        borderpad=0.3,              # 图例边框与内容的填充（减小）
        labelspacing=0.3,           # 图例标签之间的垂直间距（减小）
        handletextpad=0.2,          # 图例中标记与标签之间的距离（减小）
        handlelength=1.0,           # 图例中标记与标签之间的长度（适当调整）
        frameon=True                 # 保持图例边框
    )

    # 设置刻度
    ax.set_xticks(range(0, len(true_values), 200))
    ax.set_yticks(yticks)

    # 关闭网格
    ax.grid(False)

    # 手动调整底部边距以确保x轴标签显示
    plt.subplots_adjust(bottom=0.2)  # 根据需要调整此值

    # 保存为SVG文件，设置DPI
    if filename:
        plt.savefig(filename, format='svg', dpi=300)
        print(f"Evaluation results plot saved as {filename}")
    else:
        plt.show()

    # 关闭图形以释放内存
    plt.close()

# 绘制并保存烷基化子工序的在线评估结果图
plot_evaluation_results(
    true_values_alkylation, predicted_values_alkylation, [0, 1, 2, 3],
    xlabel='Samples', ylabel='Alkylation Level', filename='alkylation_results.svg'
)

# 绘制并保存烷基转移子工序的在线评估结果图
plot_evaluation_results(
    true_values_transalkylation, predicted_values_transalkylation, [0, 1, 2, 3],
    xlabel='Samples', ylabel='Transalkylation Level', filename='transalkylation_results.svg'
)

# 绘制并保存全流程层的在线评估结果图
plot_evaluation_results(
    true_values_full_process, predicted_values_full_process, [0, 1, 2, 3],
    xlabel='Samples', ylabel='Level', filename='full_process_results.svg'
)




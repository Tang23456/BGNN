from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                             f1_score, accuracy_score, recall_score, precision_score, confusion_matrix)
from sklearn.model_selection import KFold
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import torch.optim as optim
from utils import *
from model import *
import pandas as pd
import numpy as np
import matplotlib
import torch
import csv
import random
import gc


#torch.autograd.set_detect_anomaly(True)
# 检查CUDA是否可用
out = []  # 用于存储每一折的训练集和测试集索引

iter = 0
k_split = 5
set_seed(123)

matplotlib.use('TkAgg')
criterion = torch.nn.MSELoss()
kf = KFold(n_splits=k_split, shuffle=True, random_state=123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']# 使用颜色编码定义颜色

matplotlib.use('TkAgg')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # 使用颜色编码定义颜色
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

set_seed(123)
epochs = 200
k_split = 5
output_dim = 32  # 低维输出维度

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# input_dim = 1396+43  # phendb  1396+43 peryton    1396+43     disbiome 1622+374 疾病特征维度     HMDAD 292+39

n = 4
# lambda_mse = 4    #4
# lambda_l2 = 5e-2  #  5e-2
# # 3e-1  图神经网络层数为2的时候中间层为62
# # lambda_constrate = 3  # 2
# lambda_constrate = 9e-2


lambda_mse = 4    #4
lambda_l2 = 5e-2  #  5e-2
# 3e-1  图神经网络层数为2的时候中间层为62
# lambda_constrate = 3  # 2
lambda_constrate = 9e-2


sk_tprs = []
sk_aucs = []
sk_precisions = []
sk_recalls = []
sk_average_precisions = []
sk_fpr = []

metrics_summary = {
    'f1_scores': [],
    'accuracies': [],
    'recalls': [],
    'specificities': [],
    'precisions': []
}

fold_metrics = {
    'aucs': [],
    'auprs': [],
    'f1_scores': [],
    'accuracies': []
}

fold_metrics_file = 'fold_metrics.csv'
with open(fold_metrics_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Fold', 'AUC', 'AUPR', 'F1', 'Accuracy'])


# HMDAD 39 292

A = pd.read_excel('./dataset/HMDAD/adj_mat.xlsx')
disease_chemical = pd.read_csv('./dataset/HMDAD/化学-疾病/complete_disease_similarity_matrix.csv')
disease_gene = pd.read_csv('./dataset/HMDAD/基因-疾病/complete_disease_similarity_matrix.csv')
disease_symptoms = pd.read_csv('./dataset/HMDAD/疾病-症状/complete_disease_similarity_matrix.csv')
disease_Semantics = pd.read_csv('./dataset/HMDAD/疾病-语义/similarity_matrix_model2.csv', header=None)
disease_pathway = pd.read_csv('./dataset/HMDAD/疾病-通路/complete_disease_similarity_matrix.csv')

micro_cos = pd.read_csv('./dataset/HMDAD/基于关联矩阵的微生物功能/Cosine_Sim.csv')
micro_gip = pd.read_csv('./dataset/HMDAD/基于关联矩阵的微生物功能/GIP_Sim.csv')
micro_sem = pd.read_csv('./dataset/HMDAD/基于疾病语义的微生物功能/functional_similarity2_matrix.csv')
micro_fun1 = pd.read_csv('./dataset/HMDAD/微生物-功能/complete_microbe_associations_ds2_matrix.csv')
micro_fun2 = pd.read_csv('./dataset/HMDAD/微生物-功能/complete_microbe_similarities_ds2_matrix.csv')
A = A.iloc[1:, 1:]

disease_chemical = disease_chemical.iloc[:, 1:]
disease_gene = disease_gene.iloc[:, 1:]
disease_symptoms = disease_symptoms.iloc[:, 1:]
disease_pathway = disease_pathway.iloc[:, 1:]
micro_cos = micro_cos.iloc[:, 1:]
micro_gip = micro_gip.iloc[:, 1:]
micro_fun1 = micro_fun1.iloc[:, 1:]
micro_fun2 = micro_fun2.iloc[:, 1:]

microbiome_matrices = [micro_cos.values, micro_gip.values, micro_sem.values, micro_fun1.values, micro_fun2.values]

disease_matrices = [disease_chemical.values, disease_gene.values, disease_symptoms.values, disease_Semantics.values,
                    disease_pathway.values]

sim_d, sim_m = calculate_combined_similarity(disease_matrices, microbiome_matrices)

input_dim = sim_m.shape[0] + sim_d.shape[0]
A = A.T.to_numpy()

x, y = A.shape
score_matrix = np.zeros([x, y])  # 初始化评分矩阵
md = A
mm = sim_m
dd = sim_d

deep_A = calculate_metapath_optimized(mm, dd, md, n)

samples, A_neg = get_all_pairs(A, deep_A)  # 返回[i, j, i_hat, j_hat] i 微生物 j疾病
samples = np.array(samples)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()

# cross validation
kf = KFold(n_splits=k_split, shuffle=True, random_state=123)  # 定义10折交叉验证
iter_ = 0  # control each iterator  控制迭代次数
sum_score = 0  # 总分数初始化为0
out = []  # 用于存储每一折的训练集和测试集索引
test_label_score = {}  # 存储测试标签和预测得分的字典
criterion = torch.nn.MSELoss()

prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))
iter_ = 0
out = []  # 用于存储每一折的训练集和测试集索引
test_label_score = {}  # 存储测试标签和预测得分的字典

adj_A_pos = build_pos_adjacency_matrix(A)
adj_A_neg = build_perturbed_adjacency_matrix_with_antidiagonal(A_neg)
lambda_l2 = (lambda_l2 * 39 * 292) / (A.shape[0] * A.shape[1])


precisions = []
precisions_6_MGCN = []            #precisions_Only_pos
precisions_3_MGCN_3_GCN= []       #precisions_Only_neg
precisions_6_GCN = []             #precisions_A_neg
precisions_6_spilt = []           #precisions_A_neg_change

sk_tprs = []
sk_aucs = []
sk_precisions = []
sk_recalls = []
sk_average_precisions = []
sk_fpr= []


sk_tprs_Only_pos = []
sk_aucs_Only_pos = []
sk_precisions_Only_pos = []
sk_recalls_Only_pos = []
sk_average_precisions_Only_pos = []
sk_fpr_Only_pos = []


sk_tprs_Only_neg = []
sk_aucs_Only_neg = []
sk_precisions_Only_neg = []
sk_recalls_Only_neg  = []
sk_average_precisions_Only_neg = []
sk_fpr_Only_neg = []


fig, ax = plt.subplots()
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
metrics_summary = {
    'f1_scores': [],
    'accuracies': [],
    'recalls': [],
    'specificities': [],
    'precisions': []
}

sk_tprs = []
sk_aucs = []
sk_precisions = []
sk_recalls = []
sk_average_precisions = []
sk_fpr = []
test_label_score = {}



#############################################    MY   begin   ############################################################

model = TwoBranchGNN(input_dim, output_dim, alpha=0.5)  # 正常情况
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.8, last_epoch=-1)

prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))



fold_data = []
for cl, (train_index, test_index) in enumerate(kf.split(samples)):  # 循环每一折的训练和测试集
    model.train()
    print('############ {} fold #############'.format(cl))  # 打印当前折数
    out.append([train_index, test_index])  # 将训练和测试集索引存入列表中
    iter_ = iter_ + 1  # 迭代次数加1

    train_samples = samples[train_index, :]  # 获取当前折的训练集样本
    test_samples = samples[test_index, :]  # 获取当前折的测试集样本
    mic_len = sim_m.shape[1]  # 计算微生物潜在表示的向量长度
    dis_len = sim_d.shape[1]  # 计算疾病潜在表示的向量长度

    train_n = train_samples.shape[0]  # 获取训练集样本数量
    test_N = test_samples.shape[0]  # 获取测试集样本数量

    mic_feature = np.zeros([mic_len, input_dim])
    dis_feature = np.zeros([dis_len, input_dim])
    mic_feature = np.concatenate([sim_m, A], axis=1)
    dis_feature = np.concatenate([sim_d, A.T], axis=1)

    m_and_d_feature = np.concatenate([mic_feature, dis_feature], axis=0)
    m_and_d_feature = torch.tensor(m_and_d_feature, dtype=torch.float32).to(device)

    test_list = []
    train_list = []

    i_list = []
    j_list = []
    i_hat_list = []
    j_hat_list = []

    i_test_list = []
    j_test_list = []
    i_test_hat_list = []
    j_test_hat_list = []

    num = 0

    for sample in train_samples:  # epoch ==1000
        # 正相关[i, j]
        i, j, i_hat, j_hat = map(int, sample)
        train_list.append([i, j, 1])
        train_list.append([i, j, 1])
        train_list.append([i, j_hat, 0])
        train_list.append([i_hat, j, 0])

        i_list.append(i)
        j_list.append(j)
        i_hat_list.append(i_hat)
        j_hat_list.append(j_hat)

    for sample in test_samples:
        # 正相关[i, j]
        i, j, i_hat, j_hat = map(int, sample)
        test_list.append([i, j, 1])
        # test_list.append([i, j, 1])
        # 负相关[i, j_hat]
        test_list.append([i, j_hat, 0])
        test_list.append([i_hat, j, 0])

        i_test_list.append(i)
        j_test_list.append(j)
        i_test_hat_list.append(i_hat)
        j_test_hat_list.append(j_hat)

    train_list = np.array(train_list)
    test_list = np.array(test_list)
    train_list_tensor = torch.tensor(train_list, dtype=torch.float32).to(device)
    test_list_tensor = torch.tensor(test_list, dtype=torch.float32).to(device)

    A_tensor = torch.tensor(A, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = 0

        output = model(m_and_d_feature, adj_A_pos, adj_A_neg)

        mic_feature_out = output[0:mic_feature.shape[0], :]
        dis_feature_out = output[mic_feature.shape[0]:output.shape[0], :]
        print("Output:", output.shape)
        print("mic_feature:", mic_feature_out.shape)
        print("dis_feature:", dis_feature_out.shape)
        prob_matrix = torch.mm(mic_feature_out, dis_feature_out.T)
        prob_matrix = (prob_matrix - prob_matrix.min()) / (prob_matrix.max() - prob_matrix.min())
        print("prob_matrix:", prob_matrix.shape)
        print("A:", A.shape)

        train_labels = train_list_tensor[:, 2]  # 实际标签
        indices = train_list_tensor[:, :2].long()  # 确保索引为整数类型
        train_label = prob_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值

        loss_l2 = lambda_l2 * torch.norm(prob_matrix, p='fro')
        # 现在 train_label 和 train_labels 都是张量，并且可以在计算损失时保持梯度追踪
        matrix_diff_loss = torch.mean((prob_matrix - A_tensor) ** 2)

        i_select = select_rows(mic_feature_out, i_list).to(device)
        i_hat_select = select_rows(mic_feature_out, i_hat_list).to(device)

        j_select = select_rows(dis_feature_out, j_list).to(device)
        j_hat_select = select_rows(dis_feature_out, j_hat_list).to(device)
        constrate_loss = constrate_loss_calculate(i_select, i_hat_select, j_select, j_hat_select)

        loss = lambda_mse * criterion(train_label, train_labels) + loss_l2 + lambda_constrate * constrate_loss

        print("Output:", output)
        if (epoch % 500) == 0:
            print('loss=', loss)
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():

        output = model(m_and_d_feature, adj_A_pos, adj_A_neg)

        mic_feature_out = output[0:mic_feature.shape[0], :]
        dis_feature_out = output[mic_feature.shape[0]:output.shape[0], :]
        print("Output:", output.shape)
        print("mic_feature:", mic_feature_out.shape)
        print("dis_feature:", dis_feature_out.shape)
        prob_matrix = torch.mm(mic_feature_out, dis_feature_out.T)
        prob_matrix = (prob_matrix - prob_matrix.min()) / (prob_matrix.max() - prob_matrix.min())
        print("prob_matrix:", prob_matrix.shape)
        print("A:", A.shape)

        prob_matrix_np = prob_matrix.cpu().numpy()  # 如果你已经确保模型和数据都在 CPU 上，可以省略 .cpu() 调用
        prob_matrix_avg += prob_matrix_np
        result = []
        # for i, j, i_hat, j_hat in train_samples_tensor:
        unique_test_list_tensor = torch.unique(test_list_tensor, dim=0)
        # test_labels = unique_test_list_tensor[:, 2]  # 实际标签
        test_labels = unique_test_list_tensor[:, 2].cpu().numpy()  # 实际标签
        indices = unique_test_list_tensor[:, :2].long()  # 确保索引为整数类型
        perdcit_score = prob_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值
        perdcit_score = perdcit_score.cpu().numpy()
        perdcit_label = [1 if prob >= 0.5 else 0 for prob in perdcit_score]

        viz = metrics.RocCurveDisplay.from_predictions(test_labels, perdcit_score,
                                                       name='ROC fold {}'.format(cl),
                                                       color=colors[cl],
                                                       alpha=0.6, lw=2, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        metrics_summary['f1_scores'].append(f1_score(test_labels, perdcit_label))
        metrics_summary['accuracies'].append(accuracy_score(test_labels, perdcit_label))
        metrics_summary['recalls'].append(recall_score(test_labels, perdcit_label))
        metrics_summary['precisions'].append(precision_score(test_labels, perdcit_label))

        tn, fp, fn, tp = confusion_matrix(test_labels, perdcit_label).ravel()
        specificity = tn / (tn + fp)
        metrics_summary['specificities'].append(specificity)

        fpr_temp, tpr_temp, _ = roc_curve(test_labels, perdcit_score)
        roc_auc = auc(fpr_temp, tpr_temp)
        sk_fpr.append(fpr_temp)
        sk_tprs.append(tpr_temp)
        sk_aucs.append(roc_auc)

        precision_temp, recall_temp, _ = precision_recall_curve(test_labels, perdcit_score)
        average_precision = average_precision_score(test_labels, perdcit_score)
        sk_precisions.append(precision_temp)
        sk_recalls.append(recall_temp)
        sk_average_precisions.append(average_precision)

        test_label_score[cl] = [test_labels, perdcit_score]

prob_matrix_avg = prob_matrix_avg / k_split

mean_fpr = np.linspace(0, 1, 100)
mean_recall = np.linspace(0, 1, 100)
tprs = []
precisions = []
for fpr_temp, tpr_temp in zip(sk_fpr, sk_tprs):
    interp_tpr = np.interp(mean_fpr, fpr_temp, tpr_temp)
    interp_tpr[0] = 0.0  # 确保曲线从 0 开始
    tprs.append(interp_tpr)
mean_tpr = np.mean(tprs, axis=0)

mean_tpr[-1] = 1.0  # 确保曲线以 1 结束

for recall_temp, precision_temp in zip(sk_recalls, sk_precisions):
    interp_precision = np.interp(mean_recall, recall_temp[::-1], precision_temp[::-1])
    precisions.append(interp_precision)

mean_precision = np.mean(precisions, axis=0)

sk_aucs = np.mean(sk_aucs)
sk_average_precisions = np.mean(sk_average_precisions)
mean_fpr = np.linspace(0, 1, 100)
mean_recall = np.linspace(0, 1, 100)
tprs = []
precisions = []
for fpr_temp, tpr_temp in zip(sk_fpr, sk_tprs):
    interp_tpr = np.interp(mean_fpr, fpr_temp, tpr_temp)
    interp_tpr[0] = 0.0  # 确保曲线从 0 开始
    tprs.append(interp_tpr)
mean_tpr = np.mean(tprs, axis=0)

mean_tpr[-1] = 1.0  # 确保曲线以 1 结束

for recall_temp, precision_temp in zip(sk_recalls, sk_precisions):
    interp_precision = np.interp(mean_recall, recall_temp[::-1], precision_temp[::-1])
    precisions.append(interp_precision)

mean_precision = np.mean(precisions, axis=0)

sk_aucs = np.mean(sk_aucs)
sk_average_precisions= np.mean(sk_average_precisions)

np.savetxt('./result/HMDAD_prob_matrix_avg.csv', prob_matrix_avg, delimiter='\t',
           fmt='%0.5f')  # HMDAD peryton Disbiome
# 删除模型变量
del model
torch.cuda.empty_cache()  # 如果使用 GPU，清除显存缓存
gc.collect()  # 手动调用垃圾回收器


#############################################    MY  end   ############################################################


A = pd.read_csv('./dataset/disbiome/process_data/adj_matrix.csv')
disease_chemical = pd.read_csv('./dataset/disbiome/化学-疾病/complete_disease_similarity_matrix.csv')
disease_gene = pd.read_csv('./dataset/disbiome/基因-疾病/complete_disease_similarity_matrix.csv')
disease_symptoms = pd.read_csv('./dataset/disbiome/疾病-症状/complete_disease_similarity_matrix.csv')
disease_Semantics = pd.read_csv('./dataset/disbiome/疾病-语义/similarity_matrix_model2.csv', header=None)
disease_pathway = pd.read_csv('./dataset/disbiome/疾病-通路/complete_disease_similarity_matrix.csv')
micro_cos = pd.read_csv('./dataset/disbiome/基于关联矩阵的微生物功能/Cosine_Sim.csv')
micro_gip = pd.read_csv('./dataset/disbiome/基于关联矩阵的微生物功能/GIP_Sim.csv')
micro_sem = pd.read_csv('./dataset/disbiome/基于疾病语义的微生物功能/functional_similarity2_matrix.csv')
micro_fun1 = pd.read_csv('./dataset/disbiome/微生物-功能/complete_microbe_associations_ds2_matrix.csv')
micro_fun2 = pd.read_csv('./dataset/disbiome/微生物-功能/complete_microbe_similarities_ds2_matrix.csv')
A = A.iloc[:, 1:]
disease_chemical = disease_chemical.iloc[:, 1:]
disease_gene = disease_gene.iloc[:, 1:]
disease_symptoms = disease_symptoms.iloc[:, 1:]
disease_pathway = disease_pathway.iloc[:, 1:]
micro_cos = micro_cos.iloc[:, 1:]
micro_gip = micro_gip.iloc[:, 1:]
micro_fun1 = micro_fun1.iloc[:, 1:]
micro_fun2 = micro_fun2.iloc[:, 1:]

microbiome_matrices = [micro_cos.values, micro_gip.values, micro_sem.values, micro_fun1.values, micro_fun2.values]

disease_matrices = [disease_chemical.values, disease_gene.values, disease_symptoms.values, disease_Semantics.values,
                    disease_pathway.values]

sim_d, sim_m = calculate_combined_similarity(disease_matrices, microbiome_matrices)

input_dim = sim_m.shape[0] + sim_d.shape[0]
A = A.T.to_numpy()

x, y = A.shape
score_matrix = np.zeros([x, y])  # 初始化评分矩阵
md = A
mm = sim_m
dd = sim_d

deep_A = calculate_metapath_optimized(mm, dd, md, n)

samples, A_neg = get_all_pairs(A, deep_A)  # 返回[i, j, i_hat, j_hat] i 微生物 j疾病
samples = np.array(samples)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()

# cross validation
kf = KFold(n_splits=k_split, shuffle=True, random_state=123)  # 定义10折交叉验证
iter_ = 0  # control each iterator  控制迭代次数
sum_score = 0  # 总分数初始化为0
out = []  # 用于存储每一折的训练集和测试集索引
test_label_score = {}  # 存储测试标签和预测得分的字典
criterion = torch.nn.MSELoss()

prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))
iter_ = 0
out = []  # 用于存储每一折的训练集和测试集索引
test_label_score = {}  # 存储测试标签和预测得分的字典

adj_A_pos = build_pos_adjacency_matrix(A)
adj_A_neg = build_perturbed_adjacency_matrix_with_antidiagonal(A_neg)
lambda_l2 = (lambda_l2 * 39 * 292) / (A.shape[0] * A.shape[1])


prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))
iter_ = 0
out = []                    # 用于存储每一折的训练集和测试集索引
test_label_score = {}       # 存储测试标签和预测得分的字典


#############################################    Only_pos  begin   ############################################################

model = TwoBranchGNN(input_dim, output_dim, alpha=0.5)  # 正常情况
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.8, last_epoch=-1)

prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))



fold_data = []
for cl, (train_index, test_index) in enumerate(kf.split(samples)):  # 循环每一折的训练和测试集
    model.train()
    print('############ {} fold #############'.format(cl))  # 打印当前折数
    out.append([train_index, test_index])  # 将训练和测试集索引存入列表中
    iter_ = iter_ + 1  # 迭代次数加1

    train_samples = samples[train_index, :]  # 获取当前折的训练集样本
    test_samples = samples[test_index, :]  # 获取当前折的测试集样本
    mic_len = sim_m.shape[1]  # 计算微生物潜在表示的向量长度
    dis_len = sim_d.shape[1]  # 计算疾病潜在表示的向量长度

    train_n = train_samples.shape[0]  # 获取训练集样本数量
    test_N = test_samples.shape[0]  # 获取测试集样本数量

    mic_feature = np.zeros([mic_len, input_dim])
    dis_feature = np.zeros([dis_len, input_dim])
    mic_feature = np.concatenate([sim_m, A], axis=1)
    dis_feature = np.concatenate([sim_d, A.T], axis=1)

    m_and_d_feature = np.concatenate([mic_feature, dis_feature], axis=0)
    m_and_d_feature = torch.tensor(m_and_d_feature, dtype=torch.float32).to(device)

    test_list = []
    train_list = []

    i_list = []
    j_list = []
    i_hat_list = []
    j_hat_list = []

    i_test_list = []
    j_test_list = []
    i_test_hat_list = []
    j_test_hat_list = []

    num = 0

    for sample in train_samples:  # epoch ==1000
        # 正相关[i, j]
        i, j, i_hat, j_hat = map(int, sample)
        train_list.append([i, j, 1])
        train_list.append([i, j, 1])
        train_list.append([i, j_hat, 0])
        train_list.append([i_hat, j, 0])

        i_list.append(i)
        j_list.append(j)
        i_hat_list.append(i_hat)
        j_hat_list.append(j_hat)

    for sample in test_samples:
        # 正相关[i, j]
        i, j, i_hat, j_hat = map(int, sample)
        test_list.append([i, j, 1])
        # test_list.append([i, j, 1])
        # 负相关[i, j_hat]
        test_list.append([i, j_hat, 0])
        test_list.append([i_hat, j, 0])

        i_test_list.append(i)
        j_test_list.append(j)
        i_test_hat_list.append(i_hat)
        j_test_hat_list.append(j_hat)

    train_list = np.array(train_list)
    test_list = np.array(test_list)
    train_list_tensor = torch.tensor(train_list, dtype=torch.float32).to(device)
    test_list_tensor = torch.tensor(test_list, dtype=torch.float32).to(device)

    A_tensor = torch.tensor(A, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = 0

        output = model(m_and_d_feature, adj_A_pos, adj_A_neg)

        mic_feature_out = output[0:mic_feature.shape[0], :]
        dis_feature_out = output[mic_feature.shape[0]:output.shape[0], :]
        print("Output:", output.shape)
        print("mic_feature:", mic_feature_out.shape)
        print("dis_feature:", dis_feature_out.shape)
        prob_matrix = torch.mm(mic_feature_out, dis_feature_out.T)
        prob_matrix = (prob_matrix - prob_matrix.min()) / (prob_matrix.max() - prob_matrix.min())
        print("prob_matrix:", prob_matrix.shape)
        print("A:", A.shape)

        train_labels = train_list_tensor[:, 2]  # 实际标签
        indices = train_list_tensor[:, :2].long()  # 确保索引为整数类型
        train_label = prob_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值

        loss_l2 = lambda_l2 * torch.norm(prob_matrix, p='fro')
        # 现在 train_label 和 train_labels 都是张量，并且可以在计算损失时保持梯度追踪
        matrix_diff_loss = torch.mean((prob_matrix - A_tensor) ** 2)

        i_select = select_rows(mic_feature_out, i_list).to(device)
        i_hat_select = select_rows(mic_feature_out, i_hat_list).to(device)

        j_select = select_rows(dis_feature_out, j_list).to(device)
        j_hat_select = select_rows(dis_feature_out, j_hat_list).to(device)
        constrate_loss = constrate_loss_calculate(i_select, i_hat_select, j_select, j_hat_select)

        loss = lambda_mse * criterion(train_label, train_labels) + loss_l2 + lambda_constrate * constrate_loss

        print("Output:", output)
        if (epoch % 500) == 0:
            print('loss=', loss)
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():

        output = model(m_and_d_feature, adj_A_pos, adj_A_neg)

        mic_feature_out = output[0:mic_feature.shape[0], :]
        dis_feature_out = output[mic_feature.shape[0]:output.shape[0], :]
        print("Output:", output.shape)
        print("mic_feature:", mic_feature_out.shape)
        print("dis_feature:", dis_feature_out.shape)
        prob_matrix = torch.mm(mic_feature_out, dis_feature_out.T)
        prob_matrix = (prob_matrix - prob_matrix.min()) / (prob_matrix.max() - prob_matrix.min())
        print("prob_matrix:", prob_matrix.shape)
        print("A:", A.shape)

        prob_matrix_np = prob_matrix.cpu().numpy()  # 如果你已经确保模型和数据都在 CPU 上，可以省略 .cpu() 调用
        prob_matrix_avg += prob_matrix_np
        result = []
        # for i, j, i_hat, j_hat in train_samples_tensor:
        unique_test_list_tensor = torch.unique(test_list_tensor, dim=0)
        # test_labels = unique_test_list_tensor[:, 2]  # 实际标签
        test_labels = unique_test_list_tensor[:, 2].cpu().numpy()  # 实际标签
        indices = unique_test_list_tensor[:, :2].long()  # 确保索引为整数类型
        perdcit_score = prob_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值
        perdcit_score = perdcit_score.cpu().numpy()
        perdcit_label = [1 if prob >= 0.5 else 0 for prob in perdcit_score]

        viz = metrics.RocCurveDisplay.from_predictions(test_labels, perdcit_score,
                                                       name='ROC fold {}'.format(cl),
                                                       color=colors[cl],
                                                       alpha=0.6, lw=2, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        metrics_summary['f1_scores'].append(f1_score(test_labels, perdcit_label))
        metrics_summary['accuracies'].append(accuracy_score(test_labels, perdcit_label))
        metrics_summary['recalls'].append(recall_score(test_labels, perdcit_label))
        metrics_summary['precisions'].append(precision_score(test_labels, perdcit_label))

        tn, fp, fn, tp = confusion_matrix(test_labels, perdcit_label).ravel()
        specificity = tn / (tn + fp)
        metrics_summary['specificities'].append(specificity)

        fpr_temp, tpr_temp, _ = roc_curve(test_labels, perdcit_score)
        roc_auc = auc(fpr_temp, tpr_temp)
        sk_fpr_Only_pos.append(fpr_temp)
        sk_tprs_Only_pos.append(tpr_temp)
        sk_aucs_Only_pos.append(roc_auc)

        precision_temp, recall_temp, _ = precision_recall_curve(test_labels, perdcit_score)
        average_precision = average_precision_score(test_labels, perdcit_score)
        sk_precisions_Only_pos.append(precision_temp)
        sk_recalls_Only_pos.append(recall_temp)
        sk_average_precisions_Only_pos.append(average_precision)

        test_label_score[cl] = [test_labels, perdcit_score]

prob_matrix_avg = prob_matrix_avg / k_split


mean_fpr_Only_pos = np.linspace(0, 1, 100)
mean_recall_Only_pos = np.linspace(0, 1, 100)
tprs = []
precisions = []
for fpr_temp, tpr_temp in zip(sk_fpr_Only_pos, sk_tprs_Only_pos):
    interp_tpr = np.interp(mean_fpr_Only_pos, fpr_temp, tpr_temp)
    interp_tpr[0] = 0.0  # 确保曲线从 0 开始
    tprs.append(interp_tpr)
mean_tpr_Only_pos = np.mean(tprs, axis=0)

mean_tpr_Only_pos[-1] = 1.0  # 确保曲线以 1 结束

for recall_temp, precision_temp in zip(sk_recalls_Only_pos, sk_precisions_Only_pos):
    interp_precision = np.interp(mean_recall_Only_pos, recall_temp[::-1], precision_temp[::-1])
    precisions.append(interp_precision)

mean_precision_Only_pos = np.mean(precisions, axis=0)

sk_aucs_Only_pos = np.mean(sk_aucs_Only_pos)
sk_average_precisions_Only_pos = np.mean(sk_average_precisions_Only_pos)
# 删除模型变量
del model
torch.cuda.empty_cache()  # 如果使用 GPU，清除显存缓存
gc.collect()  # 手动调用垃圾回收器
np.savetxt('./result/Disbiome_prob_matrix_avg.csv', prob_matrix_avg, delimiter='\t',
           fmt='%0.5f')  # HMDAD peryton Disbiome

#############################################    Only_pos  end   ############################################################

A = pd.read_csv('./dataset/peryton/adjacency_matrix.csv')
disease_chemical = pd.read_csv('./dataset/peryton/化学-疾病/complete_disease_similarity_matrix.csv')
disease_gene = pd.read_csv('./dataset/peryton/基因-疾病/complete_disease_similarity_matrix.csv')
disease_symptoms = pd.read_csv('./dataset/peryton/疾病-症状/complete_disease_similarity_matrix.csv')
disease_Semantics = pd.read_csv('./dataset/peryton/疾病-语义/similarity_matrix_model2.csv', header=None)
disease_pathway = pd.read_csv('./dataset/peryton/疾病-通路/complete_disease_similarity_matrix.csv')
micro_cos = pd.read_csv('./dataset/peryton/基于关联矩阵的微生物功能/Cosine_Sim.csv')
micro_gip = pd.read_csv('./dataset/peryton/基于关联矩阵的微生物功能/GIP_Sim.csv')
micro_sem = pd.read_csv('./dataset/peryton/基于疾病语义的微生物功能/functional_similarity2_matrix.csv')
micro_fun1 = pd.read_csv('./dataset/peryton/微生物-功能/complete_microbe_associations_ds2_matrix.csv')
micro_fun2 = pd.read_csv('./dataset/peryton/微生物-功能/complete_microbe_similarities_ds2_matrix.csv')
A = A.iloc[:, 1:]

disease_chemical = disease_chemical.iloc[:, 1:]
disease_gene = disease_gene.iloc[:, 1:]
disease_symptoms = disease_symptoms.iloc[:, 1:]
disease_pathway = disease_pathway.iloc[:, 1:]
micro_cos = micro_cos.iloc[:, 1:]
micro_gip = micro_gip.iloc[:, 1:]
micro_fun1 = micro_fun1.iloc[:, 1:]
micro_fun2 = micro_fun2.iloc[:, 1:]

microbiome_matrices = [micro_cos.values, micro_gip.values, micro_sem.values, micro_fun1.values, micro_fun2.values]

disease_matrices = [disease_chemical.values, disease_gene.values, disease_symptoms.values, disease_Semantics.values,
                    disease_pathway.values]

sim_d, sim_m = calculate_combined_similarity(disease_matrices, microbiome_matrices)

input_dim = sim_m.shape[0] + sim_d.shape[0]
A = A.T.to_numpy()

x, y = A.shape
score_matrix = np.zeros([x, y])  # 初始化评分矩阵
md = A
mm = sim_m
dd = sim_d

deep_A = calculate_metapath_optimized(mm, dd, md, n)

samples, A_neg = get_all_pairs(A, deep_A)  # 返回[i, j, i_hat, j_hat] i 微生物 j疾病
samples = np.array(samples)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()

# cross validation
kf = KFold(n_splits=k_split, shuffle=True, random_state=123)  # 定义10折交叉验证
iter_ = 0  # control each iterator  控制迭代次数
sum_score = 0  # 总分数初始化为0
out = []  # 用于存储每一折的训练集和测试集索引
test_label_score = {}  # 存储测试标签和预测得分的字典
criterion = torch.nn.MSELoss()

prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))
iter_ = 0
out = []  # 用于存储每一折的训练集和测试集索引
test_label_score = {}  # 存储测试标签和预测得分的字典

adj_A_pos = build_pos_adjacency_matrix(A)
adj_A_neg = build_perturbed_adjacency_matrix_with_antidiagonal(A_neg)
lambda_l2 = (lambda_l2 * 39 * 292) / (A.shape[0] * A.shape[1])


prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))
iter_ = 0
out = []                    # 用于存储每一折的训练集和测试集索引
test_label_score = {}       # 存储测试标签和预测得分的字典

#############################################    Only_neg  begin   ############################################################
model = TwoBranchGNN(input_dim, output_dim, alpha=0.5)  # 正常情况
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.8, last_epoch=-1)

prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))



fold_data = []
for cl, (train_index, test_index) in enumerate(kf.split(samples)):  # 循环每一折的训练和测试集
    model.train()
    print('############ {} fold #############'.format(cl))  # 打印当前折数
    out.append([train_index, test_index])  # 将训练和测试集索引存入列表中
    iter_ = iter_ + 1  # 迭代次数加1

    train_samples = samples[train_index, :]  # 获取当前折的训练集样本
    test_samples = samples[test_index, :]  # 获取当前折的测试集样本
    mic_len = sim_m.shape[1]  # 计算微生物潜在表示的向量长度
    dis_len = sim_d.shape[1]  # 计算疾病潜在表示的向量长度

    train_n = train_samples.shape[0]  # 获取训练集样本数量
    test_N = test_samples.shape[0]  # 获取测试集样本数量

    mic_feature = np.zeros([mic_len, input_dim])
    dis_feature = np.zeros([dis_len, input_dim])
    mic_feature = np.concatenate([sim_m, A], axis=1)
    dis_feature = np.concatenate([sim_d, A.T], axis=1)

    m_and_d_feature = np.concatenate([mic_feature, dis_feature], axis=0)
    m_and_d_feature = torch.tensor(m_and_d_feature, dtype=torch.float32).to(device)

    test_list = []
    train_list = []

    i_list = []
    j_list = []
    i_hat_list = []
    j_hat_list = []

    i_test_list = []
    j_test_list = []
    i_test_hat_list = []
    j_test_hat_list = []

    num = 0

    for sample in train_samples:  # epoch ==1000
        # 正相关[i, j]
        i, j, i_hat, j_hat = map(int, sample)
        train_list.append([i, j, 1])
        train_list.append([i, j, 1])
        train_list.append([i, j_hat, 0])
        train_list.append([i_hat, j, 0])

        i_list.append(i)
        j_list.append(j)
        i_hat_list.append(i_hat)
        j_hat_list.append(j_hat)

    for sample in test_samples:
        # 正相关[i, j]
        i, j, i_hat, j_hat = map(int, sample)
        test_list.append([i, j, 1])
        # test_list.append([i, j, 1])
        # 负相关[i, j_hat]
        test_list.append([i, j_hat, 0])
        test_list.append([i_hat, j, 0])

        i_test_list.append(i)
        j_test_list.append(j)
        i_test_hat_list.append(i_hat)
        j_test_hat_list.append(j_hat)

    train_list = np.array(train_list)
    test_list = np.array(test_list)
    train_list_tensor = torch.tensor(train_list, dtype=torch.float32).to(device)
    test_list_tensor = torch.tensor(test_list, dtype=torch.float32).to(device)

    A_tensor = torch.tensor(A, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = 0

        output = model(m_and_d_feature, adj_A_pos, adj_A_neg)

        mic_feature_out = output[0:mic_feature.shape[0], :]
        dis_feature_out = output[mic_feature.shape[0]:output.shape[0], :]
        print("Output:", output.shape)
        print("mic_feature:", mic_feature_out.shape)
        print("dis_feature:", dis_feature_out.shape)
        prob_matrix = torch.mm(mic_feature_out, dis_feature_out.T)
        prob_matrix = (prob_matrix - prob_matrix.min()) / (prob_matrix.max() - prob_matrix.min())
        print("prob_matrix:", prob_matrix.shape)
        print("A:", A.shape)

        train_labels = train_list_tensor[:, 2]  # 实际标签
        indices = train_list_tensor[:, :2].long()  # 确保索引为整数类型
        train_label = prob_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值

        loss_l2 = lambda_l2 * torch.norm(prob_matrix, p='fro')
        # 现在 train_label 和 train_labels 都是张量，并且可以在计算损失时保持梯度追踪
        matrix_diff_loss = torch.mean((prob_matrix - A_tensor) ** 2)

        i_select = select_rows(mic_feature_out, i_list).to(device)
        i_hat_select = select_rows(mic_feature_out, i_hat_list).to(device)

        j_select = select_rows(dis_feature_out, j_list).to(device)
        j_hat_select = select_rows(dis_feature_out, j_hat_list).to(device)
        constrate_loss = constrate_loss_calculate(i_select, i_hat_select, j_select, j_hat_select)

        loss = lambda_mse * criterion(train_label, train_labels) + loss_l2 + lambda_constrate * constrate_loss

        print("Output:", output)
        if (epoch % 500) == 0:
            print('loss=', loss)
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():

        output = model(m_and_d_feature, adj_A_pos, adj_A_neg)

        mic_feature_out = output[0:mic_feature.shape[0], :]
        dis_feature_out = output[mic_feature.shape[0]:output.shape[0], :]
        print("Output:", output.shape)
        print("mic_feature:", mic_feature_out.shape)
        print("dis_feature:", dis_feature_out.shape)
        prob_matrix = torch.mm(mic_feature_out, dis_feature_out.T)
        prob_matrix = (prob_matrix - prob_matrix.min()) / (prob_matrix.max() - prob_matrix.min())
        print("prob_matrix:", prob_matrix.shape)
        print("A:", A.shape)

        prob_matrix_np = prob_matrix.cpu().numpy()  # 如果你已经确保模型和数据都在 CPU 上，可以省略 .cpu() 调用
        prob_matrix_avg += prob_matrix_np
        result = []
        # for i, j, i_hat, j_hat in train_samples_tensor:
        unique_test_list_tensor = torch.unique(test_list_tensor, dim=0)
        # test_labels = unique_test_list_tensor[:, 2]  # 实际标签
        test_labels = unique_test_list_tensor[:, 2].cpu().numpy()  # 实际标签
        indices = unique_test_list_tensor[:, :2].long()  # 确保索引为整数类型
        perdcit_score = prob_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值
        perdcit_score = perdcit_score.cpu().numpy()
        perdcit_label = [1 if prob >= 0.5 else 0 for prob in perdcit_score]

        viz = metrics.RocCurveDisplay.from_predictions(test_labels, perdcit_score,
                                                       name='ROC fold {}'.format(cl),
                                                       color=colors[cl],
                                                       alpha=0.6, lw=2, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        metrics_summary['f1_scores'].append(f1_score(test_labels, perdcit_label))
        metrics_summary['accuracies'].append(accuracy_score(test_labels, perdcit_label))
        metrics_summary['recalls'].append(recall_score(test_labels, perdcit_label))
        metrics_summary['precisions'].append(precision_score(test_labels, perdcit_label))

        tn, fp, fn, tp = confusion_matrix(test_labels, perdcit_label).ravel()
        specificity = tn / (tn + fp)
        metrics_summary['specificities'].append(specificity)

        fpr_temp, tpr_temp, _ = roc_curve(test_labels, perdcit_score)
        roc_auc = auc(fpr_temp, tpr_temp)
        sk_fpr_Only_neg.append(fpr_temp)
        sk_tprs_Only_neg.append(tpr_temp)
        sk_aucs_Only_neg.append(roc_auc)

        precision_temp, recall_temp, _ = precision_recall_curve(test_labels, perdcit_score)
        average_precision = average_precision_score(test_labels, perdcit_score)
        sk_precisions_Only_neg.append(precision_temp)
        sk_recalls_Only_neg.append(recall_temp)
        sk_average_precisions_Only_neg.append(average_precision)

        test_label_score[cl] = [test_labels, perdcit_score]

prob_matrix_avg = prob_matrix_avg / k_split


mean_fpr_Only_neg = np.linspace(0, 1, 100)
mean_recall_Only_neg = np.linspace(0, 1, 100)
tprs = []
precisions = []
for fpr_temp, tpr_temp in zip(sk_fpr_Only_neg, sk_tprs_Only_neg):
    interp_tpr = np.interp(mean_fpr_Only_neg, fpr_temp, tpr_temp)
    interp_tpr[0] = 0.0  # 确保曲线从 0 开始
    tprs.append(interp_tpr)
mean_tpr_Only_neg = np.mean(tprs, axis=0)

mean_tpr_Only_neg[-1] = 1.0  # 确保曲线以 1 结束

for recall_temp, precision_temp in zip(sk_recalls_Only_neg, sk_precisions_Only_neg):
    interp_precision = np.interp(mean_recall_Only_neg, recall_temp[::-1], precision_temp[::-1])
    precisions.append(interp_precision)

mean_precision_Only_neg = np.mean(precisions, axis=0)

sk_aucs_Only_neg = np.mean(sk_aucs_Only_neg)
sk_average_precisions_Only_neg = np.mean(sk_average_precisions_Only_neg)

np.savetxt('./result/peryton_prob_matrix_avg.csv', prob_matrix_avg, delimiter='\t',
           fmt='%0.5f')  # HMDAD peryton Disbiome
# 删除模型变量
del model
torch.cuda.empty_cache()  # 如果使用 GPU，清除显存缓存
gc.collect()  # 手动调用垃圾回收器

#############################################    Only_neg  end   ############################################################

prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))
iter_ = 0
out = []                    # 用于存储每一折的训练集和测试集索引
test_label_score = {}       # 存储测试标签和预测得分的字典


def compute_mean(values):
    return np.mean(values, axis=0)




import json
import matplotlib.pyplot as plt

# 数据定义（假设这些变量已经正确计算）
model_labels = ['HMDAD', 'Disbiome', 'Peryton']

model_precisions = [mean_precision, mean_precision_Only_pos, mean_precision_Only_neg]
model_recalls = [mean_recall, mean_recall_Only_pos, mean_recall_Only_neg]
model_auprs = [sk_average_precisions, sk_average_precisions_Only_pos, sk_average_precisions_Only_neg]

model_tprs = [mean_tpr, mean_tpr_Only_pos, mean_tpr_Only_neg]
model_fprs = [mean_fpr, mean_fpr_Only_pos, mean_fpr_Only_neg]
model_aucs = [sk_aucs, sk_aucs_Only_pos, sk_aucs_Only_neg]

# 将 numpy 数据转为 list
def convert_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_list(i) for i in obj]
    else:
        return obj

# 保存 Precision-Recall 数据
pr_roc_data = {
    "precision_recall": [
        {
            "label": label,
            "precisions": convert_to_list(precisions),
            "recalls": convert_to_list(recalls),
            "aupr": aupr
        }
        for label, precisions, recalls, aupr in zip(model_labels, model_precisions, model_recalls, model_auprs)
    ],
    "roc": [
        {
            "label": label,
            "tprs": convert_to_list(tprs),
            "fprs": convert_to_list(fprs),
            "auc": auc
        }
        for label, tprs, fprs, auc in zip(model_labels, model_tprs, model_fprs, model_aucs)
    ]
}

with open("pr_roc_data.json", "w") as json_file:
    json.dump(pr_roc_data, json_file, indent=4)

# 绘制 Precision-Recall 曲线
fig2, axs2 = plt.subplots(1, 1, figsize=(5, 5))
for precisions, recalls, auprs, label in zip(model_precisions, model_recalls, model_auprs, model_labels):
    axs2.step(recalls, precisions, where='post', label=f'{label} AUPR={auprs:.3f}')

axs2.plot([0, 1], [1, 0], '--', color='r', label='Random')
axs2.set_xlabel('Recall')
axs2.set_ylabel('Precision')
axs2.set_ylim([-0.05, 1.05])
axs2.set_xlim([-0.05, 1.05])
axs2.set_title('Precision-Recall curve')
axs2.legend(loc="best")
plt.show()

# 绘制 ROC 曲线
fig3, axs3 = plt.subplots(1, 1, figsize=(5, 5))
for tprs, fprs, aucs, label in zip(model_tprs, model_fprs, model_aucs, model_labels):
    axs3.step(fprs, tprs, where='post', label=f'{label} AUC={aucs:.3f}')

axs3.plot([0, 1], [0, 1], '--', color='r', label='Random')
axs3.set_xlabel('FPR')
axs3.set_ylabel('TPR')
axs3.set_ylim([-0.05, 1.05])
axs3.set_xlim([-0.05, 1.05])
axs3.set_title('ROC curve')
axs3.legend(loc="best")
plt.show()

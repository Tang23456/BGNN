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


matplotlib.use('TkAgg')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # 使用颜色编码定义颜色
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

set_seed(123)
epochs = 200
k_split = 5
output_dim = 32  # 低维输出维度

n = 4
out = []  # 用于存储每一折的训练集和测试集索引
k_split = 5
set_seed(123)
lambda_mse = 4
lambda_l2 = 5e-2
lambda_constrate = 9e-2




matplotlib.use('TkAgg')
criterion = torch.nn.MSELoss()
kf = KFold(n_splits=k_split, shuffle=True, random_state=123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # 使用颜色编码定义颜色
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

#
# lambda_l2_list=[1e-2,2e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,9e-2,1e-1]
lambda_l2_list=[0,1e-2,2e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,9e-2]
lambda_mse_list=[0,1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 ,9 ]
# lambda_constrate_list=[0,1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 ,9 ]
lambda_constrate_list=[0,1e-2,2e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,9e-2]
# lambda_constrate_list=[1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 ,9 , 10]
n_list=[1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 ,9 , 10]


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

#HMDAD
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

#disbiome
# A = pd.read_csv('./dataset/disbiome/process_data/adj_matrix.csv')
# disease_chemical = pd.read_csv('./dataset/disbiome/化学-疾病/complete_disease_similarity_matrix.csv')
# disease_gene = pd.read_csv('./dataset/disbiome/基因-疾病/complete_disease_similarity_matrix.csv')
# disease_symptoms = pd.read_csv('./dataset/disbiome/疾病-症状/complete_disease_similarity_matrix.csv')
# disease_Semantics = pd.read_csv('./dataset/disbiome/疾病-语义/similarity_matrix_model2.csv', header=None)
# disease_pathway = pd.read_csv('./dataset/disbiome/疾病-通路/complete_disease_similarity_matrix.csv')
# micro_cos = pd.read_csv('./dataset/disbiome/基于关联矩阵的微生物功能/Cosine_Sim.csv')
# micro_gip = pd.read_csv('./dataset/disbiome/基于关联矩阵的微生物功能/GIP_Sim.csv')
# micro_sem = pd.read_csv('./dataset/disbiome/基于疾病语义的微生物功能/functional_similarity2_matrix.csv')
# micro_fun1 = pd.read_csv('./dataset/disbiome/微生物-功能/complete_microbe_associations_ds2_matrix.csv')
# micro_fun2 = pd.read_csv('./dataset/disbiome/微生物-功能/complete_microbe_similarities_ds2_matrix.csv')
# A = A.iloc[:, 1:]


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

#######################################################################
n = 4

lambda_mse_AUC = []
lambda_mse_AUPR = []

lambda_constrate_AUC = []
lambda_constrate_AUPR = []

lambda_l2_AUC = []
lambda_l2_AUPR = []

n_AUC = []
n_AUPR = []

deep_A = calculate_metapath_optimized(mm, dd, md, n)
samples, A_neg = get_all_pairs(A, deep_A)  # 返回[i, j, i_hat, j_hat] i 微生物 j疾病
samples = np.array(samples)
lambda_l2 = (lambda_l2 * 39 * 292) / (A.shape[0] * A.shape[1])

def test( lambda_mse_in, lambda_constrate_in):
    prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))

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
    lambda_mse = lambda_mse_in
    lambda_constrate = lambda_constrate_in

    sk_tprs = []
    sk_aucs = []
    sk_precisions = []
    sk_recalls = []
    sk_average_precisions = []
    sk_fpr = []
    test_label_score = {}

    adj_A_pos = build_pos_adjacency_matrix(A)
    adj_A_neg = build_perturbed_adjacency_matrix_with_antidiagonal(A_neg)

    # 初始化模型
    model = TwoBranchGNN(input_dim, output_dim, alpha=0.6)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.8, last_epoch=-1)

    iter_ = 0
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

            prob_matrix = torch.mm(mic_feature_out, dis_feature_out.T)
            prob_matrix = (prob_matrix - prob_matrix.min()) / (prob_matrix.max() - prob_matrix.min())


            train_labels = train_list_tensor[:, 2]  # 实际标签
            indices = train_list_tensor[:, :2].long()  # 确保索引为整数类型
            train_label = prob_matrix[indices[:, 0], indices[:, 1]]  # 使用张量索引获取预测值

            loss_l2 = lambda_l2 * torch.norm(prob_matrix, p='fro')
            # 现在 train_label 和 train_labels 都是张量，并且可以在计算损失时保持梯度追踪
            matrix_diff_loss = torch.mean((prob_matrix - A_tensor) ** 2)

            i_select = select_rows(mic_feature_out, i_list).to(device)
            j_select = select_rows(dis_feature_out, j_list).to(device)
            i_hat_select = select_rows(mic_feature_out, i_hat_list).to(device)
            j_hat_select = select_rows(dis_feature_out, j_hat_list).to(device)
            constrate_loss = constrate_loss_calculate(i_select, i_hat_select, j_select, j_hat_select)

            loss = lambda_mse * criterion(train_label, train_labels) + loss_l2 + lambda_constrate * constrate_loss

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

            prob_matrix = torch.mm(mic_feature_out, dis_feature_out.T)
            prob_matrix = (prob_matrix - prob_matrix.min()) / (prob_matrix.max() - prob_matrix.min())
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
                                                       alpha=0.6, lw=2, ax=ax)  # 创建ROC曲线显示对象   绘制了每一折的AUC曲线
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)  # 对TPR进行插值
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)  # 将插值后的TPR添加到列表中
        aucs.append(viz.roc_auc)  # 将每一次交叉验证的ROC AUC值添加到aucs列表中。

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

        precision, recall, _ = precision_recall_curve(test_labels, perdcit_score)
        pr_auc = auc(recall, precision)

        # fold_metrics['aucs'].append(roc_auc)
        # fold_metrics['auprs'].append(pr_auc)
        # fold_metrics['f1_scores'].append(f1_score(test_labels, perdcit_label))
        # fold_metrics['accuracies'].append(accuracy_score(test_labels, perdcit_label))

        # 计算Precision-Recall曲线和AUPR
        precision_temp, recall_temp, _ = precision_recall_curve(test_labels, perdcit_score)
        average_precision = average_precision_score(test_labels, perdcit_score)
        sk_precisions.append(precision_temp)
        sk_recalls.append(recall_temp)
        sk_average_precisions.append(average_precision)

        test_label_score[cl] = [test_labels, perdcit_score]  # 将每次测试的标签和预测概率存储到字典中，以便于后续分析。
    print('############ avg score #############')
    for metric, values in metrics_summary.items():
        print(f"{metric}: {np.mean(values):.2f} ± {np.std(values):.2f}")

    print('mean AUC = ', np.mean(sk_aucs))
    print('mean AUPR = ', np.mean(sk_average_precisions))

    AUPR = np.mean(sk_average_precisions)
    AUC = np.mean(sk_aucs)
    del model
    torch.cuda.empty_cache()  # 如果使用 GPU，清除显存缓存
    gc.collect()  # 手动调用垃圾回收器

    return AUPR, AUC


#
# # 初始化用于保存 AUPR 和 AUC 的结果
# results = {}
#
# for n in n_list:
#     for lambda_constrate in lambda_constrate_list:
#         for lambda_mse in lambda_mse_list:
#             # Call the test function with the current parameter values
#             print("n:",n)
#             print("lambda_constrate:", lambda_constrate)
#             print("lambda_mse:", lambda_mse)
#             AUPR, AUC = test(n, lambda_mse, lambda_constrate)
#
#             # Store the results in the dictionary
#             if n not in results:
#                 results[n] = {}
#             if lambda_constrate not in results[n]:
#                 results[n][lambda_constrate] = {}
#             results[n][lambda_constrate][lambda_mse] = {
#                 'AUPR': AUPR,
#                 'AUC': AUC
#             }
#
# # Save the results to a file (e.g., as a JSON file)
# import json
#
# with open('results.json', 'w') as f:
#     json.dump(results, f, indent=4)
#
# # To load the results later
# with open('results.json', 'r') as f:
#     loaded_results = json.load(f)
#
#
#
# ################################# plot  ######################################################
# import json
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.colors as mcolors
#
# # 从 JSON 文件中加载结果数据
# with open('results.json', 'r', encoding='utf-8') as f:
#     results = json.load(f)
#
# # 准备绘图所需的列表
# n_values = []
# lambda_mse_values = []
# lambda_constrate_values = []
# AUC_values = []
# AUPR_values = []
#
# # 遍历结果数据并提取值
# for n, n_data in results.items():
#     for lambda_constrate, lambda_constrate_data in n_data.items():
#         for lambda_mse, metrics in lambda_constrate_data.items():
#             n_values.append(int(n))
#             lambda_mse_values.append(int(lambda_mse))
#             lambda_constrate_values.append(int(lambda_constrate))
#             AUC_values.append(metrics['AUC'])
#             AUPR_values.append(metrics['AUPR'])
#
# # 创建一个具有明确区间的自定义颜色映射
# colors = [(0.0, 'blue'), (0.8, 'yellow'), (0.85, 'red'),
#           (0.9, 'purple'), (0.95, 'darkred'), (1.0, 'black')]
#
# # 通过插值创建更多的颜色区间
# cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
#
# # 创建 AUC 的 3D 散点图
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111, projection='3d')
# scatter1 = ax1.scatter(n_values, lambda_mse_values, lambda_constrate_values, c=AUC_values, cmap=cmap, vmin=0.8, vmax=1.0)
# ax1.set_xlabel('n')
# ax1.set_ylabel('lambda_mse')
# ax1.set_zlabel('lambda_constrate')
# ax1.set_title('3D Visualization of AUC Scores')
# ax1.view_init(elev=30, azim=120)
#
# # 创建并调整颜色条的位置
# cbar1 = plt.colorbar(scatter1, ax=ax1, pad=0.1, aspect=10, shrink=0.7)
# cbar1.set_label('AUC')
# cbar1.ax.tick_params(labelsize=8)  # 调整颜色条刻度的字体大小
#
# # 调整布局以避免标签或标题被截断
# plt.tight_layout()
# plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)  # 微调布局以防颜色条遮挡
#
# # 保存为 PDF 格式
# fig1.savefig('AUC_3D_Visualization.pdf', format='pdf')
#
# # 创建 AUPR 的 3D 散点图
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, projection='3d')
# scatter2 = ax2.scatter(n_values, lambda_mse_values, lambda_constrate_values, c=AUPR_values, cmap=cmap, vmin=0.8, vmax=1.0)
# ax2.set_xlabel('n')
# ax2.set_ylabel('lambda_mse')
# ax2.set_zlabel('lambda_constrate')
# ax2.set_title('3D Visualization of AUPR Scores')
# ax2.view_init(elev=30, azim=120)
#
# # 创建并调整颜色条的位置
# cbar2 = plt.colorbar(scatter2, ax=ax2, pad=0.1, aspect=10, shrink=0.7)
# cbar2.set_label('AUPR')
# cbar2.ax.tick_params(labelsize=8)  # 调整颜色条刻度的字体大小
#
# # 调整布局以避免标签或标题被截断
# plt.tight_layout()
# plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)  # 微调布局以防颜色条遮挡
#
# # 保存为 PDF 格式
# fig2.savefig('AUPR_3D_Visualization.pdf', format='pdf')
#
# # 分别显示图形
# plt.show()



# 初始化用于保存 AUPR 和 AUC 的结果
results = {}


for lambda_constrate in lambda_constrate_list:
    for lambda_mse in lambda_mse_list:
        # Call the test function with the current parameter values
        print("lambda_constrate:", lambda_constrate)
        print("lambda_mse:", lambda_mse)

        AUPR, AUC = test(lambda_mse, lambda_constrate)
        if lambda_constrate not in results:
            results[lambda_constrate] = {}

        results[lambda_constrate][lambda_mse] = {
            'AUPR': AUPR,
            'AUC': AUC
        }

# Save the results to a file (e.g., as a JSON file)
import json

with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)

# To load the results later
with open('results.json', 'r') as f:
    loaded_results = json.load(f)

import json
import matplotlib.pyplot as plt
import numpy as np

# 从 JSON 文件中加载结果数据
with open('results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# 假设lambda_constrate_list 和 lambda_mse_list已经定义
lambda_constrate_list = list(results.keys())
lambda_mse_list = list(results[lambda_constrate_list[0]].keys())

# 初始化AUC和AUPR矩阵
AUC_matrix = np.zeros((len(lambda_constrate_list), len(lambda_mse_list)))
AUPR_matrix = np.zeros((len(lambda_constrate_list), len(lambda_mse_list)))

# 填充矩阵
for i, lambda_constrate in enumerate(lambda_constrate_list):
    for j, lambda_mse in enumerate(lambda_mse_list):
        AUC_matrix[i, j] = results[lambda_constrate][lambda_mse]['AUC']
        AUPR_matrix[i, j] = results[lambda_constrate][lambda_mse]['AUPR']


# # 创建热图
# def plot_heatmap(matrix, xlabel, ylabel, title, save_path):
#     plt.figure(figsize=(8, 6))
#     plt.imshow(matrix, cmap='viridis', aspect='auto', origin='lower')
#     plt.colorbar(label='Score')
#
#     plt.xticks(np.arange(len(lambda_mse_list)), lambda_mse_list, rotation=45)
#     plt.yticks(np.arange(len(lambda_constrate_list)), lambda_constrate_list)
#
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.show()
#
#
# # 绘制 AUC 热图
# plot_heatmap(AUC_matrix, 'lambda_mse', 'lambda_constrate', 'AUC Heatmap', 'AUC_heatmap.png')
#
# # 绘制 AUPR 热图
# plot_heatmap(AUPR_matrix, 'lambda_mse', 'lambda_constrate', 'AUPR Heatmap', 'AUPR_heatmap.png')

def plot_heatmap(matrix, xlabel, ylabel, title, save_path, xticks, yticks):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(label='Score')

    plt.xticks(np.arange(len(xticks)), xticks, rotation=45)
    plt.yticks(np.arange(len(yticks)), yticks)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # 在每个方格上标注数值
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f"{matrix[i, j]:.2f}", ha='center', va='center', color='white', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# 绘制 AUC 热图并标注数值
plot_heatmap(
    AUC_matrix,
    xlabel='lambda_mse',
    ylabel='lambda_constrate',
    title='AUC Heatmap',
    save_path='AUC_heatmap.png',
    xticks=lambda_mse_list,
    yticks=lambda_constrate_list
)

# 绘制 AUPR 热图并标注数值
plot_heatmap(
    AUPR_matrix,
    xlabel='lambda_mse',
    ylabel='lambda_constrate',
    title='AUPR Heatmap',
    save_path='AUPR_heatmap.png',
    xticks=lambda_mse_list,
    yticks=lambda_constrate_list
)
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

#########################################
#
#                   ST
#
#########################################


# 示例用法
if __name__ == "__main__":


    matplotlib.use('TkAgg')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # 使用颜色编码定义颜色
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    set_seed(123)
    epochs = 200
    k_split = 5
    output_dim = 32  # 低维输出维度

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # input_dim = 1396+43  # phendb  1396+43 peryton    1396+43     disbiome 1622+374 疾病特征维度     HMDAD 292+39

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


    #Disbiome 355 1584

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




    #peryton
    # A = pd.read_csv('./dataset/peryton/adjacency_matrix.csv')
    # disease_chemical = pd.read_csv('./dataset/peryton/化学-疾病/complete_disease_similarity_matrix.csv')
    # disease_gene = pd.read_csv('./dataset/peryton/基因-疾病/complete_disease_similarity_matrix.csv')
    # disease_symptoms = pd.read_csv('./dataset/peryton/疾病-症状/complete_disease_similarity_matrix.csv')
    # disease_Semantics = pd.read_csv('./dataset/peryton/疾病-语义/similarity_matrix_model2.csv', header=None)
    # disease_pathway = pd.read_csv('./dataset/peryton/疾病-通路/complete_disease_similarity_matrix.csv')
    # micro_cos = pd.read_csv('./dataset/peryton/基于关联矩阵的微生物功能/Cosine_Sim.csv')
    # micro_gip = pd.read_csv('./dataset/peryton/基于关联矩阵的微生物功能/GIP_Sim.csv')
    # micro_sem = pd.read_csv('./dataset/peryton/基于疾病语义的微生物功能/functional_similarity2_matrix.csv')
    # micro_fun1 = pd.read_csv('./dataset/peryton/微生物-功能/complete_microbe_associations_ds2_matrix.csv')
    # micro_fun2 = pd.read_csv('./dataset/peryton/微生物-功能/complete_microbe_similarities_ds2_matrix.csv')
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

    # A = A.astype(float).astype(int).to_numpy()
    print("the number of miRNAs and diseases", A.shape)
    print("the number of associations", sum(sum(A)))

    x, y = A.shape
    score_matrix = np.zeros([x, y])  # 初始化评分矩阵

    md = A
    mm = sim_m
    dd = sim_d
    n = 4

    deep_A = calculate_metapath_optimized(mm, dd, md, n)
    lambda_mse = 4    #4
    lambda_l2 = 5e-2  # 5e-2
    # 3e-1  图神经网络层数为2的时候中间层为62
    # lambda_constrate = 3  # 2
    lambda_constrate = 9e-2
    lambda_l2 = (lambda_l2 * 39 * 292) / (A.shape[0] * A.shape[1])

    print("the number of microbes and diseases", A.shape)
    print("the number of associations", sum(sum(A)))
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

    prob_matrix_avg = np.zeros((A.shape[0], A.shape[1]))
    iter_ = 0
    out = []  # 用于存储每一折的训练集和测试集索引
    test_label_score = {}  # 存储测试标签和预测得分的字典


    adj_A_pos = build_pos_adjacency_matrix(A)
    adj_A_neg = build_perturbed_adjacency_matrix_with_antidiagonal(A_neg)

    # 初始化模型
    model = TwoBranchGNN(input_dim, output_dim, alpha=0.5)            #正常情况
    # model = TwoBranchGNN_Only_Pos(input_dim, output_dim, alpha=0.5)   #只存在正边
    # model = TwoBranchGNN_Only_Neg(input_dim, output_dim, alpha=0.5)  # 只存在负边
    # model = TwoBranchGNN_Normal_Neg_A(input_dim, output_dim, alpha=0.5)  #其中负边不进行处理
    # model = TwoBranchGNN_Normal_Neg_A_change(input_dim, output_dim, alpha=0.5)  # 其中负边进行部分处理，没有对前一部分的负边进行拟合消除


    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.8, last_epoch=-1)

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

            mic_feature_out = output[0:mic_feature.shape[0],:]
            dis_feature_out = output[mic_feature.shape[0]:output.shape[0],:]
            # print("Output:", output.shape)
            # print("mic_feature:", mic_feature_out.shape)
            # print("dis_feature:", dis_feature_out.shape)
            prob_matrix = torch.mm(mic_feature_out, dis_feature_out.T)
            prob_matrix = (prob_matrix - prob_matrix.min()) / (prob_matrix.max() - prob_matrix.min())
            # print("prob_matrix:",prob_matrix.shape)
            # print("A:", A.shape)

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

            loss = lambda_mse * criterion(train_label,train_labels)  + loss_l2 + lambda_constrate * constrate_loss

            # print("Output:", output)
            if (epoch % 20) == 0:
                print('epoch = ',epoch, 'loss=', loss)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():

            output = model(m_and_d_feature, adj_A_pos, adj_A_neg)

            mic_feature_out = output[0:mic_feature.shape[0], :]
            dis_feature_out = output[mic_feature.shape[0]:output.shape[0], :]
            # print("Output:", output.shape)
            # print("mic_feature:", mic_feature_out.shape)
            # print("dis_feature:", dis_feature_out.shape)
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
            perdcit_label = [1 if prob >= 0.1 else 0 for prob in perdcit_score]

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
        with open(fold_metrics_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([cl + 1, roc_auc, pr_auc, f1_score(test_labels, perdcit_label),
                             accuracy_score(test_labels, perdcit_label)])

prob_matrix_avg = prob_matrix_avg / k_split

np.savetxt('./result/Disbiome_prob_matrix_avg.csv', prob_matrix_avg, delimiter='\t',
           fmt='%0.5f')  # HMDAD peryton Disbiome
print('############ avg score #############')
for metric, values in metrics_summary.items():
    print(f"{metric}: {np.mean(values):.2f} ± {np.std(values):.2f}")

print('mean AUC = ', np.mean(sk_aucs))
print('mean AUPR = ', np.mean(sk_average_precisions))
folds = test_label_score  # 将测试标签和预测概率的数据字典赋值给folds变量，以便于后续使用。

mean_tpr = np.mean(tprs, axis=0)  # 计算所有折次的真正率(TPR)的平均值。
mean_tpr[-1] = 1.0  # 确保曲线的结束点位于(1,1)
mean_auc = metrics.auc(mean_fpr, mean_tpr)  # 计算平均AUC值。
std_auc = np.std(aucs)  # 计算AUC值的标准差。
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

# 计算TPR的标准差，并绘制TPR的置信区间。
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)  # 计算上界。
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)  # 计算下界。

# 左侧绘制ROC曲线
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
ax.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=.8)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.3, label=r'± 1 std. dev.')
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver Operating Characteristic", xlabel='False Positive Rate',
       ylabel='True Positive Rate')
ax.legend(loc="lower right")

fig2, axs2 = plt.subplots(1, 1, figsize=(5, 5))
for i in range(5):
    plt.plot(sk_fpr[i], sk_tprs[i], label=f'Fold {i + 1} AUC = {sk_aucs[i]:.2f}')
axs2.plot([0, 1], [0, 1], 'k--', label='Random', color='r')
axs2.set_xlim([-0.05, 1.05])
axs2.set_ylim([0.0, 1.05])
axs2.set_xlabel('False Positive Rate')
axs2.set_ylabel('True Positive Rate')
axs2.set_title('ROC Curves ')
axs2.legend(loc="lower right")
plt.show()

# 绘制Precision-Recall曲线
fig3, axs3 = plt.subplots(1, 1, figsize=(5, 5))
for i in range(5):
    axs3.plot(sk_recalls[i], sk_precisions[i], label=f'Fold {i + 1} AUPR = {sk_average_precisions[i]:.2f}')
axs3.plot([0, 1], [1, 0], 'k--', label='Random', color='r')
axs3.set_xlim([-0.05, 1.05])
axs3.set_ylim([0.0, 1.05])
axs3.set_xlabel('Recall')
axs3.set_ylabel('Precision')
axs3.set_title('Precision-Recall Curvesin ')
axs3.legend(loc="lower left")
plt.show()

mean_fpr = np.linspace(0, 1, 100)
mean_recall = np.linspace(0, 1, 100)

# 用于存储插值后的数据
tprs = []
precisions = []

# 插值TPR数据
for fpr_temp, tpr_temp in zip(sk_fpr, sk_tprs):
    interp_tpr = np.interp(mean_fpr, fpr_temp, tpr_temp)
    interp_tpr[0] = 0.0  # 确保曲线从0开始
    tprs.append(interp_tpr)
mean_tprs = np.mean(tprs, axis=0)
mean_tprs[-1] = 1.0  # 确保曲线以1结束
mean_auc = np.mean(sk_aucs)

# 插值Precision数据
for recall_temp, precision_temp in zip(sk_recalls, sk_precisions):
    interp_precision = np.interp(mean_recall, recall_temp[::-1], precision_temp[::-1])
    precisions.append(interp_precision)
mean_precisions = np.mean(precisions, axis=0)
mean_average_precision = np.mean(sk_average_precisions)

# 保存所有绘图所需数据
np.savez('complete_average_plot_data.npz', mean_fpr=mean_fpr, mean_tprs=mean_tprs, mean_auc=mean_auc,
         mean_recall=mean_recall, mean_precisions=mean_precisions, mean_average_precision=mean_average_precision)

# 加载数据
data = np.load('complete_average_plot_data.npz')
mean_fpr = data['mean_fpr']
mean_tprs = data['mean_tprs']
mean_auc = data['mean_auc']
mean_recall = data['mean_recall']
mean_precisions = data['mean_precisions']
mean_average_precision = data['mean_average_precision']

# 绘制平均ROC曲线
fig4, axs4 = plt.subplots(1, 1, figsize=(5, 5))
axs4.plot(mean_fpr, mean_tprs, label=f'Average AUC = {mean_auc:.2f}')
axs4.plot([0, 1], [0, 1], 'k--', label='Random', color='r')
axs4.set_xlim([-0.05, 1.05])
axs4.set_ylim([0.0, 1.05])
axs4.set_xlabel('False Positive Rate')
axs4.set_ylabel('True Positive Rate')
axs4.set_title('Average ROC Curve')
axs4.legend(loc="lower right")
plt.show()

# 绘制平均Precision-Recall曲线
fig5, axs5 = plt.subplots(1, 1, figsize=(5, 5))
axs5.plot(mean_recall, mean_precisions, label=f'Average AUPR = {mean_average_precision:.2f}')
axs5.plot([0, 1], [1, 0], 'k--', label='Random', color='r')
axs5.set_xlim([-0.05, 1.05])
axs5.set_ylim([0.0, 1.05])
axs5.set_xlabel('Recall')
axs5.set_ylabel('Precision')
axs5.set_title('Average Precision-Recall Curve')
axs5.legend(loc="lower left")
plt.show()







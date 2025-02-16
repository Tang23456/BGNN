import numpy as np
import torch
import scipy.sparse as sp
import pandas as pd
import math
import random
from sklearn.preprocessing import minmax_scale, scale
import matplotlib.pyplot as plt
from sklearn import metrics
from itertools import cycle
import heapq
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F



def set_seed(seed=123):
    random.seed(seed)  # Python内置随机库
    np.random.seed(seed)  # NumPy随机库
    torch.manual_seed(seed)  # PyTorch随机库
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch CUDA随机库
        torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
        torch.backends.cudnn.deterministic = True  # 确定性算法，可能会影响性能
        torch.backends.cudnn.benchmark = False


def calculate_combined_similarity(disease_matrices, microbe_matrices):
    """
    计算疾病和微生物的综合相似性矩阵。

    参数：
    - disease_matrices：包含5个疾病相似性矩阵的列表。
    - microbe_matrices：包含5个微生物相似性矩阵的列表。

    返回值：
    - sim_disease：疾病的综合相似性矩阵。
    - sim_microbe：微生物的综合相似性矩阵。
    """

    # 计算疾病的综合相似性矩阵
    sum_disease_similarity = np.zeros_like(disease_matrices[0], dtype=np.float64)
    count_disease_nonzero = np.zeros_like(disease_matrices[0])

    for sim_matrix in disease_matrices:
        sum_disease_similarity += sim_matrix.astype(float)
        count_disease_nonzero += (sim_matrix != 0).astype(int)

    count_disease_nonzero[count_disease_nonzero == 0] = 1
    sim_disease = sum_disease_similarity / count_disease_nonzero

    # 计算微生物的综合相似性矩阵
    sum_microbe_similarity = np.zeros_like(microbe_matrices[0])
    count_microbe_nonzero = np.zeros_like(microbe_matrices[0])

    for sim_matrix in microbe_matrices:
        sum_microbe_similarity += sim_matrix.astype(float)
        count_microbe_nonzero += (sim_matrix != 0).astype(int)

    count_microbe_nonzero[count_microbe_nonzero == 0] = 1
    sim_microbe = sum_microbe_similarity / count_microbe_nonzero

    return sim_disease, sim_microbe



def calculate_metapath_optimized(mm, dd, md, n):        #计算n层元路径
    """
    优化版：计算微生物-疾病第n层元路径。

    参数:
    - mm: 微生物相似度矩阵
    - dd: 疾病相似度矩阵
    - md: 微生物疾病关联矩阵
    - n: 元路径的层数

    返回:
    - n层元路径矩阵
    """
    # 基本情况，如果n为1，直接计算并返回第一层元路径矩阵
    dm = md.T
    MM = md @ dd @ dm @ mm
    MD = md @ dd
    if n == 1:
        return mm @ md @ dd
    else:
        #k = n / 2
        k = n
        k = int(k)
        MK = matrixPow(MM, k)
        deep_A = mm @ MK @ MD

        #if n % 2 ==0:
        #    deep_A = mm @ MK @ MD
        #else:
        #    deep_A = mm @ MK

    return deep_A


def get_all_pairs(A_in, deep_A):
    A = A_in.copy()
    A_neg = np.zeros(A.shape)
    m, n = A.shape
    pairs = []
    for i in range(m):
        for j in range(n):
            if A[i, j] == 1:
                j_hat = np.argmin(np.where(A[i] == 0, deep_A[i], np.inf))
                # 将找到的位置在A中置为-1
                if j_hat < n:  # 确保找到的索引在范围内
                    A[i, j_hat] = -1
                    A_neg[i, j_hat] = 1

                # 在deep_A的第j_hat列中找到最小值且A中对应位置为0的元素的行索引i_hat
                #i_hat = np.argmin(np.where(A[:, j_hat] == 0, deep_A[:, j_hat], np.inf))
                i_hat = np.argmin(np.where(A[:, j] == 0, deep_A[:, j], np.inf))
                # 将找到的位置在A中置为-1
                if i_hat < m:  # 确保找到的索引在范围内
                    #A[i_hat, j] = -1
                    pass
                    A_neg[i_hat, j] = 1




                pairs.append([i, j, i_hat, j_hat])
    return pairs, A_neg


def get_neg_pairs(A_in):
    A = A_in.copy()
    m, n = A.shape
    pairs = []

    # Create a set to track used (i-neg, j-neg)
    used_indices = set()

    for i in range(m):
        for j in range(n):
            if A[i, j] == 1:
                while True:
                    # Randomly select i-neg and j-neg
                    i_neg = random.randint(0, m - 1)
                    j_neg = random.randint(0, n - 1)

                    # Check if the condition is met and (i-neg, j-neg) is not used
                    if A[i_neg, j_neg] == 0 and (i_neg, j_neg) not in used_indices:
                        # Mark (i-neg, j-neg) as used
                        used_indices.add((i_neg, j_neg))
                        pairs.append([i_neg, j_neg])
                        break

    return pairs


def matrixPow(Matrix, n):                           # 计算矩阵的n次幂
    """
    计算矩阵的n次幂。

    参数:
    - Matrix: 输入矩阵
    - n: 幂次

    返回:
    - 矩阵的n次幂
    """
    if(type(Matrix) == list):
        Matrix = np.array(Matrix)
    if(n == 1):
        return Matrix
    else:
        return np.matmul(Matrix, matrixPow(Matrix, n - 1))

def adj_matrix_to_edge_index(adj_matrix):
    """
    将邻接矩阵转换为图神经网络的 edge_index 格式。

    参数：
        adj_matrix (numpy.ndarray 或 torch.Tensor): n x n 的邻接矩阵。

    返回：
        torch.Tensor: 2 x E 的 edge_index，其中 E 是边的数量。
    """
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = torch.tensor(adj_matrix)

    # 获取非零元素的索引
    indices = adj_matrix.nonzero(as_tuple=True)  # 返回分开的索引

    # 构造 edge_index
    edge_index = torch.stack(indices, dim=0)
    return edge_index


def build_pos_adjacency_matrix(A, epsilon=1e-5):
    m, d = A.shape
    # 创建单位矩阵 E
    E_microbe = np.eye(m)
    E_disease = np.eye(d)

    # 拼接左上角和右下角的 E
    E = np.block([
        [E_microbe, np.zeros((m, d))],
        [np.zeros((d, m)), E_disease]
    ])

    # 拼接整个大矩阵 B
    B = np.block([
        [E[:m, :m], A],
        [A.T, E[m:, m:]]
    ])

    # 添加扰动到对角线，确保矩阵可逆
    perturbation = epsilon * np.eye(m + d)
    B_perturbed = B + perturbation


    return B_perturbed


def build_negative_adjacency_matrix(A):
    m, d = A.shape

    # 构造零矩阵
    zero_microbe = np.zeros((m, m))  # 左上角零矩阵
    zero_disease = np.zeros((d, d))  # 右下角零矩阵

    # 构造大矩阵 B
    B = np.block([
        [zero_microbe, A],  # 左上角与右上角
        [A.T, zero_disease]  # 左下角与右下角
    ])

    return B


def build_perturbed_adjacency_matrix_with_antidiagonal(A, epsilon=1e-5):
    m, d = A.shape

    # 构造零矩阵
    zero_microbe = np.zeros((m, m))  # 左上角零矩阵
    zero_disease = np.zeros((d, d))  # 右下角零矩阵

    # 构造大矩阵 B
    B = np.block([
        [zero_microbe, A],  # 左上角与右上角
        [A.T, zero_disease]  # 左下角与右下角
    ])

    # 添加副对角线扰动
    n = B.shape[0]
    for i in range(n):
        j = n - 1 - i  # 副对角线索引
        B[i, j] += epsilon

    # 如果阶数为奇数，确保中间元素为 0
    if n % 2 == 1:
        mid = n // 2
        B[mid, mid] = 0

    return B


def process_neg_matrices(pos, neg_set):
    """
    输入一个矩阵 pos，一个三维矩阵集合 neg_set，以及一个整数 l。
    输出一个三维矩阵集合 out_set，
    第一个二维矩阵是 pos 左乘 neg_set 的第一个元素，
    后续矩阵按顺序为 pos 右乘 neg_set 中的矩阵。

    :param pos: numpy.ndarray, 2D 矩阵
    :param neg_set: numpy.ndarray, 3D 矩阵集合
    :param l: int, 未明确使用但保留
    :return: numpy.ndarray, 3D 矩阵集合
    """
    # 获取 neg_set 的矩阵数量
    num_matrices = neg_set.shape[0]

    # 初始化输出集合
    out_set = []

    # 计算第一个矩阵：pos 左乘 neg_set 中的第一个矩阵
    first_matrix = np.dot(neg_set[0], pos)  # 左乘
    out_set.append(first_matrix)

    # 计算后续矩阵：pos 右乘 neg_set 中的每个矩阵
    for i in range(num_matrices):
        right_mult_matrix = np.dot(pos, neg_set[i])  # 右乘
        out_set.append(right_mult_matrix)

    # 转换为三维矩阵并返回
    return np.stack(out_set)

def select_rows(A, selected_rows):
    """
    从矩阵A中选择指定的行。

    参数:
    A (torch.Tensor): 原始矩阵 (m x n)。
    selected_rows (list): 需要保留的行的索引列表。

    返回:
    torch.Tensor: 仅保留指定行的新矩阵。
    """
    device = A.device  # 获取A所在的设备
    dtype = A.dtype  # 获取A的数据类型
    m, n = A.shape  # 获取原始矩阵的行数和列数
    k = len(selected_rows)  # 要保留的行数

    # 构造选择矩阵 M 并移动到与 A 相同的设备和数据类型
    M = torch.zeros(k, m, device=device, dtype=dtype)
    for idx, row in enumerate(selected_rows):
        M[idx, row] = 1

    # 通过矩阵乘法得到结果矩阵 B
    B = torch.mm(M, A)

    return B

def sim_cos( z1, z2):
    z1_out = F.normalize(z1, p=2, dim=1)
    z2_out = F.normalize(z2, p=2, dim=1)
    return torch.mm(z1_out, z2_out.t())
def constrate_loss_calculate(train_i_mic_feature_tensor,train_i_hat_mic_feature_tensor,
                            train_j_disease_feature_tensor,train_j_hat_disease_feature_tensor, tau=1):
    f = lambda x: torch.exp(x / tau)
    i_j_sim = f(sim_cos(train_i_mic_feature_tensor, train_j_disease_feature_tensor))
    i_j_hat_sim = f(sim_cos(train_i_mic_feature_tensor, train_j_hat_disease_feature_tensor))
    i_hat_j_sim = f(sim_cos(train_i_hat_mic_feature_tensor, train_j_disease_feature_tensor))
    diag_i_j_sim = i_j_sim.diag()
    diag_i_j_hat_sim = i_j_hat_sim.diag()
    diag_i_hat_j_sim = i_hat_j_sim.diag()

    constrate_loss = -torch.log(diag_i_j_sim / (diag_i_j_sim + diag_i_j_hat_sim + diag_i_hat_j_sim)).mean()
    return  constrate_loss
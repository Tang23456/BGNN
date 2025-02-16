import torch
import torch.nn as nn
from utils import  *
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNBranch_pos(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNBranch_pos, self).__init__()

        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, 62)
        self.linear3 = nn.Linear(62, output_dim)


        self.gcn1 = GCNConv(256, 256)
        self.gcn2 = GCNConv(62, 62)
        self.gcn3 = GCNConv(output_dim, output_dim)
        self.gcn4 = GCNConv(output_dim, output_dim)
        self.gcn5 = GCNConv(output_dim, output_dim)
        self.gcn6 = GCNConv(output_dim, output_dim)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weight = 0.5


    def forward(self, x, A_pos):

        edge_index_pos = adj_matrix_to_edge_index(A_pos).to(self.device)

        x1_Lin_out = self.linear1(x)
        x1_out = x1_Lin_out + F.relu(self.gcn1(x1_Lin_out, edge_index_pos))

        x2_Lin_out = self.linear2(x1_out)
        x2_out = x2_Lin_out + F.relu(self.gcn2(x2_Lin_out, edge_index_pos))


        x3_Lin_out = self.linear3(x2_out)
        x3_out = x3_Lin_out + self.weight *F.relu(self.gcn3(x3_Lin_out, edge_index_pos))

        x4_out = x3_out + self.weight *F.relu(self.gcn4(x3_out, edge_index_pos))

        self.weight *= 0.5

        x5_out = x4_out + self.weight *F.relu(self.gcn5(x4_out, edge_index_pos))

        x6_out = x5_out + self.weight *self.gcn6(x5_out, edge_index_pos)



        return x6_out

class GCNBranch_neg(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNBranch_neg, self).__init__()

        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, 62)
        self.linear3 = nn.Linear(62, output_dim)


        self.gcn1 = GCNConv(256, 256)
        self.gcn2 = GCNConv(62, 62)
        self.gcn3 = GCNConv(output_dim, output_dim)
        self.gcn4 = GCNConv(output_dim, output_dim)
        self.gcn5 = GCNConv(output_dim, output_dim)
        self.gcn6 = GCNConv(output_dim, output_dim)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weight = 0.5

        self.attention_weight2 = nn.Parameter(torch.ones(5))  # Learnable weights for the 4 GCN outputs


    def forward(self, x, A_neg, A_pos):

        attention_scores2 = F.softmax(self.attention_weight2, dim=0)
        edge_index_neg = adj_matrix_to_edge_index(A_neg).to(self.device)
        x1_Lin_out = self.linear1(x)
        x1_out = x1_Lin_out + F.relu(self.gcn1(x1_Lin_out, edge_index_neg))

        temp_set = np.expand_dims(A_neg, axis=0)
        A_neg_set = process_neg_matrices(A_pos, temp_set)
        noise = np.random.normal(0, 1e-5, size=A_neg.shape)
        A_neg = A_neg + noise
        temp_A_neg_neg = np.linalg.inv(A_neg)
        A_neg = np.max(A_neg_set, axis=0)
        A_neg = np.dot(A_neg, temp_A_neg_neg)
        edge_index_neg = adj_matrix_to_edge_index(A_neg).to(self.device)
        x2_Lin_out = self.linear2(x1_out)
        x2_out = x2_Lin_out + F.relu(self.gcn2(x2_Lin_out, edge_index_neg))


        A_neg_set = process_neg_matrices(A_pos, A_neg_set)
        noise = np.random.normal(0, 1e-5, size=A_neg.shape)
        A_neg = A_neg + noise
        temp_A_neg_neg = np.linalg.inv(A_neg)
        A_neg = np.max(A_neg_set, axis=0)
        A_neg = np.dot(A_neg, temp_A_neg_neg)
        edge_index_neg = adj_matrix_to_edge_index(A_neg).to(self.device)
        x3_Lin_out = self.linear3(x2_out)
        x3_out = F.relu(self.gcn3(x3_Lin_out, edge_index_neg))

        A_neg_set = process_neg_matrices(A_pos, A_neg_set)
        noise = np.random.normal(0, 1e-5, size=A_neg.shape)
        A_neg = A_neg + noise
        temp_A_neg_neg = np.linalg.inv(A_neg)
        A_neg = np.max(A_neg_set, axis=0)
        A_neg = np.dot(A_neg, temp_A_neg_neg)
        edge_index_neg = adj_matrix_to_edge_index(A_neg).to(self.device)
        x4_out = F.relu(self.gcn4(x3_out, edge_index_neg))


        A_neg_set = process_neg_matrices(A_pos, A_neg_set)
        noise = np.random.normal(0, 1e-5, size=A_neg.shape)
        A_neg = A_neg + noise
        temp_A_neg_neg = np.linalg.inv(A_neg)
        A_neg = np.max(A_neg_set, axis=0)
        A_neg = np.dot(A_neg, temp_A_neg_neg)
        edge_index_neg = adj_matrix_to_edge_index(A_neg).to(self.device)
        x5_out = F.relu(self.gcn5(x4_out, edge_index_neg))

        A_neg_set = process_neg_matrices(A_pos, A_neg_set)
        noise = np.random.normal(0, 1e-5, size=A_neg.shape)
        A_neg = A_neg + noise
        temp_A_neg_neg = np.linalg.inv(A_neg)
        A_neg = np.max(A_neg_set, axis=0)
        A_neg = np.dot(A_neg, temp_A_neg_neg)
        edge_index_neg = adj_matrix_to_edge_index(A_neg).to(self.device)
        x6_out = self.gcn6(x5_out, edge_index_neg)


        fin_out = x3_Lin_out * attention_scores2[0] + x3_out * attention_scores2[1] + x4_out * attention_scores2[2] + x5_out * attention_scores2[3] + x6_out *  attention_scores2[4]

        return fin_out




class GCNBranch_neg_Normal_A(nn.Module):        #负边只是A
    def __init__(self, input_dim, output_dim):
        super(GCNBranch_neg_Normal_A, self).__init__()

        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, 62)
        self.linear3 = nn.Linear(62, output_dim)


        self.gcn1 = GCNConv(256, 256)
        self.gcn2 = GCNConv(62, 62)
        self.gcn3 = GCNConv(output_dim, output_dim)
        self.gcn4 = GCNConv(output_dim, output_dim)
        self.gcn5 = GCNConv(output_dim, output_dim)
        self.gcn6 = GCNConv(output_dim, output_dim)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weight = 0.5

    def forward(self, x, A_neg, A_pos):
        edge_index_neg = adj_matrix_to_edge_index(A_neg).to(self.device)
        x1_Lin_out = self.linear1(x)
        x1_out = x1_Lin_out + F.relu(self.gcn1(x1_Lin_out, edge_index_neg))

        x2_Lin_out = self.linear2(x1_out)
        x2_out = x2_Lin_out + F.relu(self.gcn2(x2_Lin_out, edge_index_neg))



        x3_Lin_out = self.linear3(x2_out)
        x3_out = x3_Lin_out + self.weight *F.relu(self.gcn3(x3_Lin_out, edge_index_neg))

        x4_out = x3_out +  self.weight *F.relu(self.gcn4(x3_out, edge_index_neg))
        #
        self.weight *= 0.5
        x5_out = x4_out +  self.weight *F.relu(self.gcn5(x4_out, edge_index_neg))
        x6_out = x5_out +  self.weight *self.gcn6(x5_out, edge_index_neg)

        return x6_out

class GCNBranch_neg_change(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNBranch_neg_change, self).__init__()

        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, 62)
        self.linear3 = nn.Linear(62, output_dim)


        self.gcn1 = GCNConv(256, 256)
        self.gcn2 = GCNConv(62, 62)
        self.gcn3 = GCNConv(output_dim, output_dim)
        self.gcn4 = GCNConv(output_dim, output_dim)
        self.gcn5 = GCNConv(output_dim, output_dim)
        self.gcn6 = GCNConv(output_dim, output_dim)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weight = 0.5

    def forward(self, x, A_neg, A_pos):
        edge_index_neg = adj_matrix_to_edge_index(A_neg).to(self.device)
        x1_Lin_out = self.linear1(x)
        x1_out = x1_Lin_out + F.relu(self.gcn1(x1_Lin_out, edge_index_neg))

        temp_set = np.expand_dims(A_neg, axis=0)
        A_neg_set = process_neg_matrices(A_pos, temp_set)
        A_neg = np.max(A_neg_set, axis=0)
        edge_index_neg = adj_matrix_to_edge_index(A_neg).to(self.device)
        x2_Lin_out = self.linear2(x1_out)
        x2_out = x2_Lin_out + F.relu(self.gcn2(x2_Lin_out, edge_index_neg))


        A_neg_set = process_neg_matrices(A_pos, A_neg_set)
        A_neg = np.max(A_neg_set, axis=0)
        edge_index_neg = adj_matrix_to_edge_index(A_neg).to(self.device)
        x3_Lin_out = self.linear3(x2_out)
        x3_out = x3_Lin_out + self.weight *F.relu(self.gcn3(x3_Lin_out, edge_index_neg))

        A_neg_set = process_neg_matrices(A_pos, A_neg_set)
        A_neg = np.max(A_neg_set, axis=0)
        edge_index_neg = adj_matrix_to_edge_index(A_neg).to(self.device)
        x4_out = x3_out +  self.weight *F.relu(self.gcn4(x3_out, edge_index_neg))

        self.weight *= 0.5

        A_neg_set = process_neg_matrices(A_pos, A_neg_set)
        A_neg = np.max(A_neg_set, axis=0)
        edge_index_neg = adj_matrix_to_edge_index(A_neg).to(self.device)
        x5_out = x4_out +  self.weight *F.relu(self.gcn5(x4_out, edge_index_neg))

        A_neg_set = process_neg_matrices(A_pos, A_neg_set)
        A_neg = np.max(A_neg_set, axis=0)
        edge_index_neg = adj_matrix_to_edge_index(A_neg).to(self.device)
        x6_out = x5_out +  self.weight *self.gcn6(x5_out, edge_index_neg)

        return x6_out

################################   实验结果  #####################################
#   一直 A neg auc 971 aupr 969
#   算法改进 a neg auc aupr 945 949
#   算法改进+逆矩阵消除上一层邻接矩阵的影响  988 987


class TwoBranchGNN(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.5):
        super(TwoBranchGNN, self).__init__()
        # input_dim ->256 -> 62-> 32(output_dim)

        # 正样本和负样本的独立 GCN 模型
        self.gcn_pos = GCNBranch_pos(input_dim, output_dim)
        self.gcn_neg = GCNBranch_neg(input_dim, output_dim)

        # 加权系数
        self.alpha = alpha
        # self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))

    def forward(self, feature, A_pos, A_neg ):
        """
        data_pos: 包含正样本图的数据对象 (PyG Data object)
        data_neg: 包含负样本图的数据对象 (PyG Data object)
        """

        # 正样本分支计算
        x_pos = self.gcn_pos(feature, A_pos)

        # 负样本分支计算
        x_neg = self.gcn_neg(feature, A_neg, A_pos)

        # 加权和
        output = self.alpha * x_pos - (1 - self.alpha) * x_neg

        return output



class TwoBranchGNN_Only_Pos(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.5):
        super(TwoBranchGNN_Only_Pos, self).__init__()
        # input_dim ->256 -> 62-> 32(output_dim)

        # 正样本和负样本的独立 GCN 模型
        self.gcn_pos = GCNBranch_pos(input_dim, output_dim)
        self.gcn_neg = GCNBranch_neg(input_dim, output_dim)

        # 加权系数
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))

    def forward(self, feature, A_pos, A_neg):
        """
        data_pos: 包含正样本图的数据对象 (PyG Data object)
        data_neg: 包含负样本图的数据对象 (PyG Data object)
        """

        # 正样本分支计算
        x_pos = self.gcn_pos(feature, A_pos)

        return x_pos


class TwoBranchGNN_Only_Neg(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.5):
        super(TwoBranchGNN_Only_Neg, self).__init__()
        # input_dim ->256 -> 62-> 32(output_dim)

        # 正样本和负样本的独立 GCN 模型
        self.gcn_pos = GCNBranch_pos(input_dim, output_dim)
        self.gcn_neg = GCNBranch_neg(input_dim, output_dim)

        # 加权系数
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))

    def forward(self, feature, A_pos, A_neg ):
        """
        data_pos: 包含正样本图的数据对象 (PyG Data object)
        data_neg: 包含负样本图的数据对象 (PyG Data object)
        """
        # # 负样本分支计算
        x_neg = self.gcn_neg(feature, A_neg, A_pos)

        return -x_neg

class TwoBranchGNN_Normal_Neg_A(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.5):
        super(TwoBranchGNN_Normal_Neg_A, self).__init__()
        # input_dim ->256 -> 62-> 32(output_dim)

        # 正样本和负样本的独立 GCN 模型
        self.gcn_pos = GCNBranch_pos(input_dim, output_dim)
        self.gcn_neg = GCNBranch_neg_Normal_A(input_dim, output_dim)

        # 加权系数
        self.alpha = alpha

    def forward(self, feature, A_pos, A_neg ):
        """
        data_pos: 包含正样本图的数据对象 (PyG Data object)
        data_neg: 包含负样本图的数据对象 (PyG Data object)
        """

        # 正样本分支计算
        x_pos = self.gcn_pos(feature, A_pos)

        # # 负样本分支计算
        x_neg = self.gcn_neg(feature, A_neg, A_pos)

        # 加权和
        output = self.alpha * x_pos - (1 - self.alpha) * x_neg

        return output

class TwoBranchGNN_Normal_Neg_A_change(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.5):
        super(TwoBranchGNN_Normal_Neg_A_change, self).__init__()
        # input_dim ->256 -> 62-> 32(output_dim)

        # 正样本和负样本的独立 GCN 模型
        self.gcn_pos = GCNBranch_pos(input_dim, output_dim)
        self.gcn_neg = GCNBranch_neg_change(input_dim, output_dim)

        # 加权系数
        self.alpha = alpha

    def forward(self, feature, A_pos, A_neg ):
        """
        data_pos: 包含正样本图的数据对象 (PyG Data object)
        data_neg: 包含负样本图的数据对象 (PyG Data object)
        """

        # 正样本分支计算
        x_pos = self.gcn_pos(feature, A_pos)

        # # 负样本分支计算
        x_neg = self.gcn_neg(feature, A_neg, A_pos)

        # 加权和
        output = self.alpha * x_pos - (1 - self.alpha) * x_neg

        return output





import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch

class UnimodalEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super(UnimodalEncoder, self).__init__()
        self.conv1d = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.layer_norm_cnn = nn.LayerNorm(hidden_dim)
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.layer_norm_out = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.layer_norm_cnn(x))
        x = self.dropout(x)
        x, _ = self.bilstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        x = self.layer_norm_out(x)
        x = self.dropout(x)
        return x

class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super(CrossModalFusion, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.multihead_attn(query, key, value)
        return self.norm(query + self.dropout(attn_output))

class IntraSentenceGNN(nn.Module):
    def __init__(self, unimodal_out_dim, hidden_dim, num_heads, dropout):
        super(IntraSentenceGNN, self).__init__()
        self.text_proj = nn.Linear(unimodal_out_dim, hidden_dim)
        self.audio_proj = nn.Linear(unimodal_out_dim, hidden_dim)
        self.video_proj = nn.Linear(unimodal_out_dim, hidden_dim)
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)

    def forward(self, text_features, audio_features, video_features):
        batch_size = text_features.size(0)
        t_p, a_p, v_p = self.text_proj(text_features), self.audio_proj(audio_features), self.video_proj(video_features)
        graph_list = []
        for i in range(batch_size):
            x = torch.stack([t_p[i], a_p[i], v_p[i]], dim=0)
            edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long, device=x.device)
            graph_list.append(Data(x=x, edge_index=edge_index))
        batch = Batch.from_data_list(graph_list).to(text_features.device)
        x = F.elu(self.gat1(batch.x, batch.edge_index))
        x = self.gat2(x, batch.edge_index)
        return global_mean_pool(x, batch.batch)

class InterSentenceGNN(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, output_dim: int, num_heads: int, dropout: float):
        super(InterSentenceGNN, self).__init__()
        self.num_relations = 3
        self.node_weight_network = nn.Sequential(nn.Linear(self.num_relations, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.gat_layer1 = GATv2Conv(node_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.gat_layer2 = GATv2Conv(hidden_dim, output_dim, heads=1, dropout=dropout, concat=False)
        self.layer_norm = nn.LayerNorm(output_dim)

    def build_graph(self, num_nodes: int, speaker_ids: torch.Tensor, device: torch.device):
        edge_list, type_list = [], []
        REL_SEQ, REL_CTX, REL_SPK = 0, 1, 2
        if num_nodes <= 1: return torch.tensor([[0], [0]], dtype=torch.long, device=device), torch.zeros((1, 3), device=device)
        u_idx = num_nodes - 1
        spk_map = {}
        for i in range(num_nodes):
            sid = speaker_ids[i].item()
            if sid not in spk_map: spk_map[sid] = []
            spk_map[sid].append(i)
        for i in range(num_nodes):
            if i > 0: edge_list.extend([[i-1, i], [i, i-1]]); type_list.extend([REL_SEQ, REL_SEQ])
            if i < u_idx: edge_list.extend([[i, u_idx], [u_idx, i]]); type_list.extend([REL_CTX, REL_CTX])
        for nodes in spk_map.values():
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    edge_list.extend([[nodes[i], nodes[j]], [nodes[j], nodes[i]]]); type_list.extend([REL_SPK, REL_SPK])
        if not edge_list: return torch.tensor([[0], [0]], dtype=torch.long, device=device), torch.zeros((num_nodes, 3), device=device)
        idx = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
        rel_f = torch.zeros((num_nodes, 3), device=device)
        rel_f.scatter_add_(0, idx[0].unsqueeze(1).expand(-1, 3), F.one_hot(torch.tensor(type_list, device=device), 3).float())
        return torch.unique(idx, dim=1), rel_f

    def forward(self, node_features_list, dialogue_speaker_ids_list):
        graphs = []
        for nf, ds in zip(node_features_list, dialogue_speaker_ids_list):
            edge_index, rel_feat = self.build_graph(nf.size(0), ds, nf.device)
            graphs.append(Data(x=nf, edge_index=edge_index, relation_features=rel_feat))
        batch = Batch.from_data_list(graphs).to(node_features_list[0].device)
        weighted_x = batch.x * self.node_weight_network(batch.relation_features.float())
        x = F.elu(self.gat_layer1(weighted_x, batch.edge_index))
        x = self.layer_norm(self.gat_layer2(x, batch.edge_index))
        return x[batch.ptr[1:] - 1]
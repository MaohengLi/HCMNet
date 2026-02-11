import torch
import torch.nn as nn
from modules import UnimodalEncoder, CrossModalFusion, IntraSentenceGNN, InterSentenceGNN

class SarcasmGNNClassifier(nn.Module):
    def __init__(self, text_dim, audio_dim, video_dim, enc_hid, enc_out, gnn_hid, out_dim, heads, dropout):
        super(SarcasmGNNClassifier, self).__init__()
        self.text_encoder = UnimodalEncoder(text_dim, enc_hid, enc_out, dropout)
        self.audio_encoder = UnimodalEncoder(audio_dim, enc_hid, enc_out, dropout)
        self.video_encoder = UnimodalEncoder(video_dim, enc_hid, enc_out, dropout)
        self.attn_audio = CrossModalFusion(enc_out, heads, dropout)
        self.attn_video = CrossModalFusion(enc_out, heads, dropout)
        self.attn_text = CrossModalFusion(enc_out, heads, dropout)
        self.fusion_projector = nn.Linear(enc_out * 3, gnn_hid)
        self.intra_gnn = IntraSentenceGNN(enc_out, gnn_hid, heads, dropout)
        self.inter_gnn = InterSentenceGNN(gnn_hid, gnn_hid, gnn_hid, heads, dropout)
        final_fusion_dim = gnn_hid + gnn_hid + (enc_out * 3)
        self.classifier = nn.Sequential(
            nn.Linear(final_fusion_dim, gnn_hid),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(gnn_hid, out_dim)
        )

    def get_cross_attn_features(self, t, a, v):
        f_a = self.attn_audio(query=t, key=a, value=a)
        f_v = self.attn_video(query=t, key=v, value=v)
        f_t = self.attn_text(query=t, key=t, value=t)
        return f_t, f_a, f_v

    def apply_cross_modal_fusion_to_node(self, t_feat, a_feat, v_feat):
        t, a, v = t_feat.unsqueeze(1), a_feat.unsqueeze(1), v_feat.unsqueeze(1)
        f_t, f_a, f_v = self.get_cross_attn_features(t, a, v)
        combined = torch.cat([f_a*0.5, f_v*0.5, f_t], dim=-1)
        return self.fusion_projector(combined).squeeze(1)

    def forward(self, t_u, a_u, v_u, t_c, a_c, v_c, d_s):
        e_t_u, e_a_u, e_v_u = self.text_encoder(t_u.unsqueeze(1)), self.audio_encoder(a_u.unsqueeze(1)), self.video_encoder(v_u.unsqueeze(1))
        t_q, a_q, v_q = e_t_u.unsqueeze(0), e_a_u.unsqueeze(0), e_v_u.unsqueeze(0)
        f_t_res, f_a_res, f_v_res = self.get_cross_attn_features(t_q, a_q, v_q)
        intra = self.intra_gnn(f_t_res.squeeze(0), f_a_res.squeeze(0), f_v_res.squeeze(0))
        nf_list = []
        for i in range(t_u.size(0)):
            c_nodes = self.apply_cross_modal_fusion_to_node(
                self.text_encoder(t_c[i].unsqueeze(1)),
                self.audio_encoder(a_c[i].unsqueeze(1)),
                self.video_encoder(v_c[i].unsqueeze(1))
            )
            u_node = self.apply_cross_modal_fusion_to_node(
                e_t_u[i].unsqueeze(0),
                e_a_u[i].unsqueeze(0),
                e_v_u[i].unsqueeze(0)
            )
            nf_list.append(torch.cat([c_nodes, u_node], dim=0))
        inter = self.inter_gnn(nf_list, [d[:n.size(0)] for d, n in zip(d_s, nf_list)])
        res = torch.cat([f_a_res, f_v_res, f_t_res], dim=-1).squeeze(0)
        logits = self.classifier(torch.cat([intra, inter, res*4], dim=1))
        return logits, inter
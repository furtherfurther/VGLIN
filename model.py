import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model_gcn import GCN


iemocap_similarity_matrix = torch.tensor([
    [1.0000, 0.5913, 0.9622, 0.6478, 0.9873, 0.3891],  # happy
    [0.5913, 1.0000, 0.2541, 1.0000, 0.7266, 0.9869],  # sad
    [0.9622, 0.2541, 1.0000, 0.3417, 0.8625, 0.0000],  # neutral
    [0.6478, 1.0000, 0.3417, 1.0000, 0.7373, 0.9250],  # angry
    [0.9873, 0.7266, 0.8625, 0.7373, 1.0000, 0.5806],  # excited
    [0.3891, 0.9869, 0.0000, 0.9250, 0.5806, 1.0000]   # frustrated
], dtype=torch.float32)  

meld_similarity_matrix = torch.tensor([
    [1.0000, 0.8756, 0.0000, 0.2662, 0.9425, 0.0086, 0.3716],  # Neutral
    [0.8756, 1.0000, 0.5731, 0.7446, 0.9792, 0.5770, 0.7632],  # Surprise
    [0.0000, 0.5731, 1.0000, 0.9438, 0.4474, 1.0000, 0.7768],  # Fear
    [0.2662, 0.7446, 0.9438, 1.0000, 0.6347, 0.9745, 0.9852],  # Sadness
    [0.9425, 0.9792, 0.4474, 0.6347, 1.0000, 0.4558, 0.6995],  # Joy
    [0.0086, 0.5770, 1.0000, 0.9745, 0.4558, 1.0000, 0.8395],  # Disgust
    [0.3716, 0.7632, 0.7768, 0.9852, 0.6995, 0.8395, 1.0000]   # Angry
], dtype=torch.float32)


class VGSRLoss(nn.Module):
    def __init__(self, gamma=2.0, size_average=True):
        super(VGSRLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logits, labels, dataset, matrix=iemocap_similarity_matrix_new):
       
        if dataset == "IEMOCAP":
            matrix = iemocap_similarity_matrix
        elif dataset == "MELD":
            matrix = meld_similarity_matrix
        similarity_matrix = matrix.to(labels.device)
        labels = labels.view(-1)

        log_p = F.log_softmax(logits, dim=-1)
        pt = torch.exp(log_p)
        sub_pt = 1 - pt

       
        similarity_weights = similarity_matrix[labels]  
      
        focal_loss = - (sub_pt ** self.gamma) * log_p * similarity_weights.unsqueeze(1)

        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class KLDivLoss(nn.Module):
  
    def __init__(self):
        super(KLDivLoss, self).__init__()
       
        self.loss = nn.KLDivLoss(reduction='sum')

    def forward(self, log_pred, target):
      
        loss = self.loss(log_pred, target) / log_pred.size(0)
        return loss


class MaskedNLLLoss(nn.Module):
    
    def __init__(self, weight=None):
       
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
     
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
      
        mask_ = mask.view(-1, 1)
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
         
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())
        return loss


def gelu(x):
 
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
 
    def __init__(self, d_model, d_ff, dropout=0.1):
      
        super(PositionwiseFeedForward, self).__init__()
    
        self.w_1 = nn.Linear(d_model, d_ff)    
        self.w_2 = nn.Linear(d_ff, d_model)
      
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
        self.actv = gelu
      
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
    
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
     
        output = self.dropout_2(self.w_2(inter))
    
        return output + x


class MultiHeadedAttention(nn.Module):
  
    def __init__(self, head_count, model_dim, dropout=0.1):
      
        assert model_dim % head_count == 0
      
        self.dim_per_head = model_dim // head_count
        
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

      
        # model_dim =  head_count * self.dim_per_head
        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
       
        self.softmax = nn.Softmax(dim=-1)
     
        self.dropout = nn.Dropout(dropout)
       
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
     
        batch_size = key.size(0)
        
        dim_per_head = self.dim_per_head
        head_count = self.head_count

       
        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)

      
        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

      
        query = query / math.sqrt(dim_per_head)
       
        scores = torch.matmul(query, key.transpose(2, 3))

    
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e10)

      
        attn = self.softmax(scores)
      
        drop_attn = self.dropout(attn)
     
        context = torch.matmul(drop_attn, value).transpose(1, 2).\
                    contiguous().view(batch_size, -1, head_count * dim_per_head)
      
        output = self.linear(context)
     
        return output


class PositionalEncoding(nn.Module):
  
    def __init__(self, dim, max_len=512):
      
        super(PositionalEncoding, self).__init__()
      
        pe = torch.zeros(max_len, dim)
      
        position = torch.arange(0, max_len).unsqueeze(1)
       
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
      
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
       
        pe = pe.unsqueeze(0)
    
        self.register_buffer('pe', pe)

    def forward(self, x, speaker_emb):
      
        L = x.size(1)
      
        pos_emb = self.pe[:, :L]
      
        x = x + pos_emb + speaker_emb
      
        return x


class TransformerEncoderLayer(nn.Module):
  
    def __init__(self, d_model, heads, d_ff, dropout):
      
        super(TransformerEncoderLayer, self).__init__()
       
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
    
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
      
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b, mask):
      
        if inputs_a.equal(inputs_b):
          
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

          
            mask = mask.unsqueeze(1)
           
            context = self.self_attn(inputs_b, inputs_b, inputs_b, mask=mask)
        else:
         
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_a, inputs_a, inputs_b, mask=mask)

       
        out = self.dropout(context) + inputs_b
       
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
  
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1):
      
        super(TransformerEncoder, self).__init__()
       
        self.d_model = d_model
        self.layers = layers
       
        self.pos_emb = PositionalEncoding(d_model)
       
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a, x_b, mask, speaker_emb):
      
        if x_a.equal(x_b):
         
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
           
            for i in range(self.layers):
     
                x_b = self.transformer_inter[i](i, x_b, x_b, mask.eq(0))
       
        else:
          
            x_a = self.pos_emb(x_a, speaker_emb)
            x_a = self.dropout(x_a)
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
               
                x_b = self.transformer_inter[i](i, x_a, x_b, mask.eq(0))
        return x_b


class EnhancedFilterModule(nn.Module):
    def __init__(self, hidden_size, dataset):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        if dataset == 'MELD':
        
            self.gate[0].weight.data.copy_(torch.eye(hidden_size, hidden_size))
           
            self.gate[0].weight.requires_grad = False

    def forward(self, x):
        gate = self.gate(x)
        out = gate * x
        return out


class Multimodal_GatedFusion(nn.Module):
   
    def __init__(self, hidden_size):
      
        super(Multimodal_GatedFusion, self).__init__()
      
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
    
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, a, b, c):
     
        a_new = a.unsqueeze(-2)
        b_new = b.unsqueeze(-2)
        c_new = c.unsqueeze(-2)
       
        utters = torch.cat([a_new, b_new, c_new], dim=-2)
       
        utters_fc = torch.cat([self.fc(a).unsqueeze(-2), self.fc(b).unsqueeze(-2), self.fc(c).unsqueeze(-2)], dim=-2)
     
        utters_softmax = self.softmax(utters_fc)

   
        utters_three_model = utters_softmax * utters
    
        final_rep = torch.sum(utters_three_model, dim=-2, keepdim=False)
   
        return final_rep


def simple_batch_graphify(features, lengths, no_cuda):
    node_features = []
    batch_size = features.size(1)
    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

    node_features = torch.cat(node_features, dim=0)

    if not no_cuda:
        node_features = node_features.to("cuda:0")
    return node_features


class GlobalSemanticPreservingTransformerInteraction(nn.Module):
    def __init__(self, hidden_dim, dataset=None, n_head=None, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.q_q = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.k_q = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.v_q = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        self.q_q_gate = EnhancedFilterModule(hidden_dim, dataset)
        self.k_q_gate = EnhancedFilterModule(hidden_dim, dataset)
        self.v_q_gate = EnhancedFilterModule(hidden_dim, dataset)

        self.features_reduce = nn.Linear(3 * hidden_dim, hidden_dim)

    def forward(self, q, k, v, u_mask, spk_embeddings):
        q_q_out = self.q_q(q, q, u_mask, spk_embeddings)
        k_q_out = self.k_q(k, q, u_mask, spk_embeddings)
        v_q_out = self.v_q(v, q, u_mask, spk_embeddings)

        q_q_out = self.q_q_gate(q_q_out)
        k_q_out = self.k_q_gate(k_q_out)
        v_q_out = self.v_q_gate(v_q_out)

        out = self.features_reduce(torch.cat([q_q_out, k_q_out, v_q_out], dim=-1))

        return out


class Transformer_Based_Model(nn.Module):
  
    def __init__(self, dataset, temp, D_text, D_visual, D_audio, n_head,
                 n_classes, hidden_dim, n_speakers, dropout):
   
    def __init__(self, dataset, temp, D_text, D_visual, D_audio, n_head,
                 n_classes, hidden_dim, n_speakers, dropout, D_g=1024, graph_hidden_size=1024, num_L = 3, num_K = 4, modals='avl'):
      
        super(Transformer_Based_Model, self).__init__()
       
        self.temp = temp
        self.n_classes = n_classes
        self.n_speakers = n_speakers
        self.dropout = dropout
       
        if self.n_speakers == 2:
            padding_idx = 2
        if self.n_speakers == 9:
            padding_idx = 9
        
       
        self.speaker_embeddings = nn.Embedding(n_speakers+1, hidden_dim, padding_idx)
        
      
        self.textf_input = nn.Conv1d(D_text, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.acouf_input = nn.Conv1d(D_audio, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.visuf_input = nn.Conv1d(D_visual, hidden_dim, kernel_size=1, padding=0, bias=False)

     
        self.last_gate = Multimodal_GatedFusion(hidden_dim)

    
        self.t_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.a_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.v_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )

        self.t_out = GlobalSemanticPreservingTransformerInteraction(hidden_dim, dataset, n_head, dropout)
        self.a_out = GlobalSemanticPreservingTransformerInteraction(hidden_dim, dataset, n_head, dropout)
        self.v_out = GlobalSemanticPreservingTransformerInteraction(hidden_dim, dataset, n_head, dropout)

      
        self.graph_model = GCN(n_dim=D_g, nhidden=graph_hidden_size,
                               dropout=self.dropout, lamda=0.5, alpha=0.1, variant=True,
                               return_feature=True, use_residue=False, n_speakers=n_speakers,
                               modals='avl', use_speaker=True, use_modal=False, num_L=num_L,
                               num_K=num_K)
        self.multi_modal = True
        self.att_type = 'concat_DHT'
        self.modals = self.modals = [x for x in modals]  # a, v, l
        self.use_residue = False

        if self.multi_modal:
            self.dropout_ = nn.Dropout(self.dropout)
            self.hidfc = nn.Linear(graph_hidden_size, n_classes)
            if self.att_type == 'concat_subsequently':
                if self.use_residue:
                    self.smax_fc = nn.Linear((D_g+graph_hidden_size)*len(self.modals), n_classes)
                else:
                    self.smax_fc = nn.Linear((graph_hidden_size)*len(self.modals), n_classes)
            elif self.att_type == 'concat_DHT':
                if self.use_residue:
                    self.smax_fc = nn.Linear((D_g+graph_hidden_size*2)*len(self.modals), n_classes)
                else:
                    # print("len(self.modals)", len(self.modals))  # 3
                    # print("graph_hidden_size", graph_hidden_size)  # 1024
                    # print("n_classes", n_classes)  # 6
                    self.smax_fc = nn.Linear((graph_hidden_size*2)*len(self.modals), n_classes)

            elif self.att_type == 'gated':
                if len(self.modals) == 3:
                    print("len(self.modals)", len(self.modals))
                    print("graph_hidden_size", len(graph_hidden_size))
                    self.smax_fc = nn.Linear(100*len(self.modals), graph_hidden_size)
                else:
                    self.smax_fc = nn.Linear(100, graph_hidden_size)
            else:
                self.smax_fc = nn.Linear(D_g+graph_hidden_size*len(self.modals), graph_hidden_size)

        self.all_output_layer = nn.Linear(hidden_dim, n_classes)

    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len, epoch=None):
        # print(qmask.shape)
        # print("textf", textf.shape)  # ([110, 16, 1024])
        # print("visuf", visuf.shape)  # ([110, 16, 342])
        # print("acouf", acouf.shape)  # ([110, 16, 1582])
      
        spk_idx = torch.argmax(qmask, -1)
        spk_idx = torch.argmax(qmask.permute(1, 0, 2), -1)
        origin_spk_idx = spk_idx
      
        if self.n_speakers == 2:
          
            for i, x in enumerate(dia_len):
                """
                ??????
                """
                spk_idx[i, x:] = (2*torch.ones(origin_spk_idx[i].size(0)-x)).int().cuda()
        if self.n_speakers == 9:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (9*torch.ones(origin_spk_idx[i].size(0)-x)).int().cuda()
       
        spk_embeddings = self.speaker_embeddings(spk_idx)

       
        textf = self.textf_input(textf.permute(1, 2, 0)).transpose(1, 2)
        acouf = self.acouf_input(acouf.permute(1, 2, 0)).transpose(1, 2)
        visuf = self.visuf_input(visuf.permute(1, 2, 0)).transpose(1, 2)
        # print("text", textf.shape)
        # print("acouf", acouf.shape)
        # print("vis", visuf.shape)

        t_transformer_out = self.t_out(textf, acouf, visuf, u_mask, spk_embeddings)
        a_transformer_out = self.a_out(acouf, textf, visuf, u_mask, spk_embeddings)
        v_transformer_out = self.v_out(visuf, textf, acouf, u_mask, spk_embeddings)

        # GCN
        # print("a_transformer_out", a_transformer_out.shape)  # ([16, 83, 1024])
        # print("v_transformer_out", v_transformer_out.shape)  #
        # print("t_transformer_out", t_transformer_out.shape)  #
        features_a = simple_batch_graphify(a_transformer_out.permute(1, 0, 2), dia_len, False)
        features_v = simple_batch_graphify(v_transformer_out.permute(1, 0, 2), dia_len, False)
        features_l = simple_batch_graphify(t_transformer_out.permute(1, 0, 2), dia_len, False)
        # print("features_a", features_a.shape)  # ([758, 1024])
        # print("features_v", features_v.shape)  #
        # print("features_l", features_l.shape)  #
        # print("dia_len", len(dia_len))  # 16
        # print("qmask", qmask.shape)  # ([62, 16, 2])
        # print("epoch", epoch)  #

        emotions_feat = self.graph_model(features_a, features_v, features_l, dia_len, qmask, epoch)
        emotions_feat = self.dropout_(emotions_feat)
        emotions_feat = nn.ReLU()(emotions_feat)
    
        # print("emotions_feat", emotions_feat.shape)  # ([782, 6144])
        # print("self.smax_fc(emotions_feat)", self.smax_fc(emotions_feat).shape)  # ([736, 6])
        log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
        prob = F.softmax(self.smax_fc(emotions_feat), 1)

     
        all_transformer_out = self.last_gate(t_transformer_out, a_transformer_out, v_transformer_out)

        # Emotion Classifier
       
        t_final_out = self.t_output_layer(t_transformer_out)
        a_final_out = self.a_output_layer(a_transformer_out)
        v_final_out = self.v_output_layer(v_transformer_out)
        all_final_out = self.all_output_layer(all_transformer_out)
        t_final_out_1 = self.t_output_layer(features_l)
        a_final_out_1 = self.a_output_layer(features_a)
        v_final_out_1 = self.v_output_layer(features_v)
        # print("t_final_out", t_final_out.shape)  # ([16, 74, 6])
        # print("t_final_out_1", t_final_out_1.shape)  # ([758, 6])

       
      
        t_log_prob = F.log_softmax(t_final_out, 2)  # 2->1
        a_log_prob = F.log_softmax(a_final_out, 2)
        v_log_prob = F.log_softmax(v_final_out, 2)
        t_log_prob_1 = F.log_softmax(t_final_out_1, 1)  # 2->1
        a_log_prob_1 = F.log_softmax(a_final_out_1, 1)
        v_log_prob_1 = F.log_softmax(v_final_out_1, 1)

        all_log_prob = F.log_softmax(all_final_out, 2)
        all_prob = F.softmax(all_final_out, 2)

     
        kl_t_log_prob = F.log_softmax(t_final_out /self.temp, 2)
        kl_a_log_prob = F.log_softmax(a_final_out /self.temp, 2)
        kl_v_log_prob = F.log_softmax(v_final_out /self.temp, 2)
        kl_t_log_prob_1 = F.log_softmax(t_final_out_1 / self.temp, 1)
        kl_a_log_prob_1 = F.log_softmax(a_final_out_1 / self.temp, 1)
        kl_v_log_prob_1 = F.log_softmax(v_final_out_1 / self.temp, 1)

        kl_all_prob = F.softmax(all_final_out /self.temp, 2)
        # print("all_final_out", all_final_out.shape)  # ([16, 74, 6])
        # print("kl_all_prob", kl_all_prob.shape)  # ([16, 74, 6])
        # print("F.softmax(self.smax_fc(emotions_feat) /self.temp, 1)", F.softmax(self.smax_fc(emotions_feat) /self.temp, 1).shape) 
        kl_all_prob_1 = F.softmax(self.smax_fc(emotions_feat) /self.temp, 1)
        # print("emotions_feat", emotions_feat.shape)  # ([758, 6144])
        # print("self.smax_fc(emotions_feat)", self.smax_fc(emotions_feat).shape)  # ([758, 6])
        # print("kl_t_log_prob", kl_t_log_prob.shape)  # ([16, 74, 6])
        # print("kl_t_log_prob_1", kl_t_log_prob_1.shape)  # ([758, 6])
        # print("kl_log_prob_1", kl_all_prob_1.shape)  # ([758, 6])
        # print("emotions_feat", emotions_feat.shape)
        # print("self.smax_fc(emotions_feat)", self.smax_fc(emotions_feat).shape)  # ([764, 6])
        # print("kl_all_prob", kl_all_prob.shape)  # ([16, 94, 6])
        # print("kl_all_prob_1", kl_all_prob_1.shape)  # ([761, 6])
       
        # return t_log_prob, a_log_prob, v_log_prob, all_log_prob, all_prob, \
        #        kl_t_log_prob, kl_a_log_prob, kl_v_log_prob, kl_all_prob
        return t_log_prob, a_log_prob, v_log_prob, log_prob, prob, \
               kl_t_log_prob_1, kl_a_log_prob_1, kl_v_log_prob_1, kl_all_prob_1

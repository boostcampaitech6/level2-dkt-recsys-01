import torch 
import torch.nn as nn
import torch.nn.functional as F 
from ..model import ModelBase

class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self,ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))
    
class LastQuery(ModelBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        max_seq_len: float = 20,
        device: str = 'cpu',
        **kwargs
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
        self.device = device
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
    
        ##### ----- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ----- #####
        # self.embedding_position = nn.Embedding(self.max_seq_len, self.hidden_dim)
        # 기존 keetar님 솔루션에서는 Positional Embedding은 사용되지 않습니다
        # 하지만 사용 여부는 자유롭게 결정해주세요 :)
        # self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim)
        ##### ----- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ----- #####
        
        # Encoder
        self.query = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.n_heads)
        self.mask = None # last query에서는 필요가 없지만 수정을 고려하여서 넣어둠
        self.ffn = Feed_Forward_block(self.hidden_dim)

        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            self.n_layers,
            batch_first=True)

        self.activation = nn.Sigmoid()

    def get_pos(self, seq_len):
        # use sine positional embeddinds
        return torch.arange(seq_len).unsqueeze(0)

    # lstm의 hidden state와 cell state를 초기화 해주는 코드임
    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)


    def forward(self, test, question, tag, correct, mask, interaction):
        # ModelBase 코드로 임베딩 
        embed, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction)
        seq_len = interaction.size(1)
        
        # Positional Embedding
        # last query에서는 positional embedding을 하지 않음
        # position = self.get_pos(seq_len).to('cuda')
        # embed_pos = self.embedding_position(position)
        # embed = embed + embed_pos
        
        ####################### ENCODER #####################
        q = self.query(embed)[:, -1:, :].permute(1, 0, 2)
        k = self.key(embed).permute(1, 0, 2)
        v = self.value(embed).permute(1, 0, 2)

        ## attention
        # last query only
        out, _ = self.attn(q, k, v)

        ## residual + layer norm
        out = out.permute(1, 0, 2)

        out = embed + out
        out = self.ln1(out)

        ## feed forward network
        out = self.ffn(out)

        ## residual + layer norm
        out = embed + out
        out = self.ln2(out)

        ###################### LSTM #####################
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(out, hidden)

        ###################### DNN #####################
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        
        return out
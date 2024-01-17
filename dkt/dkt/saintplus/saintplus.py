import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FFN(nn.Module):
    def __init__(self, d_ffn, d_model, dropout=0.1):
        super(FFN, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ffn) #[batch, seq_len, ffn_dim]
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(d_ffn, d_model) #[batch, seq_len, d_model]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.linear_2(x)
        return self.dropout(x)

class SaintPlus(nn.Module):
    def __init__(self, args, drop_out=0.1):
        super(SaintPlus, self).__init__()
        self.args = args
        hidden_dim = args.hidden_dim
        n_questions = args.n_questions
        num_heads = args.n_heads
        self.seq_len = args.max_seq_len
        
        # Embeddings
        # hd: Hidden dimension, intd: Intermediate hidden dimension
        hd, intd = hidden_dim, hidden_dim // 3
        self.embedding_interaction = nn.Embedding(3, intd) # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_test = nn.Embedding(args.n_tests + 1, intd)
        self.embedding_question = nn.Embedding(n_questions + 1, intd)
        self.embedding_tag = nn.Embedding(args.n_tags + 1, intd)
        self.pos_emb = nn.Embedding(self.seq_len, hd)

        self.emb_dense1 = nn.Linear(4*intd, hd)
        self.emb_dense2 = nn.Linear(4*intd, hd)

        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=args.n_layers, 
                                          num_decoder_layers=args.n_layers, dropout=drop_out)
        self.layer_norm = nn.LayerNorm(hd)
        self.FFN = FFN(hd, hd, dropout=drop_out)
        self.final_layer = nn.Linear(hd, 1)
    
    def forward(self, test, question, tag, correct, mask, interaction):
        device = self.args.device
        batch_size = interaction.size(0)
        seq_len = self.seq_len

       # Embedding
        embed_interaction = self.embedding_interaction(interaction.int())
        embed_test = self.embedding_test(test.int())
        embed_question = self.embedding_question(question.int())
        embed_tag = self.embedding_tag(tag.int())


        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            dim=2,
        )
        #encoder_val = torch.cat((embed_question, embed_tag, ), axis=-1)
        encoder_val = self.emb_dense1(embed)
        #decoder_val = torch.cat((embed_interaction, answer_emb), axis=-1)
        decoder_val = self.emb_dense2(embed)
        
        pos = torch.arange(seq_len).unsqueeze(0).to(device)
        pos_emb = self.pos_emb(pos)
        encoder_val += pos_emb
        decoder_val += pos_emb

        over_head_mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1).astype('bool'))
        over_head_mask = over_head_mask.to(device)

        encoder_val = encoder_val.permute(1, 0, 2)
        decoder_val = decoder_val.permute(1, 0, 2)
        decoder_val = self.transformer(encoder_val, decoder_val, src_mask=over_head_mask, tgt_mask=over_head_mask, memory_mask=over_head_mask)

        decoder_val = self.layer_norm(decoder_val)
        decoder_val = decoder_val.permute(1, 0, 2)
        
        final_out = self.FFN(decoder_val)
        final_out = self.layer_norm(final_out + decoder_val)
        final_out = self.final_layer(final_out)
        #final_out = torch.sigmoid(final_out)
        return final_out.squeeze(-1)

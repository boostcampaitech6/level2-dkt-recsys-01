import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import copy
from .model import ModelBase 


class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff , out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff , out_features=dim_ff)

    def forward(self,ffn_in):
        return  self.layer2(   F.relu( self.layer1(ffn_in) )   )
        

class Encoder_block(nn.Module):
    """
    M = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    O = SkipConct(FFN(LayerNorm(M)))
    """

    def __init__(self , hidden_dim, heads_en, total_ex ,total_cat, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.embd_ex =   nn.Embedding( total_ex , embedding_dim = hidden_dim )                   # embedings  q,k,v = E = exercise ID embedding, category embedding, and positionembedding.
        self.embd_cat =  nn.Embedding( total_cat, embedding_dim = hidden_dim )
        self.embd_pos   = nn.Embedding(  seq_len , embedding_dim = hidden_dim )                  #positional embedding

        self.multi_en = nn.MultiheadAttention( embed_dim= hidden_dim, num_heads= heads_en,  )     # multihead attention    ## todo add dropout, LayerNORM
        self.ffn_en = Feed_Forward_block( hidden_dim )                                            # feedforward block     ## todo dropout, LayerNorm
        self.layer_norm1 = nn.LayerNorm( hidden_dim )
        self.layer_norm2 = nn.LayerNorm( hidden_dim )


    def forward(self, in_ex, in_cat, first_block=True):

        ## todo create a positional encoding ( two options numeric, sine)
        if first_block:
            in_ex = self.embd_ex( in_ex )
            in_cat = self.embd_cat( in_cat )
            #in_pos = self.embd_pos( in_pos )
            #combining the embedings
            out = in_ex + in_cat #+ in_pos                      # (b,n,d)
        else:
            out = in_ex
        
        in_pos = get_pos(self.seq_len)
        in_pos = self.embd_pos( in_pos )
        out = out + in_pos                                      # Applying positional embedding

        out = out.permute(1,0,2)                                # (n,b,d)  
        print('pre multi', out.shape )
        
        #Multihead attention                            
        n,_,_ = out.shape
        out = self.layer_norm1( out )                           # Layer norm
        skip_out = out 
        out, attn_wt = self.multi_en( out , out , out ,
                                attn_mask=get_mask(seq_len=n))  # attention mask upper triangular
        out = out + skip_out                                    # skip connection

        #feed forward
        out = out.permute(1,0,2)                                # (b,n,d)
        out = self.layer_norm2( out )                           # Layer norm 
        skip_out = out
        out = self.ffn_en( out )
        out = out + skip_out                                    # skip connection

        return out


class Decoder_block(nn.Module):
    """
    M1 = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    M2 = SkipConct(Multihead(LayerNorm(M1;O;O)))
    L = SkipConct(FFN(LayerNorm(M2)))
    """

    def __init__(self,hidden_dim ,total_in, heads_de,seq_len  ):
        super().__init__()
        self.seq_len    = seq_len
        self.embd_in    = nn.Embedding(  total_in , embedding_dim = hidden_dim )                  #interaction embedding
        self.embd_pos   = nn.Embedding(  seq_len , embedding_dim = hidden_dim )                  #positional embedding
        self.multi_de1  = nn.MultiheadAttention( embed_dim= hidden_dim, num_heads= heads_de  )  # M1 multihead for interaction embedding as q k v
        self.multi_de2  = nn.MultiheadAttention( embed_dim= hidden_dim, num_heads= heads_de  )  # M2 multihead for M1 out, encoder out, encoder out as q k v
        self.ffn_en     = Feed_Forward_block( hidden_dim )                                         # feed forward layer

        self.layer_norm1 = nn.LayerNorm( hidden_dim )
        self.layer_norm2 = nn.LayerNorm( hidden_dim )
        self.layer_norm3 = nn.LayerNorm( hidden_dim )


    def forward(self, in_in, en_out,first_block=True):

         ## todo create a positional encoding ( two options numeric, sine)
        if first_block:
            in_in = self.embd_in( in_in )

            #combining the embedings
            out = in_in #+ in_cat #+ in_pos                         # (b,n,d)
        else:
            out = in_in

        in_pos = get_pos(self.seq_len)
        in_pos = self.embd_pos( in_pos )
        out = out + in_pos                                          # Applying positional embedding

        out = out.permute(1,0,2)                                    # (n,b,d)# print('pre multi', out.shape )
        n,_,_ = out.shape

        #Multihead attention M1                                     ## todo verify if E to passed as q,k,v
        out = self.layer_norm1( out )
        skip_out = out
        out, attn_wt = self.multi_de1( out , out , out, 
                                     attn_mask=get_mask(seq_len=n)) # attention mask upper triangular
        out = skip_out + out                                        # skip connection

        #Multihead attention M2                                     ## todo verify if E to passed as q,k,v
        en_out = en_out.permute(1,0,2)                              # (b,n,d)-->(n,b,d)
        en_out = self.layer_norm2( en_out )
        skip_out = out
        out, attn_wt = self.multi_de2( out , en_out , en_out,
                                    attn_mask=get_mask(seq_len=n))  # attention mask upper triangular
        out = out + skip_out

        #feed forward
        out = out.permute(1,0,2)                                    # (b,n,d)
        out = self.layer_norm3( out )                               # Layer norm 
        skip_out = out
        out = self.ffn_en( out )                                    
        out = out + skip_out                                        # skip connection

        return out

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_mask(seq_len):
    ##todo add this to device
    return torch.from_numpy( np.triu(np.ones((seq_len ,seq_len)), k=1).astype('bool'))

def get_pos(seq_len):
    # use sine positional embeddinds
    return torch.arange( seq_len ).unsqueeze(0) 

class Saint(ModelBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        seq_len: float = 20,
        **kwargs
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
        self.n_heads = n_heads
        self.drop_out = drop_out

        self.num_en = num_en = 6
        self.num_de = num_de = 6
        heads_en = heads_de = 8

        self.encoder = get_clones( Encoder_block(hidden_dim, heads_en , n_tests ,n_tags,seq_len) , num_en)
        self.decoder = get_clones( Decoder_block(hidden_dim ,n_questions, heads_de,seq_len), num_de)

        self.out = nn.Linear(in_features= hidden_dim , out_features=1)
    
    def forward(self,in_ex,  in_in, in_cat):
        
        ## pass through each of the encoder blocks in sequence
        first_block = True
        for x in range(self.num_en):
            if x>=1:
                first_block = False
            in_ex = self.encoder[x]( in_ex, in_cat ,first_block=first_block)
            in_cat = in_ex                                  # passing same output as q,k,v to next encoder block

        
        ## pass through each decoder blocks in sequence
        first_block = True
        for x in range(self.num_de):
            if x>=1:
                first_block = False
            in_in = self.decoder[x]( in_in , en_out= in_ex, first_block=first_block )

        ## Output layer
        in_in = torch.sigmoid( self.out( in_in ) )
        return in_in


## forward prop on dummy data

# seq_len = 100
# total_ex = 1538
# total_cat = 913
# total_in = 9455


# def random_data(bs, seq_len , total_ex, total_cat, total_in = 2):
#     ex = torch.randint( 0 , total_ex ,(bs , seq_len) )
#     cat = torch.randint( 0 , total_cat ,(bs , seq_len) )
#     de = torch.randint( 0 , total_in ,(bs , seq_len) )
#     return ex,cat, de


# in_ex, in_cat, in_de = random_data(64, seq_len , total_ex, total_cat, total_in)


# model = saint(hidden_dim=128,
#             num_en=6,
#             num_de=6,
#             heads_en=8,
#             heads_de=8,
#             total_ex=total_ex,
#             total_cat=total_cat,
#             total_in=total_in,
#             seq_len=seq_len
#             )

# outs = model(in_ex, in_cat, in_de)

# print(outs.shape)
# print(outs)
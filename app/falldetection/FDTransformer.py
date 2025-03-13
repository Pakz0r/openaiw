import torch
from torch import nn
from math import log

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, d_model, 2) * (log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args in input:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = torch.transpose(x, 0, 1) # Batch first -> batch second
        x = x + self.pe[:x.size(0)]
        x = torch.transpose(x, 0, 1) # Batch second -> batch first
        return self.dropout(x)


class FDTransformer(nn.Module):
    def __init__(self, dropout=0.1, embed_dim=38, input_dim=38, num_heads=2, num_encoder_layers=1, num_decoder_layers=1):
        super(FDTransformer, self).__init__()
        print("Loading FDTransformer model...")
        self.pos_embedding = PositionalEncoding(input_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.linear = nn.Linear(embed_dim, input_dim)
        self.sigmoid = nn.Sigmoid()
        
    def get_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, device):
        src = self.pos_embedding(src)                               #positional with sin/cos
        src = self.transformer_encoder(src)
        
        tgt_mask = self.get_mask(tgt.size(1)).to(device)
        tgt = self.pos_embedding(tgt)

        output = self.transformer_decoder(tgt = tgt, memory = src, tgt_mask = tgt_mask)
        #tgt_mask is to avoid looking at the future tokens (the ones on the right)
        #tgt_key_padding_mask = tgt_key_padding_mask, # to avoid working on padding
        #memory_key_padding_mask = src_key_padding_mask # avoid looking on padding of the src
    
        output = self.linear(output)
        output = self.sigmoid(output)
        return output
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CCAPNet(nn.Module):
    """[C]anadian [C]ivic [A]ddress [P]arsing neural [net]work.""" 
    def __init__(self, x_vocab_size, y_vocab_size):
        super().__init__()
        n_x_tokens = x_vocab_size + 1 # final index is a pad token
        n_y_tokens = y_vocab_size + 1
        
        self.embed_size = 4
        self.hidden_size = 16 # needed in forward()
        
        self.embed = nn.Embedding(n_x_tokens, self.embed_size, padding_idx=x_vocab_size, max_norm=1.)
        self.gru_bi = nn.GRU(self.embed_size, self.hidden_size, batch_first=True, bidirectional=True)
        
        self.gru_f1 = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.gru_b1 = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        
        self.gru_f2 = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.gru_b2 = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        self.linear = nn.Linear(2*self.hidden_size, n_y_tokens)

    
    def forward(self, x, lengths):
        x = self.embed(x)
        
        # pack sequence for GRU
        #
        # this context manager is needed because:
        #   - pack_padded_sequence requires 'lengths' to be a CPU tensor
        #   - if creating tensors on GPU by default, pack_padded_sequence breaks
        with torch.device('cpu'):
           x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            
        x, h = self.gru_bi(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
       
        xf0, xb0 = x[:,:,:self.hidden_size], x[:,:,self.hidden_size:]
        hf0, hb0 = h[:1,:,:], h[1:,:,:]
        
        with torch.device('cpu'):
            xf = pack_padded_sequence(xf0, lengths, batch_first=True, enforce_sorted=False)
            xb = pack_padded_sequence(xb0, lengths, batch_first=True, enforce_sorted=False)

        xf, hf1 = self.gru_f1(xf, hf0)
        xb, hb1 = self.gru_b1(xb, hb0)

        xf, _ = pad_packed_sequence(xf, batch_first=True)
        xb, _ = pad_packed_sequence(xb, batch_first=True)
        
        # residual connection
        xf1 = xf + xf0
        xb1 = xb + xb0

        with torch.device('cpu'):
            xf = pack_padded_sequence(xf1, lengths, batch_first=True, enforce_sorted=False)
            xb = pack_padded_sequence(xb1, lengths, batch_first=True, enforce_sorted=False)

        xf, _ = self.gru_f2(xf, hf1)
        xb, _ = self.gru_b2(xb, hb1)

        xf, _ = pad_packed_sequence(xf, batch_first=True)
        xb, _ = pad_packed_sequence(xb, batch_first=True)

        xf2 = xf + xf1
        xb2 = xb + xb1

        x = torch.cat([xf2, xb2], dim=2)

        logits = self.linear(x)
        
        return logits

"""Validation-search models. Keep v1 architectures unchanged for reproducibility."""
import torch
from torch import nn

def reverse_valid_prefix(x,lengths):
    """Reverse real tokens only; keep padding positions fixed (B,T,D tensor)."""
    pos=torch.arange(x.size(1),device=x.device)[None,:].expand(x.size(0),-1)
    source=torch.where(pos<lengths[:,None],lengths[:,None]-1-pos,pos)
    return x.gather(1,source[:,:,None].expand(-1,-1,x.size(2)))

def masked_mean_max(states,lengths):
    valid=torch.arange(states.size(1),device=states.device)[None,:]<lengths[:,None]
    mean=(states*valid[:,:,None]).sum(1)/lengths[:,None]
    maximum=states.masked_fill(~valid[:,:,None],-1e4).amax(1)
    return torch.cat([mean,maximum],dim=1)

class PooledLSTM(nn.Module):
    def __init__(self,vocab_size,emb=64,hidden=64,dropout=.3,bidirectional=False,pooling='meanmax',**unused):
        super().__init__();self.bidirectional=bidirectional;self.pooling=pooling
        self.embedding=nn.Embedding(vocab_size,emb,padding_idx=0)
        self.embedding_dropout=nn.Dropout(.10)
        self.forward_lstm=nn.LSTM(emb,hidden,batch_first=True)
        if bidirectional:self.backward_lstm=nn.LSTM(emb,hidden,batch_first=True)
        features=hidden*(2 if bidirectional else 1)*(2 if pooling=='meanmax' else 1)
        self.dropout=nn.Dropout(dropout);self.classifier=nn.Linear(features,3)
    def forward(self,tokens,lengths):
        embedded=self.embedding_dropout(self.embedding(tokens))
        forward,_=self.forward_lstm(embedded)
        if self.bidirectional:
            backward,_=self.backward_lstm(reverse_valid_prefix(embedded,lengths))
            backward=reverse_valid_prefix(backward,lengths)
            states=torch.cat([forward,backward],dim=2)
        else:states=forward
        if self.pooling=='meanmax':summary=masked_mean_max(states,lengths)
        elif not self.bidirectional:
            summary=states[torch.arange(tokens.size(0),device=tokens.device),lengths-1]
        else:
            # Terminal states of the two directions, not a padded final position.
            summary=torch.cat([forward[torch.arange(tokens.size(0),device=tokens.device),lengths-1],backward[:,0]],dim=1)
        return self.classifier(self.dropout(summary))

class SearchCNN(nn.Module):
    def __init__(self,vocab_size,emb=64,hidden=64,dropout=.3,**unused):
        super().__init__();self.embedding=nn.Embedding(vocab_size,emb,padding_idx=0)
        self.embedding_dropout=nn.Dropout(.10)
        self.convs=nn.ModuleList([nn.Conv1d(emb,hidden,k) for k in [3,4,5]])
        self.dropout=nn.Dropout(dropout);self.classifier=nn.Linear(hidden*3,3)
    def forward(self,tokens,lengths):
        x=self.embedding_dropout(self.embedding(tokens)).transpose(1,2);pooled=[]
        for conv in self.convs:
            h=torch.relu(conv(x));valid=(lengths-conv.kernel_size[0]+1).clamp(min=1)
            bad=torch.arange(h.size(2),device=h.device)[None,:]>=valid[:,None]
            pooled.append(h.masked_fill(bad[:,None,:],-1e4).amax(2))
        return self.classifier(self.dropout(torch.cat(pooled,dim=1)))

def make_model(config,vocab_size):
    cls=SearchCNN if config['architecture']=='cnn' else PooledLSTM
    return cls(vocab_size,**config)

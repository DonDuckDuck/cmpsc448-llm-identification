"""Small text CNN and LSTM; no pretrained LLM or transformer classifier."""
import torch
from torch import nn

class TextCNN(nn.Module):
    def __init__(self,vocab_size,emb=64,hidden=64,classes=3,dropout=.3):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,emb,padding_idx=0)
        self.convs=nn.ModuleList([nn.Conv1d(emb,hidden,k) for k in (3,4,5)])
        self.dropout=nn.Dropout(dropout)
        self.classifier=nn.Linear(hidden*3,classes)
    def forward(self,tokens,lengths):
        x=self.embedding(tokens).transpose(1,2); pools=[]
        for conv in self.convs:
            h=torch.relu(conv(x))
            # Only pool windows within the true sequence, not trailing padding.
            valid=(lengths-conv.kernel_size[0]+1).clamp(min=1)
            positions=torch.arange(h.size(2),device=h.device)[None,:]
            h=h.masked_fill((positions>=valid[:,None])[:,None,:],-1e4)
            pools.append(h.amax(dim=2))
        return self.classifier(self.dropout(torch.cat(pools,dim=1)))

class TextLSTM(nn.Module):
    def __init__(self,vocab_size,emb=64,hidden=64,classes=3,dropout=.3):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,emb,padding_idx=0)
        self.rnn=nn.LSTM(emb,hidden,batch_first=True)
        self.dropout=nn.Dropout(dropout)
        self.classifier=nn.Linear(hidden,classes)
    def forward(self,tokens,lengths):
        sequence,_=self.rnn(self.embedding(tokens))
        # Unidirectional recurrence: later padding cannot affect the true final state.
        last=sequence[torch.arange(tokens.size(0),device=tokens.device),lengths-1]
        return self.classifier(self.dropout(last))

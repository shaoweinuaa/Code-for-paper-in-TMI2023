import torch
import torch.nn as nn
import torch.nn.functional as F

def xavier_init(m):  
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class fusion(nn.Module):
    def __init__(self, config):
        super(fusion, self).__init__()

        self.attention = nn.Sequential(
            nn.Linear(config ,  128),  
            nn.Tanh(),
            nn.Linear(128 , 1) 
        )
        self.attention.apply(xavier_init)

    def forward(self,x,y,z):
        x=x.view(x.shape[0],1,-1)
        y=y.view(y.shape[0],1,-1)
        z = z.view(y.shape[0], 1, -1)
        featurecom = torch.cat([x,y,z],1)
        A = self.attention(featurecom)
        A = torch.transpose(A,2,1)
        A=F.softmax(A, dim=2)
        M = torch.bmm(A, featurecom)
        M=M.view(M.shape[0],-1)
        return M




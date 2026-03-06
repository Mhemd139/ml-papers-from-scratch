import torch.nn as nn
import torch

def scaled_dot_product_attention(q,k,v,mask): 
    tensored_QK = torch.matmul(q,k.transpose(-2, -1))
    sqrt_dk = torch.sqrt(torch.tensor(float(k.size(-1))))
    matrix = tensored_QK/sqrt_dk
    masked = matrix.masked_fill(mask == True, float('-inf'))
    m = torch.softmax(masked,dim = -1)
    result = torch.matmul(m,v)
    return result

class MultiHeadAttention(nn.Module) :
    def __init__(self , d_model, head_num):
        super().__init__()
        d_k = d_model//head_num
        self.d_model = d_model
        self.head_num = head_num
        self.d_k = d_k
        self.WQ = nn.Linear(d_model,d_model)
        self.WK = nn.Linear(d_model,d_model)
        self.WV = nn.Linear(d_model,d_model)
        self.WO = nn.Linear(d_model,d_model)

    
        pass
    def forward(self,x, mask):
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)
        batch_size, seq_len, _ = x.size()

        Q= Q.reshape(batch_size,seq_len,self.head_num,self.d_k)
        Q = Q.permute(0, 2, 1, 3)

        K = K.reshape(batch_size,seq_len,self.head_num,self.d_k)
        K = K.permute(0, 2, 1, 3)

        V = V.reshape(batch_size,seq_len,self.head_num,self.d_k)
        V = V.permute(0, 2, 1, 3)
        heads = scaled_dot_product_attention(Q,K,V,mask)
        heads = heads.permute(0, 2, 1, 3)
        heads = heads.contiguous().view(batch_size,seq_len,self.d_model)
        WO = self.WO(heads)
        return WO


rnd_tensor = torch.rand(2,5,512)
model = MultiHeadAttention(512,8)
mask = torch.zeros(2,8,5,5).bool()

output = model(rnd_tensor, mask)
print(output.shape)
import einops
from tqdm.notebook import tqdm

from torchsummary import summary 

import torch
import torch.nn as nn 
import torchvision
import torch.optim as optim
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

patch_size= 16
latent_size= 768
n_channels= 3
num_heads=12
num_encoders= 12
dropout= 0.1
num_classes = 43
size= 224

epochs = 10
base_lr= 10e-3
weight_decay= 0.03
batch_size= 1

class InputEmbedding(nn.Module):
    def __init__(self, patch_size=patch_size, n_channels= n_channels, device=device, latent_size=latent_size, batch_size=batch_size):
        super(InputEmbedding,self).__init__()
        self.latent_size= latent_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.device= device
        self.batch_size= batch_size
        self.input_size= self.patch_size*self.patch_size*self.n_channels

        self.linearProjection = nn.Linear(self.input_size, self.latent_size)

        self.class_token = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(self.device)

        self.pos_embedding = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(self.device)

    def forward(self,input_data):
        input_data= input_data.to(self.device)

        patches = einops.rearrange(
            input_data, 'b c (h h1) (w w1) -> b (h w) (h1 w1 c)', h1=self.patch_size, w1= self.patch_size)
        
        #print(input_data.size())
        #print(patches.size())

        linear_projection= self.linearProjection(patches).to(self.device)
        b, n, _ = linear_projection.shape
        #print(self.class_token.size())
        #print(linear_projection.size())
        linear_projection = torch.cat((self.class_token, linear_projection), dim=1)
        pos_embed = einops.repeat(self.pos_embedding, 'b 1 d -> b m d', m= n+1)
        
        linear_projection += pos_embed

        return linear_projection


class EncoderBlock(nn.Module):
    def __init__(self, latent_size=latent_size, num_heads= num_heads, device=device, dropout= dropout):
        super(EncoderBlock,self).__init__()

        self.latent_size= latent_size
        self.num_heads= num_heads
        self.device = device
        self.dropout= dropout

        self.norm = nn.LayerNorm(self.latent_size)

        self.multihead = nn.MultiheadAttention(
            self.latent_size, self.num_heads, dropout = self.dropout
        )
        
        self.enc_MLP= nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size*4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_size*4, self.latent_size),
            nn.Dropout(self.dropout),
        )
        
    def forward(self, embedded_patches):
        firstnorm_out= self.norm(embedded_patches)
        attention_out = self.multihead(firstnorm_out, firstnorm_out, firstnorm_out)[0]

        first_added= attention_out + embedded_patches

        secondnorm_out = self.norm(first_added)
        ff_out= self.enc_MLP(secondnorm_out)


        return ff_out + first_added
    
class VisionTransformer(nn.Module):
    def __init__(self,input_size= size, num_encoders=num_encoders, latent_size= latent_size, device= device, num_classes= num_classes, dropout=dropout):
        super(VisionTransformer,self).__init__()
        self.num_encoders= num_encoders
        self.latent_size= latent_size
        self.device = device
        self.num_classes = num_classes
        self.dropout= dropout
        self.input_size= input_size

        self.embedding = InputEmbedding()

        self.encStack = nn.ModuleList([EncoderBlock() for i in range(self.num_encoders)])

        self.MLP_head= nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, self.latent_size),
            nn.Linear(self.latent_size, self.num_classes)
        )

    def forward(self, test_input):
        enc_output = self.embedding(test_input)

        for enc_layer in self.encStack:
            enc_output = enc_layer(enc_output)

        cls_token_embed= enc_output[:, 0]

        return self.MLP_head(cls_token_embed)


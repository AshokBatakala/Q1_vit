
# -----------------------Imports-----------------------

import torch
import torch.nn.functional as f
import matplotlib.pyplot as plt
from collections import OrderedDict

from torch import nn
from torch import tensor
import torch.optim as optim
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor,transforms
from torch.utils.data import DataLoader



# -----------------------mlp-----------------------
class MLP_1(nn.Module):
    """mlp with 1 hidden layer"""

    def __init__(self, in_dim, hidden_dim,out_dim,droupout_p):
        super(MLP_1,self).__init__()
                
        self.linear1 = nn.Linear(in_features=in_dim,out_features=hidden_dim)
        self.activation = nn.GELU()
        self.droupout_1=nn.Dropout(p=droupout_p)
        self.linear2 = nn.Linear(in_features=hidden_dim,out_features=out_dim)
        self.droupout_2 = nn.Dropout(p=droupout_p)
        
        
    def forward(self,x: tensor) -> tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.droupout_1(x)
        x = self.linear2(x)
        x = self.droupout_2(x)

        return x




# -----------------------tokenizer-----------------------

class Tokenizer(nn.Module):
    """image to token conversion"""
    
    def __init__(self, token_dim: int, patch_size: int,stride = None) :
        
        super().__init__()

        if stride is None:
            stride = patch_size # non-overlapping patches

        self.input_to_tokens = nn.Conv2d(in_channels=3,out_channels=token_dim,kernel_size=patch_size,stride=patch_size)
        
    def forward(self, input:tensor)->tensor:
        """Returns token in shape of (batch_size, n_token, token_dim)"""
        
        output = self.input_to_tokens(input)                    # each small image patch is converted to a log. (imagine pile of logs)
        output = torch.flatten(output,start_dim=-2,end_dim=-1) # flatten the last two dimensions only,( lay those logs flat on ground ) 
        output = output.transpose(-2,-1)
        
        return output
    
class ClasstokenConcatenator(nn.Module):
    """Concatenate the Class with set of tokens"""
    
    def __init__(self, token_dim: int) -> None:
        
        super().__init__()
        self.class_token = nn.Parameter(torch.zeros(token_dim))
        
    def forward(self, input:tensor) -> tensor:

        class_token = self.class_token.expand(len(input),1,-1)
        output = torch.cat((class_token,input),dim=1)
                    
        return output
    
class PositionEmbeddingAdder(nn.Module):
    """adds learnable parameters to token for position embedding"""
    
    def __init__(self, n_token: int, token_dim: int) -> None:
        super().__init__()
        
        position_embedding = torch.zeros(n_token,token_dim)
        self.position_embedding =  nn.Parameter(position_embedding)
        
    def forward(self, input:tensor)->tensor:
        
        output = input+self.position_embedding
        return output

# -----------------------Attention Module-----------------------
class QueriesKeyValuesExtractor(nn.Module):
    """get queries key value from multi head self attention"""
    
    def __init__(self,token_dim:int,head_dim:int,n_heads:int) -> None:
        super().__init__()
        
        self.head_dim = head_dim
        self.n_heads = n_heads
        queries_key_values_dim = 3*self.n_heads*self.head_dim
        
        self.input_to_queries_key_values = nn.Linear(in_features=token_dim,out_features=queries_key_values_dim,bias = False)
        
        
        
    def forward(self,input: tensor):
        
        
        batch_size,n_token,token_dim = input.shape
        queries_key_values = self.input_to_queries_key_values(input)            #input -> [batch_size, n_tokens, token_dim]
        queries_key_values = queries_key_values.reshape(batch_size,3,self.n_heads,n_token,self.head_dim)
        
        queries, keys, values = queries_key_values.unbind(dim=1)
        
        return queries, keys, values
    
    
def get_attention(queries: tensor, keys: tensor, values: tensor) -> tensor:
        
        
    scale = queries.shape[-1]**(-0.5)
    attention_scores = (queries @  keys.transpose(-1,-2)) * scale
        
    attention_prob = f.softmax(attention_scores,dim=-1)
        
    attention = attention_prob @ values
        
    return attention
        

# -----------------------Multi Head Attention-----------------------
class Multiheadselfattention(nn.Module):
    """Multi head self attention"""
    
    def __init__(self,token_dim: int , head_dim: int , n_heads : int, droupout_p : float) -> None:
        super(Multiheadselfattention,self).__init__()
        
        
        self.query_key_value_extractor = QueriesKeyValuesExtractor(token_dim=token_dim,head_dim=head_dim,n_heads=n_heads)
        self.concatenated_head_dim = n_heads*head_dim
        
        self.attention_to_output = nn.Linear(in_features=self.concatenated_head_dim,out_features=token_dim)
        
        self.output_dropout = nn.Dropout(p=droupout_p)
        
        
    def forward(self, input: tensor) -> tensor:
        
        batch_size, n_tokens, token_dim = input.shape
        querys, keys, values = self.query_key_value_extractor(input)
        
        attention = get_attention(queries=querys,keys=keys,values=values)
        
        attention = attention.transpose(1,2).reshape(batch_size,n_tokens,self.concatenated_head_dim)
        
        output = self.attention_to_output(attention)
        output = self.output_dropout(output)
        
        return output
        
    # -----------------------Transformer Block-----------------------
class TransformerBlock(nn.Module):
    """Transformer Block"""
    
    def __init__(self, token_dim: int, multihead_attention_head_dim: int, multihead_attention_n_heads: int, 
                 multilayer_perceptron_hidden_dim: int, dropout_p: float) -> None:
        super().__init__()
        
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=token_dim)
        self.multi_head_attention = Multiheadselfattention(token_dim=token_dim,head_dim=multihead_attention_head_dim,
                                                           n_heads=multihead_attention_n_heads,droupout_p=dropout_p)
        
        self.layer_norm_2 = nn.LayerNorm(normalized_shape= token_dim)
        
        self.multilayer_perceptron = MLP_1(in_dim=token_dim,hidden_dim=multilayer_perceptron_hidden_dim,
                                                          out_dim=token_dim,droupout_p=dropout_p)
        
    def forward(self, input: tensor) -> tensor:
        """Runs the input through transformer block"""
        
        residual = input
        output = self.layer_norm_1(input)
        output = self.multi_head_attention(output)
        output += residual
        
        residual = output
        output = self.layer_norm_2(output)
        output = self.multilayer_perceptron(output)
        output += residual
        
        return output



# -----------------------Transformer-----------------------

class Transformer(nn.Module):
    """Transformer Encoder"""
    
    def __init__(self, n_attention_layers: int,
                        token_dim: int,
                        multihead_attention_head_dim: int,
                        multihead_attention_n_heads: int,
                        mulitlayer_perceptron_hidden_dim : int,
                        dropout_p : float) :

        super().__init__()
        transformer_blocks =[]

        # layers
        for i in range(1, n_attention_layers+1):

            transformer_block = TransformerBlock(   token_dim=token_dim,
                                                    multihead_attention_head_dim=multihead_attention_head_dim,
                                                    multihead_attention_n_heads=multihead_attention_n_heads,
                                                    multilayer_perceptron_hidden_dim= mulitlayer_perceptron_hidden_dim,
                                                    dropout_p=dropout_p
                                                 )
            transformer_block = (f'transformer_block_{i}',transformer_block)
            transformer_blocks.append(transformer_block)
            
        transformer_blocks = OrderedDict(transformer_blocks)                # ordered dict to preserve the order of the blocks
        self.transformer_blocks = nn.Sequential(transformer_blocks)
        
        
    def forward(self, input: tensor) -> tensor:
        
        output = self.transformer_blocks(input)
        
        return output
            

# -----------------------vision Transfomer-----------------------
class visionTransformer(nn.Module):
    """Vision Transformer"""
    
    def __init__(self,  token_dim: int,
                        patch_size: int,
                        image_size: int,
                        n_attention_layers: int,
                        multihead_attention_head_dim : int,
                        multihead_attention_n_heads: int,
                        multilayer_perceptron_hidden_dim: int,
                        dropout_p: float, n_classes: int,
                        stride_in_tokenization: int = None):
        """
        stride_in_tokenization: if None, then no overlap in patching i.e. stride = patch_size

        """
        super().__init__()
        
        self.tokenizer = Tokenizer( token_dim=token_dim,
                                    patch_size=patch_size,
                                    stride=stride_in_tokenization
                                    )

        self.concat_class_token = ClasstokenConcatenator(token_dim=token_dim)
        n_tokens = (image_size//patch_size)**2 + 1 # +1 for class token
        
        self.add_position_embedding = PositionEmbeddingAdder(n_token=n_tokens,token_dim=token_dim)
        
        self.transformer = Transformer( n_attention_layers=n_attention_layers,
                                        token_dim=token_dim,
                                        multihead_attention_head_dim=multihead_attention_head_dim,
                                        multihead_attention_n_heads=multihead_attention_n_heads,
                                        mulitlayer_perceptron_hidden_dim=multilayer_perceptron_hidden_dim,
                                        dropout_p=dropout_p)
        
        self.head = nn.Linear(in_features=token_dim,out_features=n_classes)
        
        
    def forward(self, input: tensor) -> tensor:
        
        output = self.tokenizer(input)
        output = self.concat_class_token(output)
        output = self.add_position_embedding(output)
        output = self.transformer(output)
        output = output[:,0]            # the first token is the class token
        
        output = self.head(output)
        return output
        
        
        
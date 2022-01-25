#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from torch.optim.lr_scheduler import LambdaLR

import numpy as np
import random
from scipy import ndimage
import ml_collections
import copy
from os.path import join as pjoin
import math
from tqdm import tqdm


# In[2]:


IMG_SIZE = 224
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 3e-2
WEIGHT_DECAY = 0
NUM_STEPS = 10000
WARMUP_STEPS = 500


# In[3]:


config = ml_collections.ConfigDict()
config.patches = ml_collections.ConfigDict({'size': (32, 32)})
config.hidden_size = 768
config.transformer = ml_collections.ConfigDict()
config.transformer.mlp_dim = 3072
config.transformer.num_heads = 12
config.transformer.num_layers = 12
config.transformer.attention_dropout_rate = 0.0
config.transformer.dropout_rate = 0.1
config.classifier = 'token'


# In[4]:


transform_train = transforms.Compose([
    transforms.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.05, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

trainset = datasets.CIFAR10(root="./data",train=True,download=True,transform=transform_train)
testset = datasets.CIFAR10(root="./data",train=False,download=True,transform=transform_test)

train_loader = DataLoader(trainset,sampler=RandomSampler(trainset),batch_size=TRAIN_BATCH_SIZE)
test_loader = DataLoader(testset,sampler=SequentialSampler(testset),batch_size=EVAL_BATCH_SIZE)


# In[6]:


def Weight_transpose(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


# In[7]:


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
img = Image.open('./Cat.jpg')

fig = plt.figure()
plt.imshow(img)


# In[28]:


transform = Compose([Resize((224, 224)), ToTensor()])
x = transform(img)
x = x.unsqueeze(0) # 添加batch维度
x.shape
print(model(x))
print(torch.max(model(x)[0][0],0))


# In[9]:


class ViT(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(ViT, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights


# In[10]:


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


# In[26]:


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        # x: torch.Size([1, 1, 768])
        B = x.shape[0]
        # cls_token: torch.Size([1, 1, 768])
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # cls_tokens: torch.Size([16, 1, 768])
        x = self.patch_embeddings(x)
        # x: torch.Size([16, 768, 14, 14])
        x = x.flatten(2)
        # x: torch.Size([16, 768, 196])
        x = x.transpose(-1, -2)
        # x: torch.Size([16, 196, 768])
        x = torch.cat((cls_tokens, x), dim=1)
        # x: torch.Size([16, 197, 768])

        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


# In[12]:


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


# In[13]:


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


# In[14]:


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"] # 12
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)  # 768 / 12 = 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 12 * 64 = 768

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        # x: torch.Size([16, 197, 768])
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # new_x_shape = (16, 197) + (12, 64) = (16, 197, 12, 64)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # hidden_states: torch.Size([16, 197, 768])
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        # layer: torch.Size([16, 197, 768])

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # layer: torch.Size([16, 12, 197, 64])

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # attention_scores: torch.Size([16, 12, 197, 197])

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        # attention_probs: torch.Size([16, 12, 197, 197])
      
        context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer: torch.Size([16, 12, 197, 64])
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # context_layer: torch.Size([16, 197, 12, 64])
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # context_layer: torch.Size([16, 197, 768])

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


# In[15]:


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# In[16]:


device = torch.device("cpu")
model = ViT(config, IMG_SIZE, zero_head=True, num_classes=10)
model.to(device)


# In[19]:


class Loss_compute(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[ ]:


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9,weight_decay=WEIGHT_DECAY)
t_total = NUM_STEPS
model.zero_grad()
losses = Loss_compute()
global_step = 1
best_acc = 0
gradient_accumulation_steps = 1
while (global_step % t_total != 0):
        model.train()
        epoch_iterator = tqdm(train_loader,desc="Training (X / X Steps) (loss=X.X)",bar_format="{l_bar}{r_bar}",dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(device) for t in batch)
            x, y = batch
            loss = model(x, y)

            
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                losses.update(loss.item()*gradient_accumulation_steps)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val))
                
                model.train()
                if global_step % t_total == 0:
                    break
        losses.reset()


# In[91]:


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predicted = torch.max(outputs[:][0],-1).indices
        c = (predicted == labels).squeeze()
        for i in range(8):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

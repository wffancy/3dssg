import torch
import torch.nn as nn
import torch.nn.functional as F

def build_mlp(dim_list, activation='relu', batch_norm='none',
              dropout=0, final_nonlinearity=True):
  layers = []
  for i in range(len(dim_list) - 1):
    dim_in, dim_out = dim_list[i], dim_list[i + 1]
    layers.append(nn.Linear(dim_in, dim_out))
    final_layer = (i == len(dim_list) - 2)
    if not final_layer or final_nonlinearity:
      if batch_norm == 'batch':
        layers.append(nn.BatchNorm1d(dim_out))
      if activation == 'relu':
        layers.append(nn.ReLU())
      elif activation == 'leakyrelu':
        layers.append(nn.LeakyReLU())
    if dropout > 0:
      layers.append(nn.Dropout(p=dropout))
  return nn.Sequential(*layers)
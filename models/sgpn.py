import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from models.pointnet import PointNetEncoder
from models.graph import GraphTripleConv, GraphTripleConvNet
from models.layers import build_mlp
from lib.config import CONF

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def load_pretained_cls_model(model):
    # load pretrained pointnet_cls model [ relPointNet ver. ]
    pretrained_dict = torch.load(os.path.join(CONF.PATH.BASE, './pointnet_cls_best_model.pth'))["model_state_dict"]
    net_state_dict = model.state_dict()
    pretrained_dict_ = {k[5:]: v for k, v in pretrained_dict.items() if 'feat' in k and v.size() == net_state_dict[k[5:]].size()}
    net_state_dict.update(pretrained_dict_)
    model.load_state_dict(net_state_dict)

class SGPN(nn.Module):
    def __init__(self, use_pretrained_cls, gconv_dim=128, gconv_hidden_dim=512,
                 gconv_pooling='avg', gconv_num_layers=5, mlp_normalization='none',
                 obj_cat_num=160, pred_cat_num=27):
        super().__init__()

        # ObjPointNet and RelPointNet
        self.objPointNet = PointNetEncoder(global_feat=True, feature_transform=True, channel=3) # (x,y,z)
        if use_pretrained_cls:
            load_pretained_cls_model(self.objPointNet)

        self.relPointNet = PointNetEncoder(global_feat=True, feature_transform=True, channel=4) # (x,y,z,M) M-> class-agnostic instance segmentation
        if use_pretrained_cls:
            load_pretained_cls_model(self.relPointNet)

        # GCN module
        if gconv_num_layers == 0:
            self.gconv = nn.Linear(1024, gconv_dim) # final feature of the pointNet2
        elif gconv_num_layers > 0:
            gconv_kwargs = {
                'input_dim': 1024,
                'output_dim': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'mlp_normalization': mlp_normalization,
            }
            self.gconv = GraphTripleConv(**gconv_kwargs)

        self.gconv_net = None
        if gconv_num_layers > 1:
            gconv_kwargs = {
                'input_dim': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'num_layers': gconv_num_layers - 1,
                'mlp_normalization': mlp_normalization,
            }
            self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

        # MLP for classification
        obj_classifier_layer = [gconv_dim, 256, obj_cat_num]
        self.obj_classifier = build_mlp(obj_classifier_layer, batch_norm=mlp_normalization)

        rel_classifier_layer = [gconv_dim, 256, pred_cat_num]
        self.rel_classifier = build_mlp(rel_classifier_layer, batch_norm=mlp_normalization)

    def forward(self, data_dict):
        objects_id = data_dict["objects_id"]
        objects_pc = data_dict["objects_pc"]
        objects_count = data_dict["aligned_obj_num"]   # namely 9
        predicate_pc_flag = data_dict["predicate_pc_flag"]
        predicate_count = data_dict["aligned_rel_num"] # namely 72
        edges = data_dict["edges"]
        trans_feat = []
        batch_size = objects_id.size(0)

        # point cloud pass objPointNet
        objects_pc = objects_pc.permute(0, 2, 1)
        obj_vecs, _, tf1 = self.objPointNet(objects_pc)

        # point cloud pass relPointNet
        predicate_pc_flag = predicate_pc_flag.permute(0, 2, 1)
        pred_vecs, _, tf2 = self.relPointNet(predicate_pc_flag)

        trans_feat.append(tf1)
        trans_feat.append(tf2)
        data_dict["trans_feat"] = trans_feat

        # obj_vecs and rel_vecs pass GCN module
        obj_vecs_list = []
        pred_vecs_list = []
        object_num = int(objects_count.item())
        predicate_num = int(predicate_count.item())
        for i in range(batch_size):
            if isinstance(self.gconv, nn.Linear):
                o_vecs = self.gconv(obj_vecs[object_num*i: object_num*(i+1)])
            else:
                o_vecs, p_vecs = self.gconv(obj_vecs[object_num*i: object_num*(i+1)], pred_vecs[predicate_num*i: predicate_num*(i+1)], edges[i])
            if self.gconv_net is not None:
                o_vecs, p_vecs = self.gconv_net(o_vecs, p_vecs, edges[i])

            obj_vecs_list.append(o_vecs)
            pred_vecs_list.append(p_vecs)

        obj_pred_list = []
        rel_pred_list = []
        for o_vec in obj_vecs_list:
            obj_pred = self.obj_classifier(o_vec)
            obj_pred_list.append(obj_pred)
        for p_vec in pred_vecs_list:
            rel_pred = self.rel_classifier(p_vec)
            rel_pred_list.append(rel_pred)
        # ATTENTION: as batch_size > 1, the value that corresponds to the "predict" key is a list
        data_dict["objects_predict"] = obj_pred_list
        data_dict["predicate_predict"] = rel_pred_list

        return data_dict

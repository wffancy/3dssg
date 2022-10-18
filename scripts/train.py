import os
import sys
import json
import time
import argparse
import torch
import torch.optim as optim
import numpy as np
# from torchstat import stat

from torch.utils.data import DataLoader
from datetime import datetime
from graphviz import Digraph
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.config import CONF
from data.dataset import D3SemanticSceneGraphDataset
from models.sgpn import SGPN
from scripts.solver import Solver
from scripts.eval import get_eval

D3SSG_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "3DSSG_subset/relationships_train.json")))["scans"]
D3SSG_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "3DSSG_subset/relationships_validation.json")))["scans"]

# one scan split include at most 9 classes object, used for visualization
node_color_list = ['aliceblue', 'antiquewhite', 'cornsilk3', 'lightpink', 'salmon', 'palegreen', 'khaki',
                   'darkkhaki', 'orange']
WORKERS = 12
VIS_WITH_GT = 1

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def read_class(path):
    file = open(os.path.join(CONF.PATH.DATA, path), 'r')
    category = file.readline()[:-1]
    word_dict = []
    while category:
        word_dict.append(category)
        category = file.readline()[:-1]

    return word_dict

def get_model(args):
    # initiate model
    use_pretrained_cls = not args.use_pretrained
    model = SGPN(use_pretrained_cls, feature_extractor=CONF.EXTRACTOR, graph_gen=CONF.GNN,
                 gnn_dim=128, gnn_hidden_dim=512, gnn_pooling='avg', gnn_num_layers=5, mlp_normalization='batch')

    # trainable model
    if args.use_pretrained:
        # load model
        print("loading pretrained model...")
        pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_last.pth")
        model.load_state_dict(torch.load(pretrained_path), strict=False)

    # to CUDA
    # model = torch.nn.DataParallel(model)
    model = model.cuda()
    # torch.distributed.init_process_group(backend="nccl")
    # model = torch.nn.parallel.DistributedDataParallel(model)

    # Use inplace operations whenever possible
    model.apply(inplace_relu)

    return model

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataloader):
    model = get_model(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    solver = Solver(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        stamp=stamp,
        val_step=args.val_step,
        lr_decay_step=CONF.SCALAR.LR_DECAY_STEP,
        lr_decay_rate=CONF.SCALAR.LR_DECAY_RATE,
        bn_decay_step=CONF.SCALAR.BN_DECAY_STEP,
        bn_decay_rate=CONF.SCALAR.BN_DECAY_RATE
    )
    num_params = get_num_params(model)
    print('sgpn params:', num_params)
    # stat(model, (9,3,1000))

    return solver, num_params, root

def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value

    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.all_scan_id)
    info["num_val_scenes"] = len(val_dataset.all_scan_id)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

def get_3dssg(d3ssg_train, d3ssg_val, num_scans):
    # get initial scan list
    train_scan_list = sorted(list([data["scan"]+"-"+str(hex(data["split"]))[-1] for data in d3ssg_train]))
    val_scan_list = sorted(list([data["scan"]+"-"+str(hex(data["split"]))[-1] for data in d3ssg_val]))
    if num_scans == -1:
        num_scans = len(train_scan_list)
    else:
        assert len(train_scan_list) >= num_scans
    
    # slice train_scan_list
    train_scan_list = train_scan_list[:num_scans]
    ratio = num_scans / len(d3ssg_train)
    val_scan_list = val_scan_list[:int(len(d3ssg_val) * ratio)]

    # filter data in chosen scenes
    new_3dssg_train = []
    for data in d3ssg_train:
        for l in train_scan_list:
            if data["scan"]==l[:-2] and data["split"]==int(l[-1],16):
                new_3dssg_train.append(data)
    new_3dssg_val = []
    for data in d3ssg_val:
        for l in val_scan_list:
            if data["scan"] == l[:-2] and data["split"] == int(l[-1], 16):
                new_3dssg_val.append(data)

    # new_3dssg_val = d3ssg_val

    print("train on {} samples and val on {} samples".format(len(new_3dssg_train), len(new_3dssg_val)))

    return new_3dssg_train, new_3dssg_val, train_scan_list, val_scan_list

def visualize(data_dict, model, obj_dict, pred_dict, with_GT):
    dot = Digraph(comment='The Scene Graph')
    dot.attr(rankdir='TB')

    with torch.no_grad():
        data_dict = model(data_dict)

    data, pred_relations = get_eval(data_dict)
    triples = data["triples"][0].cpu().numpy().copy()
    object_id = data["objects_id"][0].cpu().numpy()
    object_cat = data["objects_cat"][0].cpu().numpy()
    object_pred = data["objects_predict"][0].cpu().numpy()

    # nodes
    obj_pred_cls = np.argmax(object_pred, axis=1)
    dot.attr('node', shape='oval', fontname='Sans', fontsize='16.0')
    for index in range(len(object_cat)):
        id = str(object_id[index])
        dot.attr('node', fillcolor=node_color_list[index], style='filled')
        pred = obj_pred_cls[index]
        gt = object_cat[index]
        if with_GT:
            note = obj_dict[pred] + '\n(GT:' + obj_dict[gt] + ')'
        else:
            note = obj_dict[pred]
        dot.node(id, note)
    # edges
    dot.attr('edge', fontname='Sans', fontsize='12.0', color='black', style='filled')
    for relation in pred_relations[0]:
        s, o, p = relation
        line = np.repeat(np.array([s, o]).reshape(1, -1), min(len(triples), len(pred_relations[0])), axis=0)
        mask = (line[:, 0] == triples[:, 0]) & (line[:, 1] == triples[:, 1])
        if type(mask) is np.ndarray and sum(mask) > 0:
            gt_p = triples[mask][0, 2]
            triples = np.delete(triples, mask, 0)
        else:
            if type(mask) is bool:
                gt_p = triples[0, 2]
                triples = np.delete(triples, 0, 0)
            else:
                continue
        if p == gt_p:
            dot.attr('edge', color='green')
        else:
            dot.attr('edge', color='black')
        if gt_p == 0:   # ignore ground truth predicate is 'None'
            continue
        if with_GT:
            dot.edge(str(s), str(o), pred_dict[p] + '\n(GT:' + pred_dict[gt_p] + ')')
        else:
            dot.edge(str(s), str(o), pred_dict[p])
    for item in triples:
        s, o, p = item
        if p == 0:
            continue
        dot.attr('edge', color='black')
        if with_GT:
            dot.edge(str(s), str(o), 'None\n(GT:' + pred_dict[p] + ')')
        else:
            dot.edge(str(s), str(o), 'None')

    # print(dot.source)
    scan = data_dict["scan_id"][0][:-2]
    split = data_dict["scan_id"][0][-1]
    dot.render(filename=os.path.join(CONF.PATH.BASE, 'vis/{}/scene_graph_{}.gv'.format(scan, split)))

def train(args):
    # init training dataset
    print("preparing data...")
    d3ssg_train, d3ssg_val, train_scene_list, val_scene_list = get_3dssg(D3SSG_TRAIN, D3SSG_VAL, args.scene_num)
    d3ssg = {
        "train": d3ssg_train,
        "val": d3ssg_val
    }

    # start_time = time.time()
    val_dataset = D3SemanticSceneGraphDataset(relationships=d3ssg["val"],
                                                all_scan_id=val_scene_list, split="val")

    if args.vis:
        start_time = time.time()
        dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True,
                                collate_fn=val_dataset.collate_fn, num_workers=WORKERS)
        use_pretrained_cls = not args.use_pretrained
        model = SGPN(use_pretrained_cls, gnn_dim=1024, gnn_hidden_dim=512,
                     gnn_pooling='avg', gnn_num_layers=5, mlp_normalization='batch')
        assert args.use_pretrained
        pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_last.pth")
        model.load_state_dict(torch.load(pretrained_path), strict=True)
        model = model.cuda()
        model.eval()

        obj_class_dict = read_class("3DSSG_subset/classes.txt")
        pred_class_dict = read_class("3DSSG_subset/relationships.txt")

        images_count = 0
        for data_dict in dataloader:
            # move to cuda
            for key in data_dict:
                if key != "scan_id":
                    data_dict[key] = data_dict[key].cuda()

            tmp_start_time = time.time()
            visualize(data_dict, model, obj_class_dict, pred_class_dict, VIS_WITH_GT)
            tmp_end_time = time.time()
            # print(data_dict['scan_id'][0] + ":" + str(tmp_end_time - tmp_start_time) + "s")
            images_count = images_count + 1
            
        end_time = time.time()
        avg_time = (end_time - start_time) / images_count
        print("total splits num:" + str(images_count))
        print("total time consumption (including data loading):" + str(end_time - start_time))
        print("average drawing time consumption: " + str(avg_time))

        print("finished rendering.")
        return

    # training seg
    train_dataset = D3SemanticSceneGraphDataset(relationships=d3ssg["train"],
                                                all_scan_id=train_scene_list, split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=train_dataset.collate_fn, num_workers=WORKERS, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=val_dataset.collate_fn, num_workers=WORKERS, pin_memory=True)
    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    print("initializing...")
    solver, num_params, root = get_solver(args, dataloader)

    print("start training...\n")
    save_info(args, root, num_params, train_dataset, val_dataset)
    solver(args.epoch, args.verbose)
    print("finished training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--scene_num", type=int, help="number of scenes", default=-1)
    parser.add_argument("--batch_size", type=int, help="batch size", default=1)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=25)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=100)    # train iter
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=1000)   # val iter
    parser.add_argument("--use_pretrained", type=str, help="Specify the folder name containing the pretrained module.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--vis", action="store_true", help="render visualization result")
    # parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    train(args)
import os
import sys
import json
import argparse
import torch
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from graphviz import Digraph
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.config import CONF
from data.dataset import D3SemanticSceneGraphDataset
from models.sgpn import SGPN
from scripts.solver import Solver

D3SSG_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "3DSSG_subset/relationships_train.json")))["scans"]
D3SSG_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "3DSSG_subset/relationships_validation.json")))["scans"]

# one scan split include at most 9 classes object, used for visualization
node_color_list = ['aliceblue', 'antiquewhite', 'darkgray', 'lightpink', 'salmon', 'palegreen', 'khaki',
                   'darkkhaki', 'orange']


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def read_class(path):
    file = open(os.path.join(CONF.PATH.DATA, path), 'r')
    category = file.readline()[:-1]
    word_dict = []
    while category:
        word_dict.append(category)
        category = file.readline()[:-1]

    return word_dict


def get_dataloader(args, d3ssg, all_scene_list, split):
    dataset = D3SemanticSceneGraphDataset(
        relationships=d3ssg[split],
        all_scan_id=all_scene_list,
        split=split
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=dataset.collate_fn, num_workers=0)

    return dataset, dataloader


def get_model(args):
    # initiate model
    model = SGPN(gconv_dim=128, gconv_hidden_dim=512,
               gconv_pooling='avg', gconv_num_layers=5, mlp_normalization='batch')

    # trainable model
    if args.use_pretrained:
        # load model
        print("loading pretrained model...")
        pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_last.pth")
        model.load_state_dict(torch.load(pretrained_path), strict=False)

    # to CUDA
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

    LR_DECAY_STEP = [80, 120, 160]
    LR_DECAY_RATE = 0.1
    BN_DECAY_STEP = 20
    BN_DECAY_RATE = 0.5

    solver = Solver(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        stamp=stamp,
        val_step=args.val_step,
        lr_decay_step=LR_DECAY_STEP,
        lr_decay_rate=LR_DECAY_RATE,
        bn_decay_step=BN_DECAY_STEP,
        bn_decay_rate=BN_DECAY_RATE
    )
    num_params = get_num_params(model)
    print('sgpn params:', num_params)

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
    val_scan_list = val_scan_list[:1]

    # filter data in chosen scenes
    new_3dssg_train = []
    for data in d3ssg_train:
        for l in train_scan_list:
            if data["scan"]==l[:-2] and data["split"]==int(l[-1],16):
                new_3dssg_train.append(data)

    new_3dssg_val = d3ssg_val

    print("train on {} samples and val on {} samples".format(len(new_3dssg_train), len(new_3dssg_val)))

    return new_3dssg_train, new_3dssg_val, train_scan_list, val_scan_list


def visualize(data_dict, model, obj_dict, pred_dict):
    scan_id = data_dict["scan_id"][0]  # here batch_size can only equal 1, or will make error
    ids = data_dict["objects_id"][0]
    gt_obj = data_dict["objects_cat"][0]
    gt_rel = data_dict["triples"][0]
    rel_pairs = data_dict["pairs"][0]
    # used for evaluation
    precision_obj = 0
    precision_rel = 0
    cls_list = []
    pred_list = []

    dot = Digraph(comment='The Scene Graph')
    dot.attr(rankdir='TB')

    with torch.no_grad():
        data_dict = model(data_dict)

    with dot.subgraph(name='cluster_Predicted') as g1:
        g1.attr(label='predicted')
        idx = 0
        data_dict = model(data_dict)
        obj_pred = data_dict["objects_predict"][0]  # since batch_size == 1, here take the first element
        rel_pred = data_dict["predicate_predict"][0]

        obj_pred_cls = torch.argmax(obj_pred, dim=1)
        g1.attr('node', shape='oval', fontname='Sans')
        for index, i in enumerate(obj_pred_cls):
            cls_list.append(i)
            id = str(ids[index].item())
            cls = obj_dict[i]
            g1.attr('node', fillcolor=node_color_list[idx], style='filled')
            g1.node(id + '_', cls)
            idx += 1

        pred_cls_num = rel_pred.size(1)
        g1.attr('edge', fontname='Sans', color='black', style='filled')
        for index, i in enumerate(rel_pred):
            for j in range(pred_cls_num):
                if i[j] >= 0.5:
                    pred_list.append(rel_pairs[index] + [j])
                    s, o = rel_pairs[index]
                    if s == o or j == 0:
                        continue
                    g1.edge(str(s.item()) + '_', str(o.item()) + '_', pred_dict[j])

    with dot.subgraph(name='cluster_GT') as g2:
        g2.attr(label='ground truth')
        g2.attr('node', shape='oval', fontname='Sans')
        for index, v in enumerate(gt_obj):
            if cls_list[index] == v:
                precision_obj += 1
            id = str(ids[index].item())
            g2.attr('node', fillcolor=node_color_list[index], style='filled')
            g2.node(id, obj_dict[v])

        g2.attr('edge', fontname='Sans', color='black', style='filled')
        for item in gt_rel:
            s, o, p = item
            if item.numpy().tolist() in pred_list:
                precision_rel += 1
            if s == o or p.item() == 0:
                continue
            g2.edge(str(s.item()), str(o.item()), pred_dict[p.item()])

    # print(dot.source)
    dot.render(filename=os.path.join(CONF.PATH.BASE, 'vis/{}/SG_{:.2f}_{:.2f}.gv'.format(scan_id, precision_obj/len(gt_obj), precision_rel/len(gt_rel))))


def train(args):
    # init training dataset
    print("preparing data...")
    d3ssg_train, d3ssg_val, train_scene_list, val_scene_list = get_3dssg(D3SSG_TRAIN, D3SSG_VAL, args.scene_num)
    d3ssg = {
        "train": d3ssg_train,
        "val": d3ssg_val
    }
    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, d3ssg, train_scene_list, "train")
    val_dataset, val_dataloader = get_dataloader(args, d3ssg, val_scene_list, "val")
    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    if args.vis:
        dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True,
                                collate_fn=val_dataset.collate_fn, num_workers=0)
        model = SGPN(gconv_dim=128, gconv_hidden_dim=512,
                     gconv_pooling='avg', gconv_num_layers=5, mlp_normalization='batch')
        assert len(args.use_pretrained) > 0
        pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_last.pth")
        model.load_state_dict(torch.load(pretrained_path), strict=False)
        model = model.cuda()
        model.eval()

        obj_class_dict = read_class("3DSSG_subset/classes.txt")
        pred_class_dict = read_class("3DSSG_subset/relationships.txt")

        dataloader = tqdm(dataloader)
        for data_dict in dataloader:
            # move to cuda
            for key in data_dict:
                if key != "scan_id":
                    data_dict[key] = data_dict[key].cuda()

            visualize(data_dict, model, obj_class_dict, pred_class_dict)

        print("finished rendering.")

    else:
        print("initializing...")
        solver, num_params, root = get_solver(args, dataloader)

        print("start training...\n")
        save_info(args, root, num_params, train_dataset, val_dataset)
        solver(args.epoch, args.verbose)
        print("finished training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--scene_num", type=int, help="number of scenes", default=-1)
    parser.add_argument("--batch_size", type=int, help="batch size", default=1)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)    # train iter
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=100)   # val iter
    parser.add_argument("--use_pretrained", type=str, help="Specify the folder name containing the pretrained detection module.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--vis", action="store_true", help="render visualization result")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    train(args)
import os
import sys
import time
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from plyfile import PlyData
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.config import CONF

# yield from the relationships_train.json
MAX_OBJECTS_NUM = 9
MAX_REL_NUM = 72    # actually 57

class DataDict:
    __slots__ = ("scan_id", "objects_id", "objects_cat", "objects_num", "objects_pc",
                 "predicate_cat", "predicate_num", "predicate_pc_flag", "pairs", "edges",
                 "triples", "objects_count", "predicate_count")

    def __init__(self, data_dict):
        self.scan_id = data_dict["scan_id"]
        self.objects_id = np.array(data_dict["objects_id"]).astype(np.uint8)
        self.objects_cat = np.array(data_dict["objects_cat"]).astype(np.uint8)
        self.objects_num = np.array(data_dict["objects_num"]).astype(np.int16)
        self.objects_pc = np.array(data_dict["objects_pc"]).astype(np.float32)
        self.predicate_cat = np.array(data_dict["predicate_cat"]).astype(np.uint8)
        self.predicate_num = np.array(data_dict["predicate_num"]).astype(np.int16)
        self.predicate_pc_flag = np.array(data_dict["predicate_pc_flag"]).astype(np.float32)    # default as double type
        self.pairs = np.array(data_dict["pairs"]).astype(np.uint8)
        self.edges = np.array(data_dict["edges"]).astype(np.uint8)
        self.triples = np.array(data_dict["triples"]).astype(np.uint8)
        self.objects_count = data_dict["objects_count"]
        self.predicate_count = data_dict["predicate_count"]

    def class2dic(self):
        data_dict = {"scan_id": self.scan_id, "objects_id": self.objects_id, "objects_cat": self.objects_cat,
                     "objects_num": self.objects_num, "objects_pc": self.objects_pc,
                     "predicate_cat": self.predicate_cat, "predicate_num": self.predicate_num,
                     "predicate_pc_flag": self.predicate_pc_flag, "pairs": self.pairs, "edges": self.edges,
                     "triples": self.triples, "objects_count": self.objects_count,
                     "predicate_count": self.predicate_count}
        return data_dict

class D3SemanticSceneGraphDataset(Dataset):

    def __init__(self, relationships, all_scan_id,
                 split="train",
                 augment=False):
        ''' target: obtain all data path and split into train/val/test set '''
        self.relationships = relationships # all relationships and classes
        # all scan id, include split id in scan id
        self.all_scan_id = all_scan_id  # useless
        self.split = split
        self.augment = augment
        self.scene_data = []
        self._load_data()

    def __len__(self):
        return len(self.relationships)

    def __getitem__(self, idx):
        """ build_in function: make this class can be indexed like list in python """
        start = time.time()
        # path = os.path.join(CONF.PATH.R3Scan, "{}/data_dict_{}.json".format(self.relationships[idx]["scan"], str(hex(self.relationships[idx]["split"]))[-1]))
        # data_dict = json.load(open(path))
        data_dict = self.scene_data[idx].class2dic()

        data_dict["objects_id"] = np.array(data_dict["objects_id"]).astype(np.int64)    # object id
        data_dict["objects_cat"] = np.array(data_dict["objects_cat"]).astype(np.int64)   # object category
        data_dict["objects_num"] = np.array(data_dict["objects_num"]).astype(np.int64)
        data_dict["objects_pc"] = np.array(data_dict["objects_pc"]).astype(np.float32)  # corresponding point cloud
        data_dict["predicate_cat"] = np.array(data_dict["predicate_cat"]).astype(np.int64)  # predicate id
        data_dict["predicate_num"] = np.array(data_dict["predicate_num"]).astype(np.int64)
        data_dict["predicate_pc_flag"] = np.array(data_dict["predicate_pc_flag"]).astype(np.float32)
        data_dict["pairs"] = np.array(data_dict["pairs"]).astype(np.int64)
        data_dict["edges"] = np.array(data_dict["edges"]).astype(np.int64)
        data_dict["triples"] = np.array(data_dict["triples"]).astype(np.int64)
        data_dict["load_time"] = time.time() - start

        return data_dict

    def _load_data(self):
        print("\nloading {} data into memory...".format(self.split))
        relations = tqdm(self.relationships)
        for relationship in relations:
            path = os.path.join(CONF.PATH.R3Scan, "{}/data_dict_{}.json".format(relationship["scan"], str(
                hex(relationship["split"]))[-1]))
            data_dict = json.load(open(path))
            self.scene_data.append(DataDict(data_dict))
            del data_dict

    def _pad_dict(self, data, key):
        # align all values in the DATA dict except for the point cloud key/value pair
        batch_size = len(data)
        assert batch_size > 0
        if not hasattr(data[0][key], 'dtype'):
            new_value_list = []
            for i in range(batch_size):
                new_value_list.append(data[i][key])
        else:
            l_list = [int(len(one_line[key])) for one_line in data]
            max_l = np.array(l_list).max()
            max_l = max(max_l, MAX_OBJECTS_NUM if 'object' in key else MAX_REL_NUM)
            # max_l = MAX_OBJECTS_NUM if 'object' in key else MAX_REL_NUM
            new_value_list = []
            for i in range(batch_size):
                assert data[i][key].ndim >= 1
                if data[i][key].ndim > 1:
                    dim2 = data[i][key].shape[1]
                    line = np.expand_dims(np.repeat(0, dim2), 0)
                    lines = np.repeat(line, max_l-data[i][key].shape[0], axis=0)
                    new_value_list.append(np.concatenate((data[i][key], lines), axis=0))
                else:
                    line = np.repeat(0, max_l-data[i][key].shape[0])
                    new_value_list.append(np.concatenate((data[i][key], line), axis=0))

        batch_values = np.stack(new_value_list)
        if key == "load_time":
            batch_values = torch.from_numpy(batch_values).type(torch.FloatTensor)
        else:
            batch_values = torch.from_numpy(batch_values).type(torch.LongTensor)
        return batch_values

    def _pad_object_pc(self, data, key):
        batch_size = len(data)
        assert batch_size > 0
        prefix = key.split('_')[0]
        num_key = prefix + '_num'

        # gather all objects' point number
        l_list = []
        for one_line in data:
            l_list.extend(one_line[num_key])
        max_l = np.array(l_list).max()
        num_lines = MAX_OBJECTS_NUM if 'object' in key else MAX_REL_NUM  # align the 'batch_size' dim of input to the pointnet

        # align objects' point cloud
        new_pc_list = []
        for i in range(batch_size):
            start = 0
            dim2 = data[i][key].shape[1]
            for j in range(len(data[i][num_key])):
                line = np.expand_dims(np.repeat(0, dim2), 0)
                num = data[i][num_key][j]
                lines = np.repeat(line, max_l-num, axis=0)
                new_pc_list.append(np.concatenate((data[i][key][start: start+num], lines), axis=0))
                start = start + num
            for j in range(len(data[i][num_key]), num_lines):
                line = np.expand_dims(np.repeat(0, dim2), 0)
                lines = np.repeat(line, max_l, axis=0)
                new_pc_list.append(lines)

        batch_pc = np.stack(new_pc_list)
        batch_pc = torch.from_numpy(batch_pc).type(torch.FloatTensor)
        return batch_pc

    def collate_fn(self, data):
        data_dict = {}
        keys = data[0].keys()
        pc_keys = ["objects_pc", "predicate_pc_flag"]
        ignore_keys = ["scan_id", "objects_num", "predicate_num"]

        scan_id = []
        for i in range(len(data)):
            scan_id.append(data[i]["scan_id"])

        for key in keys:
            if key in pc_keys:
                data_dict[key] = self._pad_object_pc(data, key)
            elif key in ignore_keys:
                continue
            else:
                data_dict[key] = self._pad_dict(data, key)
        data_dict["scan_id"] = scan_id
        data_dict["aligned_obj_num"] = torch.tensor(MAX_OBJECTS_NUM)
        data_dict["aligned_rel_num"] = torch.tensor(MAX_REL_NUM)

        return data_dict

if __name__ == "__main__":
    scans = json.load(open(os.path.join(CONF.PATH.DATA, "3DSSG_subset/relationships_train.json")))["scans"]
    max_objects_num = 0
    max_rel_num = 0
    min_objects_num = 20
    min_rel_num = 200
    for scan in scans:
        if len(scan["relationships"]) == 0:
            print(scan)
        max_objects_num = max(max_objects_num, len(scan["objects"]))
        max_rel_num = max(max_rel_num, len(scan["relationships"]))
        min_objects_num = min(min_objects_num, len(scan["objects"]))
        min_rel_num = min(min_rel_num, len(scan["relationships"]))
    print(max_objects_num, max_rel_num)  # 9 57
    print(min_objects_num, min_rel_num)  # 2 0
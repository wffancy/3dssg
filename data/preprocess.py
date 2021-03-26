import os
import sys
import json
import numpy as np
from plyfile import PlyData

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.config import CONF
import multiprocessing

lock = multiprocessing.Lock()

def read_ply(filename):
    """ read point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z, r, g, b, oid, cid, nyu, mpr] for x, y, z, r, g, b, oid, cid, nyu, mpr in pc])
    return pc_array

def read_obj(filename):
    """ read point cloud from OBJ file"""
    with open(filename) as file:
        point_cloud = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                point_cloud.append((float(strs[1]), float(strs[2]), float(strs[3])))
        point_cloud = np.array(point_cloud)
    return point_cloud

def pc_normalize(pc):
    pc_ = pc[:,:3]
    centroid = np.mean(pc_, axis=0)
    pc_ = pc_ - centroid
    m = np.max(np.sqrt(np.sum(pc_ ** 2, axis=1)))
    pc_ = pc_ / m
    if pc.shape[1] > 3:
        pc = np.concatenate((pc_, pc[:,3].reshape(-1,1)), axis=1)
    else:
        pc = pc_
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    if N < npoint:
        return point
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def judge_obb_intersect(p, obb):
    # judge one point is or not in the obb
    center = np.array(obb["centroid"])
    axis_len = np.array(obb["axesLengths"])
    axis_x = np.array(obb["normalizedAxes"][0:3])
    axis_y = np.array(obb["normalizedAxes"][3:6])
    axis_z = np.array(obb["normalizedAxes"][6:9])
    project_x = axis_x.dot(p - center)
    project_y = axis_y.dot(p - center)
    project_z = axis_z.dot(p - center)
    return project_x >= -axis_len[0]/2 and project_x <= axis_len[0]/2 \
           and project_y >= -axis_len[1]/2 and project_y <= axis_len[1]/2 \
           and project_z >= -axis_len[2]/2 and project_z <= axis_len[2]/2

def process_one_scan(relationships_scan):
    scan_id = relationships_scan["scan"] + "-" + str(hex(relationships_scan["split"]))[-1]
    # load class and relationships dict
    word2idx = {}
    index = 0
    file = open(os.path.join(CONF.PATH.DATA, "3DSSG_subset/classes.txt"), 'r')
    category = file.readline()[:-1]
    while category:
        word2idx[category] = index
        category = file.readline()[:-1]
        index += 1

    rel2idx = {}
    index = 0
    file = open(os.path.join(CONF.PATH.DATA, "3DSSG_subset/relationships.txt"), 'r')
    category = file.readline()[:-1]
    while category:
        rel2idx[category] = index
        category = file.readline()[:-1]
        index += 1

    # read point cloud from OBJ file
    scan = scan_id[:-2]
    pc_array = read_obj(os.path.join(CONF.PATH.R3Scan, "{}/mesh.refined.obj".format(scan)))
    # group points in the same segment
    segments = {}  # key:segment id, value: points belong to this segment
    with open(os.path.join(CONF.PATH.R3Scan, "{}/mesh.refined.0.010000.segs.json".format(scan)), 'r') as f:
        seg_indices = json.load(f)["segIndices"]
        for index, i in enumerate(seg_indices):
            if i not in segments:
                segments[i] = []
            segments[i].append(pc_array[index])

    # group points of the same object
    # filter the object which does not belong to this split
    obj_id_list = []
    for k, _ in relationships_scan["objects"].items():
        obj_id_list.append(int(k))

    with open(os.path.join(CONF.PATH.R3Scan, "{}/semseg.json".format(scan)), 'r') as f:
        seg_groups = json.load(f)["segGroups"]
        objects = {}  # object mapping to its belonging points
        obb = {}  # obb in this scan split, size equals objects num
        labels = {}  # { id: 'category name', 6:'trash can'}
        seg2obj = {}  # mapping between segment and object id
        for o in seg_groups:
            id = o["id"]
            if id not in obj_id_list:  # no corresponding relationships in this split
                continue
            if o["label"] not in word2idx:  # Categories not under consideration
                continue
            labels[id] = o["label"]
            segs = o["segments"]
            objects[id] = []
            obb[id] = o["obb"]
            for i in segs:
                seg2obj[i] = id
                for j in segments[i]:
                    objects[id] = j.reshape(1, -1) if len(objects[id]) == 0 else np.concatenate((objects[id], j.reshape(1, -1)), axis=0)
    # sample and normalize point cloud
    obj_sample = CONF.SCALAR.OBJ_PC_SAMPLE
    for obj_id, obj_pc in objects.items():
        pc = farthest_point_sample(obj_pc, obj_sample)
        objects[obj_id] = pc_normalize(pc)

    objects_id = []
    objects_cat = []
    objects_pc = []
    objects_num = []
    for k, v in objects.items():
        objects_id.append(k)
        objects_cat.append(word2idx[labels[k]])
        objects_num = objects_num + [len(v)]
        objects_pc = v if not len(objects_pc) else np.concatenate((objects_pc, v), axis=0)

    # predicate input of PointNet, including points in the union bounding box of subject and object
    # here consider every possible combine between objects, if there doesn't exist relation in the training file,
    # add the relation with the predicate id replaced by 0
    triples = []
    pairs = []
    relationships_triples = relationships_scan["relationships"]
    for triple in relationships_triples:
        triples.append(triple[:3])
        if triple[:2] not in pairs:
            pairs.append(triple[:2])
    for i in objects_id:
        for j in objects_id:
            if i == j or [i, j] in pairs:
                continue
            triples.append([i, j, 0])   # supplement the 'none' relation
            pairs.append(([i, j]))

    union_point_cloud = []
    predicate_cat = []
    predicate_num = []
    for rel in pairs:
        s, o = rel
        union_pc = []
        pred_cls = np.zeros(len(rel2idx))
        for triple in triples:
            if rel == triple[:2]:
                pred_cls[triple[2]] = 1

        for index, point in enumerate(pc_array):
            if seg_indices[index] not in seg2obj:
                continue
            if judge_obb_intersect(point, obb[s]) or judge_obb_intersect(point, obb[o]):
                if (seg2obj[seg_indices[index]] == s):
                    point = np.append(point, 1)
                elif (seg2obj[seg_indices[index]] == o):
                    point = np.append(point, 2)
                else:
                    point = np.append(point, 0)
                union_pc.append(point)
        union_point_cloud.append(union_pc)
        predicate_cat.append(pred_cls.tolist())
    # sample and normalize point cloud
    rel_sample = CONF.SCALAR.REL_PC_SAMPLE
    for index, _ in enumerate(union_point_cloud):
        pc = np.array(union_point_cloud[index])
        pc = farthest_point_sample(pc, rel_sample)
        union_point_cloud[index] = pc_normalize(pc)
        predicate_num.append(len(pc))

    predicate_pc_flag = []
    for pc in union_point_cloud:
        predicate_pc_flag = pc if len(predicate_pc_flag) == 0 else np.concatenate((predicate_pc_flag, pc), axis=0)

    object_id2idx = {}  # convert object id to the index in the tensor
    for index, v in enumerate(objects_id):
        object_id2idx[v] = index
    s, o, p = np.split(np.array(triples), 3, axis=1)  # All have shape (T, 1)
    s, o, p = [np.squeeze(x, axis=1) for x in [s, o, p]]  # Now have shape (T,)

    for index, v in enumerate(s):
        s[index] = object_id2idx[v]  # s_idx
    for index, v in enumerate(o):
        o[index] = object_id2idx[v]  # o_idx
    edges = np.stack((s, o), axis=1)

    # # since point cloud in 3DSGG has been processed, there is no need to sample any more => actually need
    # point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)

    data_dict = {}
    data_dict["scan_id"] = scan_id
    data_dict["objects_id"] = objects_id  # object id
    data_dict["objects_cat"] = objects_cat  # object category
    data_dict["objects_num"] = objects_num
    data_dict["objects_pc"] = objects_pc.tolist()  # corresponding point cloud
    data_dict["predicate_cat"] = predicate_cat  # predicate id
    data_dict["predicate_num"] = predicate_num
    data_dict["predicate_pc_flag"] = predicate_pc_flag.tolist()  # corresponding point cloud in the union bounding box
    data_dict["pairs"] = pairs
    data_dict["edges"] = edges.tolist()
    data_dict["triples"] = triples
    data_dict["objects_count"] = len(objects_cat)
    data_dict["predicate_count"] = len(predicate_cat)

    return data_dict

def write_into_json(relationship):
    data_dict = process_one_scan(relationship)
    lock.acquire()
    scan_id = data_dict["scan_id"]
    path = os.path.join(CONF.PATH.R3Scan, "{}/data_dict_{}.json".format(scan_id[:-2], scan_id[-1]))
    print("{}/data_dict_{}.json".format(scan_id[:-2], scan_id[-1]))
    with open(path, 'w') as f:
        f.write(json.dumps(data_dict, indent=4))
    lock.release()

if __name__ == '__main__':
    relationships_train = json.load(open(os.path.join(CONF.PATH.DATA, "3DSSG_subset/relationships_train.json")))["scans"]
    relationships_val = json.load(open(os.path.join(CONF.PATH.DATA, "3DSSG_subset/relationships_validation.json")))["scans"]
    # merge two dicts
    relationships = relationships_train + relationships_val

    pool = multiprocessing.Pool(12)
    pool.map(write_into_json, relationships)
    pool.close()
    pool.join()
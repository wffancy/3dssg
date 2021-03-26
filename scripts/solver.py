import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.config import CONF
from lib.pointnet2.pytorch_utils import BNMomentumScheduler
from models.loss import CrossEntropyFocalLoss, PerClassBCEFocalLosswithLogits
from scripts.eval import get_eval
from utils.eta import decode_eta
from models.pointnet import feature_transform_reguliarzer

ITER_REPORT_TEMPLATE = """
------------------------iter: [{epoch_id}: {iter_id}/{total_iter}]-------------------------
[loss] train_loss: {train_loss}
[loss] train_obj_loss: {train_obj_loss}
[loss] train_pred_loss: {train_pred_loss}
[sco.] train_Recall@5_ratio_o: {top5_ratio_o}, train_Recall@10_ratio_o: {top10_ratio_o}
[sco.] train_Recall@3_ratio_r: {top3_ratio_r}, train_Recall@5_ratio_r: {top5_ratio_r}
[sco.] train_Recall@50_predicate: {top50_predicate}, train_Recall@100_predicate: {top100_predicate}
[info] mean_fetch_time: {mean_fetch_time}s
[info] mean_forward_time: {mean_forward_time}s
[info] mean_backward_time: {mean_backward_time}s
[info] mean_eval_time: {mean_eval_time}s
[info] mean_iter_time: {mean_iter_time}s
[info] ETA: {eta_h}h {eta_m}m {eta_s}s
"""

EPOCH_REPORT_TEMPLATE = """
---------------------------------summary---------------------------------
[train] train_loss: {train_loss}
[train] train_obj_loss: {train_obj_loss}
[train] train_pred_loss: {train_pred_loss}
[train] train_Recall@5_ratio_o: {train_top5_ratio_o}, train_Recall@10_ratio_o: {train_top10_ratio_o}
[train] train_Recall@3_ratio_r: {train_top3_ratio_r}, train_Recall@5_ratio_r: {train_top5_ratio_r}
[train] train_Recall@50_predicate: {train_top50_predicate}, train_Recall@100_predicate: {train_top100_predicate}
[valid] val_loss: {val_loss}
[valid] val_obj_loss: {val_obj_loss}
[valid] val_pred_loss: {val_pred_loss}
[valid] val_Recall@5_ratio_o: {val_top5_ratio_o}, val_Recall@10_ratio_o: {val_top10_ratio_o}
[valid] val_Recall@3_ratio_r: {val_top3_ratio_r}, val_Recall@5_ratio_r: {val_top5_ratio_r}
[valid] val_Recall@50_predicate: {val_top50_predicate}, val_Recall@100_predicate: {val_top100_predicate}
"""

BEST_REPORT_TEMPLATE = """
---------------------------------best---------------------------------
[best] epoch: {epoch}
[loss] loss: {loss}
[loss] obj_loss: {obj_loss}
[loss] pred_loss: {pred_loss}
[sco.] train_Recall@5_ratio_o: {top5_ratio_o}
[sco.] train_Recall@10_ratio_o: {top10_ratio_o}
[sco.] train_Recall@3_ratio_r: {top3_ratio_r}
[sco.] train_Recall@5_ratio_r: {top5_ratio_r}
[sco.] train_Recall@50_predicate: {top50_predicate}
[sco.] train_Recall@100_predicate: {top100_predicate}
"""

class Solver():
    def __init__(self, model, dataloader, optimizer, stamp, val_step=10,
                 lr_decay_step=None, lr_decay_rate=None, bn_decay_step=None, bn_decay_rate=None):
        self.epoch = 0
        self.verbose = 0

        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.stamp = stamp
        self.val_step = val_step

        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.bn_decay_step = bn_decay_step
        self.bn_decay_rate = bn_decay_rate

        self.obj_criterion = CrossEntropyFocalLoss()
        self.rel_criterion = PerClassBCEFocalLosswithLogits(alpha=0.25)

        self.best = {
            "epoch": 0,
            "loss": float("inf"),
            "L_obj": float("inf"),
            "L_pred": float("inf"),
            "top5_ratio_o": -float("inf"),
            "top10_ratio_o": -float("inf"),
            "top3_ratio_r": -float("inf"),
            "top5_ratio_r": -float("inf"),
            "top50_predicate": -float("inf"),
            "top100_predicate": -float("inf")
        }
        # init log
        # contains all necessary info for all phases
        self.log = {
            "train": {},
            "val": {}
        }

        # tensorboard
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train"), exist_ok=True)
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"), exist_ok=True)
        self._log_writer = {
            "train": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train")),
            "val": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"))
        }

        # training log
        log_path = os.path.join(CONF.PATH.OUTPUT, stamp, "log.txt")
        self.log_fout = open(log_path, "a")

        # private
        # only for internal access and temporary results
        self._running_log = {}
        self._global_iter_id = 1
        self._total_iter = {}  # set in __call__

        # templates
        self.__iter_report_template = ITER_REPORT_TEMPLATE
        self.__epoch_report_template = EPOCH_REPORT_TEMPLATE
        self.__best_report_template = BEST_REPORT_TEMPLATE

        # lr scheduler
        if lr_decay_step and lr_decay_rate:
            if isinstance(lr_decay_step, list):
                self.lr_scheduler = MultiStepLR(optimizer, lr_decay_step, lr_decay_rate)
            else:
                self.lr_scheduler = StepLR(optimizer, lr_decay_step, lr_decay_rate)
        else:
            self.lr_scheduler = None

        # bn scheduler
        if bn_decay_step and bn_decay_rate:
            it = -1
            start_epoch = 0
            BN_MOMENTUM_INIT = 0.5
            BN_MOMENTUM_MAX = 0.001
            bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * bn_decay_rate ** (int(it / bn_decay_step)), BN_MOMENTUM_MAX)
            self.bn_scheduler = BNMomentumScheduler(model, bn_lambda=bn_lbmd, last_epoch=start_epoch - 1)
        else:
            self.bn_scheduler = None

    def __call__(self, epoch, verbose):
        self.epoch = epoch
        self.verbose = verbose
        self._total_iter["train"] = len(self.dataloader["train"]) * epoch
        self._total_iter["val"] = len(self.dataloader["val"]) * self.val_step

        for epoch_id in range(epoch):
            try:
                self._log("epoch {} starting...".format(epoch_id + 1))

                # feed
                self._feed(self.dataloader["train"], "train", epoch_id)

                # save model
                self._log("saving last models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))

                # update lr scheduler
                if self.lr_scheduler:
                    print("update learning rate --> {}\n".format(self.lr_scheduler.get_last_lr()))
                    self.lr_scheduler.step()

                # update bn scheduler
                if self.bn_scheduler:
                    print("update batch normalization momentum --> {}\n".format(
                        self.bn_scheduler.lmbd(self.bn_scheduler.last_epoch)))
                    self.bn_scheduler.step()

            except KeyboardInterrupt:
                # finish training
                self._finish(epoch_id)
                exit()

        # finish training
        self._finish(epoch_id)

    def _feed(self, dataloader, phase, epoch_id):
        # switch mode
        self._set_phase(phase)
        # re-init log
        self._reset_log(phase)
        # change dataloader
        # tqdm(dataloader[val]) to indicating the process of loading data
        dataloader = dataloader if phase == "train" else tqdm(dataloader)

        for data_dict in dataloader:
            # move to cuda
            for key in data_dict:
                if key != "scan_id":
                    data_dict[key] = data_dict[key].cuda()

            # initialize the running loss
            self._running_log = {
                # loss
                "loss": 0,
                "L_obj": 0,
                "L_pred": 0,
                # recall
                "top5_ratio_o": 0,
                "top10_ratio_o": 0,
                "top3_ratio_r": 0,
                "top5_ratio_r": 0,
                "top50_predicate": 0,
                "top100_predicate": 0
            }

            # load
            self.log[phase]["fetch"].append(data_dict["load_time"].sum().item())

            with torch.autograd.set_detect_anomaly(True):
                # forward
                start = time.time()
                if phase == "train":
                    data_dict = self._forward(data_dict)
                    data_dict = self._compute_loss(data_dict)
                else:
                    with torch.no_grad():
                        data_dict = self._forward(data_dict)
                        data_dict = self._compute_loss(data_dict)
                self.log[phase]["forward"].append(time.time() - start)

                # backward
                if phase == "train":
                    start = time.time()
                    self._backward(data_dict)
                    self.log[phase]["backward"].append(time.time() - start)

            # eval
            start = time.time()
            self._eval(data_dict)
            self.log[phase]["eval"].append(time.time() - start)

            # delete reference to the input/output
            del data_dict

            # record log
            self.log[phase]["loss"].append(self._running_log["loss"].item())
            self.log[phase]["L_obj"].append(self._running_log["L_obj"].item())
            self.log[phase]["L_pred"].append(self._running_log["L_pred"].item())

            self.log[phase]["top5_ratio_o"].append(self._running_log["top5_ratio_o"])
            self.log[phase]["top10_ratio_o"].append(self._running_log["top10_ratio_o"])
            self.log[phase]["top3_ratio_r"].append(self._running_log["top3_ratio_r"])
            self.log[phase]["top5_ratio_r"].append(self._running_log["top5_ratio_r"])
            self.log[phase]["top50_predicate"].append(self._running_log["top50_predicate"])
            self.log[phase]["top100_predicate"].append(self._running_log["top100_predicate"])

            # report
            if phase == "train":
                iter_time = self.log[phase]["fetch"][-1]
                iter_time += self.log[phase]["forward"][-1]
                iter_time += self.log[phase]["backward"][-1]
                iter_time += self.log[phase]["eval"][-1]
                self.log[phase]["iter_time"].append(iter_time)
                if (self._global_iter_id + 1) % self.verbose == 0:
                    self._train_report(epoch_id)

                # evaluation
                if self._global_iter_id % self.val_step == 0:
                    print("evaluating...")
                    # val
                    self._feed(self.dataloader["val"], "val", epoch_id)
                    self._dump_log("val")
                    self._set_phase("train")
                    self._epoch_report(epoch_id)

                # dump log
                self._dump_log("train")
                self._global_iter_id += 1

        # check best
        if phase == "val":
            cur_criterion = "top5_ratio_o"
            cur_best = np.mean(self.log[phase][cur_criterion])
            if cur_best > self.best[cur_criterion]:
                self._log("best {} achieved: {}".format(cur_criterion, cur_best))
                self._log("current train_loss: {}".format(np.mean(self.log["train"]["loss"])))
                self._log("current val_loss: {}".format(np.mean(self.log["val"]["loss"])))
                self.best["epoch"] = epoch_id + 1
                self.best["loss"] = np.mean(self.log[phase]["loss"])
                self.best["L_obj"] = np.mean(self.log[phase]["L_obj"])
                self.best["L_pred"] = np.mean(self.log[phase]["L_pred"])
                self.best["top5_ratio_o"] = np.mean(self.log[phase]["top5_ratio_o"])
                self.best["top10_ratio_o"] = np.mean(self.log[phase]["top10_ratio_o"])
                self.best["top3_ratio_r"] = np.mean(self.log[phase]["top3_ratio_r"])
                self.best["top5_ratio_r"] = np.mean(self.log[phase]["top5_ratio_r"])
                self.best["top50_predicate"] = np.mean(self.log[phase]["top50_predicate"])
                self.best["top100_predicate"] = np.mean(self.log[phase]["top100_predicate"])

                # save model
                self._log("saving best models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.state_dict(), os.path.join(model_root, "model.pth"))

    def _log(self, info_str):
        self.log_fout.write(info_str + "\n")
        self.log_fout.flush()
        print(info_str)

    def _reset_log(self, phase):
        self.log[phase] = {
            # info
            "forward": [],
            "backward": [],
            "eval": [],
            "fetch": [],
            "iter_time": [],
            # loss (float, not torch.cuda.FloatTensor)
            "loss": [],
            "L_obj": [],
            "L_pred": [],
            # scores (float, not torch.cuda.FloatTensor)
            "top5_ratio_o": [],
            "top10_ratio_o": [],
            "top3_ratio_r": [],
            "top5_ratio_r": [],
            "top50_predicate": [],
            "top100_predicate": []
        }

    def _set_phase(self, phase):
        if phase == "train":
            self.model.train()
        elif phase == "val":
            self.model.eval()
        else:
            raise ValueError("invalid phase")

    def _forward(self, data_dict):
        data_dict = self.model(data_dict)
        return data_dict

    def _backward(self, data_dict):
        # optimize
        self.optimizer.zero_grad()
        data_dict["loss"].backward()
        self.optimizer.step()

    def _compute_loss(self, data_dict, lambda_obj=0.1, mat_diff_loss_scale=0.001):
        batch_size = data_dict["objects_id"].size(0)
        Focal_Lobj = []
        Focal_Lpred = []
        # object classification
        for index in range(batch_size):
            obj_count = int(data_dict["objects_count"][index].item())
            rel_count = int(data_dict["predicate_count"][index].item())
            Focal_Lobj.append(self.obj_criterion(data_dict["objects_predict"][index][:obj_count], data_dict["objects_cat"][index][:obj_count]))
            # predicate classification: per-class binary cross entropy.
            Focal_Lpred.append(self.rel_criterion(data_dict["predicate_predict"][index][:rel_count], data_dict["predicate_cat"][index][:rel_count]))

        #  PointNet feature transform reguliarzer
        mat_diff_loss = 0
        for it in data_dict["trans_feat"]:
            tmp = feature_transform_reguliarzer(it)
            mat_diff_loss = mat_diff_loss + tmp * mat_diff_loss_scale
        # total loss
        L_obj_sum = np.array(Focal_Lobj).sum()
        L_pred_sum = np.array(Focal_Lpred).sum()
        Loss = lambda_obj * L_obj_sum + L_pred_sum + mat_diff_loss

        # dump
        self._running_log["loss"] = Loss
        self._running_log["L_obj"] = L_obj_sum / len(Focal_Lobj)    # average object classification loss in one scene
        self._running_log["L_pred"] = L_pred_sum / len(Focal_Lpred)
        data_dict["loss"] = Loss
        data_dict["L_obj"] = L_obj_sum / len(Focal_Lobj)
        data_dict["L_pred"] =  L_pred_sum / len(Focal_Lpred)

        return data_dict

    def _eval(self, data_dict):
        data_dict = get_eval(data_dict)
        # dump
        self._running_log["top5_ratio_o"] = data_dict["top5_ratio_o"]
        self._running_log["top10_ratio_o"] = data_dict["top10_ratio_o"]
        self._running_log["top3_ratio_r"] = data_dict["top3_ratio_r"]
        self._running_log["top5_ratio_r"] = data_dict["top5_ratio_r"]
        self._running_log["top50_predicate"] = data_dict["top50_predicate"]
        self._running_log["top100_predicate"] = data_dict["top100_predicate"]

    def _dump_log(self, phase):
        log = {
            "loss": ["loss", "L_obj", "L_pred"],
            "score": ["top5_ratio_o", "top10_ratio_o", "top3_ratio_r", "top5_ratio_r", "top50_predicate", "top100_predicate"]
        }
        for key in log:
            for item in log[key]:
                self._log_writer[phase].add_scalar(
                    "{}/{}".format(key, item),
                    np.mean([v for v in self.log[phase][item]]),
                    self._global_iter_id
                )

    def _finish(self, epoch_id):
        # print best
        self._best_report()

        # save check point
        self._log("saving checkpoint...\n")
        save_dict = {
            "epoch": epoch_id,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        checkpoint_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(save_dict, os.path.join(checkpoint_root, "checkpoint.tar"))

        # save model
        self._log("saving last models...\n")
        model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))

        # export
        for phase in ["train", "val"]:
            self._log_writer[phase].export_scalars_to_json(
                os.path.join(CONF.PATH.OUTPUT, self.stamp, "tensorboard/{}".format(phase), "all_scalars.json"))

    def _train_report(self, epoch_id):
        # compute ETA
        fetch_time = self.log["train"]["fetch"]
        forward_time = self.log["train"]["forward"]
        backward_time = self.log["train"]["backward"]
        eval_time = self.log["train"]["eval"]
        iter_time = self.log["train"]["iter_time"]

        mean_train_time = np.mean(iter_time)
        mean_est_val_time = np.mean([fetch + forward for fetch, forward in zip(fetch_time, forward_time)])
        eta_sec = (self._total_iter["train"] - self._global_iter_id - 1) * mean_train_time
        eta_sec += len(self.dataloader["val"]) * np.ceil(self._total_iter["train"] / self.val_step) * mean_est_val_time
        eta = decode_eta(eta_sec)

        # print report
        iter_report = self.__iter_report_template.format(
            epoch_id=epoch_id + 1,
            iter_id=self._global_iter_id + 1,
            total_iter=self._total_iter["train"],
            train_loss=round(np.mean([v for v in self.log["train"]["loss"]]), 5),
            train_obj_loss=round(np.mean([v for v in self.log["train"]["L_obj"]]), 5),
            train_pred_loss=round(np.mean([v for v in self.log["train"]["L_pred"]]), 5),
            top5_ratio_o=round(np.mean([v for v in self.log["train"]["top5_ratio_o"]]), 5),
            top10_ratio_o=round(np.mean([v for v in self.log["train"]["top10_ratio_o"]]), 5),
            top3_ratio_r=round(np.mean([v for v in self.log["train"]["top3_ratio_r"]]), 5),
            top5_ratio_r=round(np.mean([v for v in self.log["train"]["top5_ratio_r"]]), 5),
            top50_predicate=round(np.mean([v for v in self.log["train"]["top50_predicate"]]), 5),
            top100_predicate=round(np.mean([v for v in self.log["train"]["top100_predicate"]]), 5),
            mean_fetch_time=round(np.mean(fetch_time), 5),
            mean_forward_time=round(np.mean(forward_time), 5),
            mean_backward_time=round(np.mean(backward_time), 5),
            mean_eval_time=round(np.mean(eval_time), 5),
            mean_iter_time=round(np.mean(iter_time), 5),
            eta_h=eta["h"],
            eta_m=eta["m"],
            eta_s=eta["s"]
        )
        self._log(iter_report)

    def _epoch_report(self, epoch_id):
        self._log("epoch [{}/{}] done...".format(epoch_id + 1, self.epoch))
        epoch_report = self.__epoch_report_template.format(
            train_loss=round(np.mean([v for v in self.log["train"]["loss"]]), 5),
            train_obj_loss=round(np.mean([v for v in self.log["train"]["L_obj"]]), 5),
            train_pred_loss=round(np.mean([v for v in self.log["train"]["L_pred"]]), 5),
            train_top5_ratio_o=round(np.mean([v for v in self.log["train"]["top5_ratio_o"]]), 5),
            train_top10_ratio_o=round(np.mean([v for v in self.log["train"]["top10_ratio_o"]]), 5),
            train_top3_ratio_r=round(np.mean([v for v in self.log["train"]["top3_ratio_r"]]), 5),
            train_top5_ratio_r=round(np.mean([v for v in self.log["train"]["top5_ratio_r"]]), 5),
            train_top50_predicate=round(np.mean([v for v in self.log["train"]["top50_predicate"]]), 5),
            train_top100_predicate=round(np.mean([v for v in self.log["train"]["top100_predicate"]]), 5),
            val_loss=round(np.mean([v for v in self.log["val"]["loss"]]), 5),
            val_obj_loss=round(np.mean([v for v in self.log["val"]["L_obj"]]), 5),
            val_pred_loss=round(np.mean([v for v in self.log["val"]["L_pred"]]), 5),
            val_top5_ratio_o=round(np.mean([v for v in self.log["val"]["top5_ratio_o"]]), 5),
            val_top10_ratio_o=round(np.mean([v for v in self.log["val"]["top10_ratio_o"]]), 5),
            val_top3_ratio_r=round(np.mean([v for v in self.log["val"]["top3_ratio_r"]]), 5),
            val_top5_ratio_r=round(np.mean([v for v in self.log["val"]["top5_ratio_r"]]), 5),
            val_top50_predicate=round(np.mean([v for v in self.log["val"]["top50_predicate"]]), 5),
            val_top100_predicate=round(np.mean([v for v in self.log["val"]["top100_predicate"]]), 5),
        )
        self._log(epoch_report)

    def _best_report(self):
        self._log("training completed...")
        best_report = self.__best_report_template.format(
            epoch=self.best["epoch"],
            loss=round(self.best["loss"], 5),   # 5 decimals
            obj_loss=round(self.best["L_obj"], 5),
            pred_loss=round(self.best["L_pred"], 5),
            top5_ratio_o=round(self.best["top5_ratio_o"], 5),
            top10_ratio_o=round(self.best["top10_ratio_o"], 5),
            top3_ratio_r=round(self.best["top3_ratio_r"], 5),
            top5_ratio_r=round(self.best["top5_ratio_r"], 5),
            top50_predicate=round(self.best["top50_predicate"], 5),
            top100_predicate=round(self.best["top100_predicate"], 5)
        )
        self._log(best_report)
        with open(os.path.join(CONF.PATH.OUTPUT, self.stamp, "best.txt"), "w") as f:
            f.write(best_report)
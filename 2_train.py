import numpy as np
from light_training.dataloading.dataset import get_train_test_loader_from_test_list
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last
from light_training.evaluation.metric import dice
import os
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, label):
        """
        logits: [B, 2, D, H, W] （未 softmax）
        label:  [B, D, H, W]，0/1
        """
        # 1. 获取前景通道概率
        prob = F.softmax(logits, dim=1)[:, 1]       # [B, D, H, W]
        label_f = label.float()                     # 转为 float

        # 2. 计算 Dice
        intersection = torch.sum(prob * label_f)
        union = torch.sum(prob) + torch.sum(label_f)
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)

        # 3. 返回 1 - Dice
        return 1 - dice_coef

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        gamma: 聚焦参数，通常设置 2.0
        alpha: 类别权重，[w_bg, w_fg]，或 None
        """
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, logits, label):
        """
        logits: [B, 2, D, H, W]
        label:  [B, D, H, W]
        """
        B, C, D, H, W = logits.shape
        # 1. 计算概率
        probs = F.softmax(logits, dim=1)            # [B,2,D,H,W]
        prob = probs.permute(0,2,3,4,1).reshape(-1, C)   # [N,2]
        lab  = label.view(-1)                             # [N]

        # 2. 取出每个点对应的预测概率 p_t
        pt = prob[torch.arange(prob.shape[0]), lab]       # [N]

        # 3. Focal Loss 公式
        log_pt = torch.log(pt + 1e-12)
        focal_term = (1 - pt) ** self.gamma
        loss = - focal_term * log_pt

        # 4. 加入类别权重
        if self.alpha is not None:
            at = self.alpha.to(logits.device)[lab]        # [N]
            loss = at * loss

        # 5. 汇总
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.5)
        self.patch_size = patch_size
        self.augmentation = augmentation
        self.train_process = 12

        pre_parameter_path = "./data/parameter/best_model_AIMS_26.pth"
        ckpt = torch.load(pre_parameter_path, map_location=f'cuda:{self.local_rank}')

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in ckpt.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v

        # from models_segmamba.segmambav2 import SegMamba
        from models.segmamba import SegMamba
        self.model = SegMamba(1, 2)
        if pre_parameter_path:
            self.model.load_state_dict(new_state_dict, strict=True)
            print("### Use Pre Parameter ###")


        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=3e-5,
                                    momentum=0.99, nesterov=True)
        # self.scheduler_type = "poly"
        self.scheduler_type = None
        
        self.loss_func = nn.CrossEntropyLoss()
        self.dice_loss_fn = DiceLoss()
        self.focal_loss_fn = FocalLoss(gamma=2.0, alpha=[0.25, 0.75])
      

    def training_step(self, batch):
        import time 
        image, label = self.get_input(batch)

        pred = self.model(image)
        # loss = self.loss_func(pred, label)
        loss = 0.3* self.loss_func(pred, label) + 0.4* self.dice_loss_fn(pred,label) + 0.3* self.focal_loss_fn(pred,label)

        self.log("train_loss", loss, step=self.global_step)
        return loss 

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]

        label = label[:, 0].long()

        return image, label 

    def cal_metric(self, gt, pred, voxel_spacing=[1.0, 1.0, 1.0]):
        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            return np.array([d, 50])
        
        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 50])
        
        else:
            return np.array([0.0, 50])
    
    def validation_step(self, batch):
        image, label = self.get_input(batch)
       
        output = self.model(image).argmax(dim=1)
        output = output.cpu().numpy()
        target = label.cpu().numpy()
        
        dices = []

        c = 2
        # for i in range(1, c):
        for i in range(0, c):
            pred_c = output == i
            target_c = target == i

            cal_dice, _ = self.cal_metric(target_c, pred_c)
            dices.append(cal_dice)
        
        return dices
    
    def validation_end(self, val_outputs):
        dices = val_outputs

        dices_mean = []
        c = 2
        for i in range(0, c):
            dices_mean.append(dices[i].mean())

        mean_dice = sum(dices_mean) / len(dices_mean)
        
        # self.log("0", dices_mean[0], step=self.epoch)
        # self.log("1", dices_mean[1], step=self.epoch)
        # self.log("2", dices_mean[2], step=self.epoch)

        # self.log("mean_dice", mean_dice, step=self.epoch)

        print("*" * 50)
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            path = os.path.join(model_save_path, f"best_model_AIMS_26.pth")
            save_new_model_and_delete_last(self.model, 
                                            path, 
                                            delete_symbol="best_model")
            print("Save best model!!!  Path: {}".format(path))

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_AIMS_26.pth"), 
                                        delete_symbol="final_model")
        print(f"epoch {self.epoch} mean_dice is {mean_dice}, best_mean_dice is {self.best_mean_dice}")
        # print(f"mean_dice is {mean_dice}")



parameter_save_path = "./data/parameter"
env = "DDP"
model_save_path = os.path.join(parameter_save_path)
max_epoch = 1000
batch_size = 24
val_every = 2
num_gpus = 1
device = "cuda:0"
patch_size = [64, 64, 64]
augmentation = True 


if __name__ == "__main__":
    data_dir = "./data/train_all_fullres_process"
    
    trainer = BraTSTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            # logdir=logdir,
                            logdir=parameter_save_path,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17731,
                            training_script=__file__)

    # from data.test_all_list_zero_zero_two import test_list
    from data.test_all_list_zero_one import test_list
    # from data.test_all_list import test_list
    train_ds, test_ds = get_train_test_loader_from_test_list(data_dir=data_dir, test_list=test_list)

    trainer.train(train_dataset=train_ds, val_dataset=test_ds)
    # End Child Processes
    # pkill -f train_1_SegMamba_copy.py 

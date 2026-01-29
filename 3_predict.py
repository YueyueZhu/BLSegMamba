import numpy as np
from light_training.dataloading.dataset import get_train_test_loader_from_test_list
import torch 
from monai.inferers import SlidingWindowInferer
# from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.evaluation.metric import ConfusionMatrix
from scipy import ndimage
set_determinism(123)
import os
from light_training.prediction import Predictor
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "01"

data_dir = "./data/train_all_fullres_process/"
env = "pytorch"
max_epoch = 1000
batch_size = 2
val_every = 2
num_gpus = 1
device = "cuda:0"

patch_size = [64, 64, 64]

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        
        self.patch_size = patch_size
        self.augmentation = False

    def convert_labels(self, labels):
        result = [labels == 0., labels == 1.]
        
        return torch.cat(result, dim=1).float()
    
    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
        label[label == -1] = 0
        properties = batch["properties"]
        raw_label = label

        label = self.convert_labels(label)
        
        return image, label, properties, raw_label
    
    def define_model_segmambav2(self):
        from models.segmamba import SegMamba
        model = SegMamba(1, 2)

        model_path = "./data/parameter/best_model_AIMS_26.pth"
        new_sd = self.filte_state_dict(torch.load(model_path, map_location="cpu"))
        model.load_state_dict(new_sd)
        model.eval()
        window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.5,
                                        progress=True,
                                        mode="gaussian")

        predictor = Predictor(window_infer=window_infer,
                              mirror_axes=[0,1,2])

        save_path = "./data/result/AIMS_TBI_26"
        os.makedirs(save_path, exist_ok=True)

        return model, predictor, save_path

    def convert_labels_dim0(self, labels):
        ## TC, WT and ET
        result = [labels == 0., labels == 1.]
        
        return torch.cat(result, dim=0).float()
    
    def validation_step(self, batch):
        image, label, properties, raw_label = self.get_input(batch)
    
        model, predictor, save_path = self.define_model_segmambav2()
       
        model_output = predictor.maybe_mirror_and_predict(image, model, device=device)

        # labels = model_output.argmax(dim=1)
        model_output = post_process_mask(model_output, thresh=0.000004)
        # model_output = post_process_mask(model_output, thresh=0.000001)

        model_output_singe_channel = model_output[0].argmax(dim=0)[None]
        # print(model_output.shape)
        model_output_singe_channel = self.convert_labels_dim0(model_output_singe_channel)
    
        # After this, shape becomes [C, D, H, W] for 3D or [C, H, W] for 2D
        label = label[0]
        c = 2
        dices = []
        for i in range(0, c):
            output_i = model_output_singe_channel[i].cpu().numpy()
            # print(output_i.shape)
            label_i = label[i].cpu().numpy()
            # print(label_i.shape)
            d = dice(output_i, label_i)
            dices.append(d)

        print(dices)
        # model_output = model_output[0].argmax(dim=0)[None]
        # model_output = model_output.float()
        # # print(model_output.shape)
        # model_output = predictor.predict_raw_probability(model_output, properties=properties) # Crop Size
        # model_output = predictor.predict_noncrop_probability(model_output, properties)
        # predictor.save_to_nii(model_output, 
        #                       raw_spacing=[1,1,1],
        #                       case_name = properties['name'][0],
        #                       save_dir=save_path)
        
        return dices


    def filte_state_dict(self, sd):
        if "module" in sd :
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k 
            new_sd[new_k] = v 
        del sd 
        return new_sd




def dice(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    print("test_empty:", test_empty)
    print("reference_empty:", reference_empty)

    if not test_empty and not reference_empty:
        return float(2. * tp / (2 * tp + fp + fn))

    elif test_empty and reference_empty:
        # if nan_for_nonexisting:
        #     return float("NaN")
        # else:
        #     return 0.
        return 1.0  # both empty, return 1.0 as per convention
    else:
        return float(0.0)  # one is empty, return 0.0 as per convention
    # return float(2. * tp / (2 * tp + fp + fn))

def post_process_mask(pred: torch.Tensor, thresh: float = 0.0005) -> torch.Tensor:
    """
    后处理网络输出的二分类 logits 或概率图。
    
    Args:
        pred (torch.Tensor): 形状 (B,2,H,W,D)，可以是 logits 或概率。dtype 任意，device 任意。
        thresh (float): 最小前景占比阈值，默认万分之五 (0.0005)。
    
    Returns:
        torch.Tensor: 与输入 pred 同 dtype、同 device，
                      形状 (B,2,H,W,D) 的 one-hot 后处理结果。
                      通道 0 是背景，通道 1 是前景。
    """
    # 记录输入的 dtype 和 device
    orig_dtype  = pred.dtype
    orig_device = pred.device

    # step1: argmax 得到 [0,1] 标签
    labels = pred.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)  # (B,H,W,D)
    B, H, W, D = labels.shape
    total_voxels = H * W * D
    # total_voxels = 256 * 256 * 256

    out_labels = np.zeros_like(labels, dtype=np.uint8)

    for b in range(B):
        mask = labels[b]
        nonzero = mask.sum()
        ratio = nonzero / total_voxels

        # 全局占比太小，跳过
        if ratio < thresh:
            continue

        # labeled, num_features = ndimage.label(mask)
        # if num_features == 0:
        #     continue

        # counts = np.bincount(labeled.ravel())
        # counts[0] = 0
        # largest_label = counts.argmax()
        # largest_count = counts[largest_label]
        # largest_ratio = largest_count / total_voxels

        # out_labels[b] = (labeled == largest_label).astype(np.uint8)
        out_labels[b] = labels[b]


    # 转 one-hot，背景=chan0，前景=chan1
    out_onehot = np.stack([1 - out_labels, out_labels], axis=1)  # (B,2,H,W,D)

    # 回到 torch，“to” 回原 dtype/device
    out_tensor = torch.from_numpy(out_onehot)             \
                       .to(device=orig_device)           \
                       .to(dtype=orig_dtype)

    return out_tensor

if __name__ == "__main__":

    trainer = BraTSTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir="",
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17752,
                            training_script=__file__)

    from data.test_all_list_zero_one import test_list
    
    train_ds, test_ds = get_train_test_loader_from_test_list(data_dir=data_dir, test_list=test_list)

    trainer.validation_single_gpu(test_ds)




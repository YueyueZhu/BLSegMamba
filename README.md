<div align="center">
  
# 🧩 BLSegMamba An Optimized SegMamba Framework for msTBI Lesion Segmentation in MRI

</div>
  
## The First Order Of Challenge AIMS-TBI 2025 

![image](./Pictures/reward.png)

## The overall architecture of BLSegMamba

![image](./Pictures/model.png)

> Visual Results on AIMS-TBI 2025.

![image](./Pictures/visual.png)

> Segmentation results under different patch sizes, batch sizes, and loss settings.

![image](./Pictures/results.png)

## ⚡ Data

### AIMS-TBI 2025

The data directory should be organized in the following manner.:

```text
data/
└── train_all_data/
    ├── scan_0001/
    │   ├── T1.nii.gz
    │   ├── seg.nii.gz
    │   └── seg.nii.gz
    ├── scan_0004/
    │   └── ...
    ├── scan_0007/
    │   └── ...
    └── ...
```

## ⚡ Environment install

Creating a virtual environment in terminal: `conda create -n BLSegMamba python=3.10`

```bash
git clone https://github.com/YueyueZhu/BLSegMamba.git

cd BLSegMamba
```

### Install causal-conv1d

```bash
cd causal-conv1d

python setup.py install
```

### Install mamba

```bash
cd mamba

python setup.py install
```

### Install monai 

```bash
pip install monai
```
















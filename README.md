# High-Quality Fusion and Visualization for MR-PET Brain Tumor Images via Multi-Dimensional Features
IEEE Transactions on Image Processing, 2024


## 🙌 Overview
This repository contains the implementation of MdAFuse for medical image fusion.

## ⚙ Network Architecture

The overall framework of the proposed MdAFuse is illustrated in ``model/framework/Fig2.pdf``, and the network implementation is provided in ``model/myNet.py``.


## 🌐 Usage
If you have any questions or need help, please contact me (wjy1361120721@163.com), thank you!

### 🏊 Training

``train/TrainProcess.ipynb``


### 🏄 Testing

**1. Pretrained models**

Pretrained models are available in ``model/MineCheckpoint.pth``, which is responsible for the Medical Image Fusion tasks. 

**2. Test datasets**

The test datasets used in the paper have been stored in ``testImages/MRI_CT``, ``testImages/MRI_PET`` 
and ``testImages/MRI_SPECT`` for Medical Image Fusion.

**3. Results in Our Paper**

If you want to infer with our MdAFuse and obtain the fusion results in our paper, please run 
```python test.py``` or ```test.ipynb``` for Medical Image Fusion. 

The testing results will be printed in the terminal and in ``testResults/``. 




## Citation
If you find this work useful for your research, please cite our paper:

@article{wen_higjquality_2024,
  title={High-Quality Fusion and Visualization for MR-PET Brain Tumor Images via Multi-Dimensional Features},
  author={Jinyu Wen, Asad Khan, Amei Chen, Weilong Peng, Meie Fang, C. L. Philip Chen, and Ping Li},
  journal={IEEE Transactions on Image Processing},
  volume={33},
  pages={3550--3563},
  year={2024},
  publisher={IEEE}
}


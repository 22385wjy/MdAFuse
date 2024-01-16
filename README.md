# MdAFuse   ## Wait to Update
Codes for ***MdAFuse: -- -- --- --Image Fusion. (IEEE TRANSACTIONS ON = = = 2024)***


## 🙌 CDDFuse

### ⚙ Network Architecture

Our MdAFuse as shown in ``model/framework/Fig2.pdf``, which is implemented in ``model/myNet.py``.


## 🌐 Usage

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




## Citation--- Wait to Update
```
@article{***,
  title={MdAFuse: *****},
  author={************},
  journal={IEEE Transactions on *************},
  year={2024},
  publisher={IEEE}
}
```

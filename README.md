# REHDR
Event-guided multimodal fusion for high dynamic range video reconstruction


### Installation
This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks. 

```python
python 3.11.5
pytorch 2.1.1
cuda 11.6
```



```
git clone https://github.com/ice-cream567/REHDR
cd REHDR
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

### <span id="dataset_section"> Dataset </span> 
download our RealHDR dataset at [BaiduYunPan](https://pan.baidu.com/s/1wt3vERs0o-MZgnZFpIsvsg?pwd=trd1)





### Train
---
* train

  * ```python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/GoPro/EFNet.yml --launcher pytorch```

* eval
  * Download [pretrained model](https://pan.baidu.com/s/1PMfcEg6SkV5_ssq8ne13Og?pwd=mvhu) to ./experiments/pretrained_models/EDMF-GoPro.pth
  * ```python basicsr/test.py -opt options/test/GoPro/EDMF.yml  ```




### Contact
Should you have any questions, please feel free to contact guoguangsha21@mails.ucas.ac.cn.

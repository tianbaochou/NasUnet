## NAS-Unet: Neural Architecture Search for Medical Image Segmentation

In this paper, we design three types of primitive operation set on search space to automatically find two
cell architecture DownSC and UpSC for semantic image segmentation especially medical image segmen-
tation. The architectures of DownSC and UpSC updated simultaneously by a differential
architecture strategy during search stage. We demonstrate the well segmentation results of the proposed
method on Promise12, Chaos and ultrasound nerve datasets, which collected by Magnetic Resonance
Imaging (MRI), Computed Tomography (CT), and ultrasound respectively.

![figure1](imgs/figure1.jpg)


## Requirement

+ Ubuntu14.04/16.04 or Window10 (Win7 may also support.)
+ python 3.7
+ torch >= 1.0
+ torchvision >= 0.2.1
+ tqdm
+ numpy
+ pydicom (for chao dataset)
+ SimpleITK (for promise12 dataset)
+ Pillow
+ matplotlib (optional)
+ pydensecrf (optional)
+ pygraphviz (optional)


TODO:

- [ ] Parallel compute no-topology related operation in DownSC and UpSC
- [ ] Optimize multi objective by adding hardware metrics (delay), GPU cost and network parameters
- [ ] Support multi gpus when update the architecture parameters
- [ ] Extend search strategy for a flexible backbone network
- [ ] Merge this work to CNASV ( Search-Train Prototype for Computer Vision [CNASV](https://github.com/tianbaochou/CNASV))

## Usage

```bash
pip3 install requirements.txt
```

**Notice**
> if you use win10, and want to show the cell architecture with graph, you 
need install the pygraphviz and add ```$workdir$\\3rd_tools\\graphviz-2.38\\bin```
into environment path. Here ```$workdir$``` is the custom work directory. such as ```E:\\workspace\\NasUnet``` 

### Search the architecture

```bash
cd experiment
# search on pascal voc2012
python train.py --config ../configs/nasunet/nas_unet_voc.yml
```

### Evaluate the architecture on medical image datasets

+ train on promise12 dataset use nasunet
```bash
python train.py --config ../configs/nasunet/nas_unet_promise12.yml --model nasunet
```

+ if you want to fine tune model:

```bash
python train.py --config ../configs/nasunet/nas_unet_promise12.yml --model nasunet --ft
```

+ use multi-gpus

**edit configs/nas_unet/nas_unet_promise12.yml**

```yaml
training:
    geno_type: NASUNET
    init_channels: 32
    depth: 5
    epoch: 200
    batch_size: 6
    report_freq: 10
    n_workers: 2
    multi_gpus: True # need set to True for multi gpus
```

and then 

```bash
python train.py --config ../configs/nasunet/nas_unet_promise12.yml --model nasunet --ft
```

We will use the all gpu devices for training.

>Both in search and train stage, if you run in one gpu, we will find a max free gpu and transfer model to it.
So you can run N instances without manual set the device ids, if you have N gpu devices.

The final architectures of DownSC and UpSC we searched on pascal voc 2012.

![figure2](imgs/figure2.jpg)


## Citation





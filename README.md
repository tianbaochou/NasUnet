## NAS-U-NET: Neural Architecture Search Algorithm for U-Like Network Searching

```python
CUDA_VISIBLE_DEVICES=1,2  python train.py
```

```python
epoches = {
    'coco': 30,
    'pascal_aug': 80,
    'pascal_voc': 50,
    'pcontext': 80,
    'ade20k': 180,
    'citys': 240,
}

lrs = {
    'coco': 0.004,
    'pascal_aug': 0.001,
    'pascal_voc': 0.0001,
    'pcontext': 0.001,
    'ade20k': 0.004,
    'citys': 0.004,
}

optimizer = 'SGD'
args.lr = lrs[args.dataset.lower()] / 16 * args.batch_size

```




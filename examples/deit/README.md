# DeiT

This code builds on [DeiT on ImageNet](https://github.com/facebookresearch/deit).

## Run
1. Clone the deit repository locally:
```
git clone https://github.com/facebookresearch/deit.git
```

2. Follow [README_deit](https://github.com/facebookresearch/deit/blob/main/README_deit.md) for setup and data preparation.

3. Copy Scion files to the deit repository:
```
cp -r main_scion.py scion.py deit/
```

4. Train DeiT-base model:
```
torchrun --nnodes=4 --nproc_per_node=4 main_scion.py --model deit_base_patch16_224 --epochs 200 --output_dir path2checkpoints_scion --batch-size 256 --lr 8e-5 --min-lr 1e-7 --warmup-epochs 0 --data-path "path_to_imagenet"
```
This should give
```
"test_acc1": 81.974, "test_acc5": 95.716
```


## CHANGELOG

Changes made:

- Modernized architecture:
    - RMS norm instead of LayerNorm
    - GELU `sqrt(2)` scaling
- Increases total batch size to 4096
- Decreases the number of epochs from 300 to 200
- No warmup

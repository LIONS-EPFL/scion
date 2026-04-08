# Modded NanoGPT

This code builds on [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt/).

## Setup

```bash
pip install -r requirements.txt
pip install -r data/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124 --upgrade
python data/cached_fineweb10B.py 24 # downloads only the first 2.4B training tokens to save time
```

## Run

```bash
torchrun --standalone --nproc_per_node=4 train_gpt_scion.py
torchrun --standalone --nproc_per_node=4 train_gpt_scionlight.py
```

Notes: 

- `ScionLight` has necessary changes tagged with "ScionLight modification" (specifically, don't zero gradients and be careful with gradient accumulation)
- When changing `n_embd`, remember to change `n_head` accordingly to `n_embd // 128` to maintain head dimension of 128.

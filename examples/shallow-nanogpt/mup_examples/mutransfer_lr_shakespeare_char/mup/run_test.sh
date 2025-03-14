for width in 256
do
    for lr in 0.00097656
    do
        for seed in 1 # 2 3
        do
            head_size=64
            n_heads=$((width / head_size))
            mup_base_width=256
            mup_width_multiplier=$(echo "scale=8; $width/$mup_base_width" | bc -l)
            out_dir="mup_examples/mutransfer_lr_shakespeare_char/mup/out/test"
            python train.py \
                --out_dir=$out_dir \
                --eval_on_end=True \
                --eval_iters=200 \
                --skip_val_loss=False \
                --eval_only=False \
                --log_interval=1 \
                --always_save_checkpoint=False \
                --never_save_checkpoint=True \
                --init_from='scratch' \
                --wandb_log=False \
                --csv_log=True \
                --dataset='shakespeare_char' \
                --gradient_accumulation_steps=1\
                --batch_size=32 \
                --block_size=1024 \
                --n_layer=2 \
                --n_head=$n_heads \
                --n_embd=$width \
                --dropout=0.0 \
                --bias=False \
                --init_std=0.02 \
                --learning_rate=$lr \
                --max_iters=122 \
                --weight_decay=1e-1 \
                --beta1=0.9 \
                --beta2=0.95 \
                --grad_clip=1.0 \
                --decay_lr=False \
                --mup_enabled=True \
                --mup_width_multiplier=$mup_width_multiplier \
                --mup_input_alpha=1.0 \
                --mup_output_alpha=1.0 \
                --seed=$seed \
                --backend='nccl' \
                --device='cuda' \
                --dtype='float32' \
                --compile=False
        done
    done
done

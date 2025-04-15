#!/bin/bash
if [ "$1" == "train" ]; then
    echo "Starting training..."
    CUDA_VISIBLE_DEVICES=2 python3 Trainer.py --DR LAB4_Dataset/ --save_root saved_models/ --exp_name sgd_lr_5e-4 --device cuda --batch_size 12 \
                    --num_workers 32 --lr 5e-4 --optim AdamW --kl_anneal_type Cyclical --num_epoch 150 --tfr 0.8
    # CUDA_VISIBLE_DEVICES=1 python3 Trainer.py --DR LAB4_Dataset/ --save_root saved_models_sgd2/ --device cuda --batch_size 12 \
    #                 --num_workers 32 --lr 2e-3 --optim SGD --kl_anneal_type Cyclical --num_epoch 500 --tfr 0.8

elif [ "$1" == "test" ]; then
    echo "Starting testing..."
    CUDA_VISIBLE_DEVICES=1 python Tester.py --DR LAB4_Dataset/ --save_root Results/ --ckpt_path saved_models/tfr=03_basic/best.ckpt --device cuda  \
                    --num_workers 32 --batch_size 24
elif [ "$1" == "exp" ]; then
    echo "Starting experiment.."
    # CUDA_VISIBLE_DEVICES=2 python3 Trainer.py --DR LAB4_Dataset/ --save_root saved_models/ --device cuda --batch_size 12 \
    #                 --num_workers 32 --lr 5e-4 --optim AdamW --kl_anneal_type Cyclical --num_epoch 100 --tfr 1.0\
    #                 --exp_name cyclical_basic
    # CUDA_VISIBLE_DEVICES=2 python3 Trainer.py --DR LAB4_Dataset/ --save_root saved_models/ --device cuda --batch_size 12 \
    #                 --num_workers 32 --lr 5e-4 --optim AdamW --kl_anneal_type Monotonic --num_epoch 100 --tfr 1.0\
    #                 --exp_name monotonic_basic
    # CUDA_VISIBLE_DEVICES=2 python3 Trainer.py --DR LAB4_Dataset/ --save_root saved_models/ --device cuda --batch_size 12 \
    #                 --num_workers 32 --lr 5e-4 --optim AdamW --kl_anneal_type None --num_epoch 100 --tfr 1.0\
    #                 --exp_name none_basic
    # CUDA_VISIBLE_DEVICES=2 python3 Trainer.py --DR LAB4_Dataset/ --save_root saved_models/ --device cuda --batch_size 12 \
    #                 --num_workers 32 --lr 5e-4 --optim AdamW --kl_anneal_type Cyclical --num_epoch 100 --tfr 1.0\
    #                 --exp_name tfr=1_basic
    # CUDA_VISIBLE_DEVICES=2 python3 Trainer.py --DR LAB4_Dataset/ --save_root saved_models/ --device cuda --batch_size 12 \
    #                 --num_workers 32 --lr 5e-4 --optim AdamW --kl_anneal_type Cyclical --num_epoch 100 --tfr 0.8\
    #                 --exp_name tfr=08_basic
    # CUDA_VISIBLE_DEVICES=2 python3 Trainer.py --DR LAB4_Dataset/ --save_root saved_models/ --device cuda --batch_size 12 \
    #                 --num_workers 32 --lr 5e-4 --optim AdamW --kl_anneal_type Cyclical --num_epoch 100 --tfr 0.3\
    #                 --exp_name tfr=03_basic
    # CUDA_VISIBLE_DEVICES=2 python3 Trainer.py --DR LAB4_Dataset/ --save_root  saved_models/ --device cuda --batch_size 12 \
    #                 --num_workers 32 --lr 5e-4 --optim Adam --kl_anneal_type Cyclical --num_epoch 500 --tfr 0.3\
    #                 --exp_name adam_lr5e-4_extra
    # CUDA_VISIBLE_DEVICES=0 python3 Trainer.py --DR LAB4_Dataset/ --save_root  saved_models/ --device cuda --batch_size 12 \
    #                 --num_workers 32 --lr 5e-4 --optim AdamW --kl_anneal_type Cyclical --num_epoch 500 --tfr 0.3\
    #                 --exp_name adamw_lr5e-4_extra
    # CUDA_VISIBLE_DEVICES=1 python3 Trainer.py --DR LAB4_Dataset/ --save_root  saved_models/ --device cuda --batch_size 12 \
    #                 --num_workers 32 --lr 1e-2 --optim SGD --kl_anneal_type Cyclical --num_epoch 500 --tfr 0.3\
    #                 --exp_name sgd_lr2e-3_extra
    # CUDA_VISIBLE_DEVICES=0 python3 Trainer.py --DR LAB4_Dataset/ --save_root  saved_models/ --device cuda --batch_size 20 \
    #                 --num_workers 32 --lr 5e-4 --optim AdamW --kl_anneal_type Cyclical --num_epoch 500 --tfr 0.3\
    #                 --exp_name with_sigmoid
fi
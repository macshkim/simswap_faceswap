#!/bin/bash

python -m simswap_lite.train_faceswap \
--data_dir /home/ubuntu/data/vggface2_crop_arcfacealign_224 \
--output_dir runs/id_encoder \
--epochs 15 \
--batch_size 256 \
--lr 1e-3

python -m simswap_lite.train_simswap \
--data_dir /home/ubuntu/data/vggface2_crop_arcfacealign_224 \
--id_ckpt runs/id_encoder/best_encoder.pt \
--output_dir runs/simswap_lite \
--epochs 25 \
--batch_size 64 \
--lr_g 2e-4 \
--lr_d 2e-4

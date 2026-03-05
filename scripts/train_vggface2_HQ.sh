#!/bin/bash

python -m simswap_lite.train_faceswap \
--data_dir /home/ubuntu/data/VGGface2_None_norm_512_true_bygfpgan \
--output_dir runs_vggface2_hq/id_encoder \
--epochs 15 \
--batch_size 256 \
--lr 1e-3

python -m simswap_lite.train_simswap \
--data_dir /home/ubuntu/data/VGGface2_None_norm_512_true_bygfpgan \
--id_ckpt runs_vggface2_hq/id_encoder/best_encoder.pt \
--output_dir runs_vggface2_hq/simswap_lite \
--epochs 25 \
--batch_size 64 \
--lr_g 2e-4 \
--lr_d 2e-4

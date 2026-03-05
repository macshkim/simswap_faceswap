# SimSwap based Face Swap System

An AI face swap system built based-on the SimSwap framework (https://github.com/neuralchen/SimSwap), trained on the LFW People dataset (https://www.kaggle.com/datasets/atulanandjha/lfwpeople).

---

## Work History

1. Read papers to get some ideas and to find the most suitable model (SimSwap)
2. Data preparation: download LFW dataset, preprocess it
3. SimSwap Lite: implement the simplified version, training on LFW dataset and test it
4. Issues: LFW data, which is not a dedicated swap dataset, and the simplified model, which is not fully reproduced the original paper 
5. SimSwap Lite training on VGGFace2-224 (zipped 10GB)
6. SimSwap Lite training on VGGFace2-HQ (zipped 90GB)
7. SimSwap Lite image and video inference
7. Original SimSwap set-up and inference for comparison
8. Update Github repo
8. ***SimSwap Lite ONNX export for the inference of images and videos (future work)***
7. ***SimSwap Lite image and video ONNX inference (future work)***. 
9. ***SimSwap Advanced: further improve Lite (future work)***

---

## SimSwap-Lite

**Two-stage pipeline system**:

1. **Identity pretrained encoder** (source image -> embedding): `train_faceswap.py`
2. **Conditional face generator (target image + embedding -> swapped image) + discriminator**: `train_simswap.py`

- This is a **compact SimSwap-inspired architecture**, not a paper-faithful reproduction.
- The generator performs **identity-conditioned synthesis in aligned face space**.
- The discriminator prevents overfitting, and improves realism and blending in face space.

---

## Project Structure

```
simswap_faceswap/
├── preprocess/
│   ├── preprocess.py     # preprocess (detect, align, crop)
│   └── preprocess_lfw.py # preprocess LFW dataset
├── scripts/              # shell scripts for training
├── simswap_lite/
│   ├── simswap_style_models.py # identity encoder, AdaIN generator, patch discriminator
│   ├── train_faceswap.py       # trains a lightweight identity encoder
│   ├── train_simswap.py        # stage-2 GAN training
│   ├── infer_simswap.py        # image inference
│   └── video_simswap.py        # video inference with temporal smoothing
├── requirements.txt      # Python dependencies
├── checkpoints/          # saved model files
├── input_samples/        # input sample files
└── output_results/       # model output files
```

---

## Setup

```bash
pip install -r requirements.txt
```
Training is performed on GPU 1x H100 (80 GB PCIe)

---

## Dataset Preparation
1. Download the LFW dataset from Kaggle: https://www.kaggle.com/datasets/atulanandjha/lfwpeople
2. Unzip the dataset into `data/lfw/lfw_funneled/`
3. Preprocess LFW dataset: detect faces, align, crop, and write metadata CSV
```bash
python -m preprocess.preprocess_lfw --lfw-dir data/lfw --out-dir data/lfw_aligned --size 256
```

---

## Training on LFW

Training process is:
- sample **identity source/target pairs**
- preserve **target pose/expression** (target image as generator input)
- inject **source identity embedding** from the pretrained encoder
- train with:
  - adversarial loss
  - reconstruction loss
  - identity cosine loss
  - perceptual loss

```bash
# Train identity encoder
python -m simswap_lite.train_faceswap \
  --data_dir data/lfw_aligned/lfw_funneled \
  --output_dir runs/id_encoder \
  --epochs 15 \
  --batch_size 256 \
  --lr 1e-3
```

```bash
# Train SimSwap-style generator
python -m simswap_lite.train_simswap \
  --data_dir /home/ubuntu/data/lfw_aligned/lfw_funneled \
  --id_ckpt runs/id_encoder/best_encoder.pt \
  --output_dir runs/simswap_lite \
  --epochs 25 \
  --batch_size 64 \
  --lr_g 2e-4 \
  --lr_d 2e-4
```

---

## Image inference

```bash
python -m simswap_lite.infer_simswap \
  --source input_examples/source_1.jpg \
  --target input_examples/target_1.jpg \
  --id_ckpt runs_lfw/id_encoder/best_encoder.pt \
  --swap_ckpt runs_lfw/simswap_lite/best_simswap_lite.pt \
  --output outputs/swapped.png
```

## Video inference

```bash
python video_simswap.py \
  --source input_examples/source_image.png \
  --video input_examples/target_video_1.mp4 \
  --id_ckpt runs_vggface2_224/id_encoder/best_encoder.pt \
  --swap_ckpt runs_vggface2_224/simswap_lite/best_simswap_lite.pt \
  --output outputs/swapped_video.mp4
```

---

## Future Work

- Stronger **identity backbones** (e.g, ArcFace or InsightFace)
- **Identity Injection Module (IIM)** via learned feature modulation from a 512D identity embedding
- **Weak Feature Matching Loss** from intermediate discriminator features
- **ONNX/TensorRT** export for more optimised inference (e.g., CPU inference)

---

## References
- SimSwap paper: *"SimSwap: An Efficient Framework For High Fidelity Face Swapping"* (Chen et al., 2020)
- GitHub: https://github.com/neuralchen/SimSwap
- LFW Dataset: https://www.kaggle.com/datasets/atulanandjha/lfwpeople

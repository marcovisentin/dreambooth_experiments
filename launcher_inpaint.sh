#!/usr/bin/env bash

export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
#export MODEL_NAME="diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
export INSTANCE_DIR="/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/images/tuning_images/dreambooth_dataset/clock"
export MASKS_DIR="/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/images/tuning_images/dreambooth_dataset/clock_masks"
export CLASS_DIR="/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/images/generated_with_prior_loss"
export OUTPUT_DIR="/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/results_inpaint"
export VAL_DIR="/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/images/tuning_images/angry_bird_speaker_inpaint"

accelerate launch train_dreambooth_inpaint.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --instance_mask_dir=$MASKS_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of a sks clock" \
  --resolution=512 \
  --train_batch_size=1 \
  --use_8bit_adam \
  --gradient_checkpointing \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --validation_dir=$VAL_DIR \
  --num_validation_images=5 \
  --validation_epochs=10 \
  --validation_prompt="A sks clock" \
  --gradient_accumulation_steps=2 \
  --mixed_precision="fp16"
#!/bin/bash

# Set the environment variables for model and directories
export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
export INSTANCE_DIR="/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/images/tuning_images/dreambooth_dataset/clock"
export CLASS_DIR="/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/images/generated_with_prior_loss"
export OUTPUT_DIR="/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/results_inpaint"
export VAL_DIR="/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/images/tuning_images/angry_bird_speaker_inpaint"

# Define arrays of learning rates and ranks
learning_rates=(5e-6 1e-6 5e-5 1e-5 5e-4 1e-4 5e-3 1e-3)
ranks=(2 4 8 16 32)

# Loop over each learning rate and rank
for lr in "${learning_rates[@]}"
do
    for rank in "${ranks[@]}"
    do
        echo "Running training with learning rate: $lr and rank: $rank"
        # Call the training script with current learning rate and rank
        accelerate launch train_dreambooth_lora_inpaint.py \
          --pretrained_model_name_or_path=$MODEL_NAME \
          --train_text_encoder \
          --instance_data_dir=$INSTANCE_DIR \
          --class_data_dir=$CLASS_DIR \
          --output_dir=$OUTPUT_DIR \
          --instance_prompt="A sks clock" \
          --resolution=512 \
          --train_batch_size=1 \
          --use_8bit_adam \
          --gradient_checkpointing \
          --learning_rate=$lr \
          --lr_scheduler="constant" \
          --lr_warmup_steps=0 \
          --num_class_images=50 \
          --max_train_steps=800 \
          --validation_dir=$VAL_DIR \
          --num_validation_images=5 \
          --validation_epochs=10 \
          --validation_prompt="A sks clock" \
          --report_to="wandb" \
          --rank=$rank
    done
done

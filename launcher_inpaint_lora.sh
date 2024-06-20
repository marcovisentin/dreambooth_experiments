export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
#export MODEL_NAME="diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
export INSTANCE_DIR="/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/images/tuning_images/dreambooth_dataset/clock"
export CLASS_DIR="/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/images/generated_with_prior_loss"
export OUTPUT_DIR="/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/results_inpaint"
export VAL_DIR="/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/images/tuning_images/angry_bird_speaker_inpaint"

accelerate launch train_dreambooth_lora_inpaint.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="A sks clock" \
  --resolution=512 \
  --train_batch_size=1 \
  --use_8bit_adam \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=50 \
  --max_train_steps=800 \
  --validation_dir=$VAL_DIR \
  --num_validation_images=5 \
  --validation_epochs=10 \
  --validation_prompt="A sks clock" \
  --report_to="wandb"\
  --rank=4

# Learning rate = [5e-6, 1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3] , rank = [2, 4, 6, 8]
# Ensure it logs the information
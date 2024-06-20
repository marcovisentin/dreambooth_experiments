#export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/results_sdxl"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
export INSTANCE_DIR="/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/images/tuning_images/dreambooth_dataset/clock"

#export MASKS_DIR="/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/images/tuning_images/dreambooth_dataset/clock_masks"
#export CLASS_DIR="/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/images/generated_with_prior_loss"
#export VAL_DIR="/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/images/tuning_images/angry_bird_speaker_inpaint"

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks clock" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks clock on a counter" \
  --validation_epochs=25 \
  --seed="0" \
  --report_to="wandb" \
  --train_text_encoder
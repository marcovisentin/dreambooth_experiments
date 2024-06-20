#export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export MODEL_NAME="stabilityai/stable-diffusion-2"
export INSTANCE_DIR="/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/images/tuning_images/angry_birds_lamp"
export OUTPUT_DIR="/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/results_sdxl"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="An sks speaker" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A sks speaker" \
  --validation_epochs=50 \
  --seed="0" \
  --train_text_encoder \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --class_data_dir="path/to/class/images/generated_with_prior_loss" \
  --class_prompt="A speaker" \
 
  
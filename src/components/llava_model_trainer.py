import os
import subprocess

def setup_environment():
    """Set up the required environment for training."""
    # Uninstall and reinstall specific transformers version
    try:
        print("Uninstalling existing 'transformers' library...")
        subprocess.run(["pip", "uninstall", "transformers", "-y"], check=True)
        
        print("Installing 'transformers==4.45.0'...")
        subprocess.run(["pip", "install", "transformers==4.45.0"], check=True)
        
        print("Installing 'wandb' for logging...")
        subprocess.run(["pip", "install", "wandb"], check=True)

        # Login to wandb
        wandb_api_key = "b35f4e6a996419a4b52f4db73d5b663b8fb03de1"
        print("Logging in to W&B...")
        subprocess.run(["wandb", "login", wandb_api_key], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while setting up the environment: {e}")
        raise

def train_model():
    """Run the model training."""
    distributed_args = os.getenv("DISTRIBUTED_ARGS", "--nproc_per_node=2")  # Set as needed
    train_command = f"""
    torchrun {distributed_args} train.py \
        --model_id llava-1.5-7b \
        --data_path '/content/Final Training Jsons/Final Training Jsons/label/traindataset_8cls.json' \
        --eval_data_path '' \
        --image_folder '/content/merged_total-4874img' \
        --video_folder '' \
        --num_frames 8 \
        --output_dir ./checkpoints/llava_1.5_7blora16 \
        --report_to wandb \
        --run_name llava7blora32 \
        --deepspeed ./ds_configs/zero2.json \
        --bf16 True \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --eval_strategy "epoch" \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --learning_rate 5e-4 \
        --weight_decay 0.0 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "linear" \
        --logging_steps 1 \
        --tf32 False \
        --model_max_length 128 \
        --gradient_checkpointing True \
        --dataloader_num_workers 2 \
        --train_vision_encoder True \
        --use_vision_lora True \
        --train_vision_projector False \
        --use_lora True \
        --q_lora True \
        --lora_r 4 \
        --lora_alpha 8
    """
    try:
        print("Starting training...")
        os.system(train_command)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise

if __name__ == "__main__":
    setup_environment()
    train_model()

# export MODEL_NAME="../pretrain/stable-video-diffusion-img2vid-xt" # SD官方模型路径 
# export DATASET_NAME="../data/nusc_video_front.pkl"

export MODEL_NAME="examples/world_model/demo_model/img2video_1024_14f" # SD官方模型路径 
export DATASET_NAME="../data/sample_nusc_video_all_cam_train.pkl"

WORK=nusc_deepspeed_svd_front_576320_30f_cond_dome
EXP=nusc_deepspeed_svd_front_cond_dome


accelerate launch --config_file examples/world_model/configs/4_gpus_deepspeed_zero2.yaml --main_process_port 12000 examples/world_model/custom_train_video_cond_dome_deepspeed_vista.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --enable_xformers_memory_efficient_attention \
  --max_train_steps=80000 \
  --learning_rate=5e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --output_dir="work_dirs/$WORK" \
  --dataloader_num_workers 2 \
  --nframes 5 \
  --nframes_past 1 \
  --fps 2 \
  --image_height 320 \
  --image_width 576 \
  --conditioning_dropout_prob=0.2 \
  --seed_for_gen=42 \
  --ddim \
  --checkpointing_steps 5000 \
  --tracker_project_name $EXP \
  --load_from_pkl \
  --gradient_checkpointing \
  --report_to wandb \



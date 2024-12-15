# train an X3D model from scratch
# python train_from_scratch.py --model_name x3d \
#                              --num_classes 50 \
#                              --num_epochs 100 \
#                              --learning_rate 0.0001 \
#                              --batch_size 12 \
#                              --train_data_path /workspace/sign-language-data/video-frames-2.0/train \
#                              --validation_data_path /workspace/sign-language-data/video-frames-2.0/validation \
#                              --warmup_steps 0. \
#                              --save_ckpt_every 100 \
#                              --accumulation_steps 2 \
#                              --device cuda:0 \
#                              --use_wandb true 

# train an X3D model from scratch
python -m train.train_from_scratch --model_name x3d \
                                    --num_classes 83 \
                                    --num_epochs 100 \
                                    --learning_rate 0.0001 \
                                    --batch_size 12 \
                                    --train_data_path /workspace/sign-language-data/data_extract_with_padding/train \
                                    --validation_data_path /workspace/sign-language-data/data_extract_with_padding/val \
                                    --warmup_steps 0. \
                                    --save_ckpt_every 100 \
                                    --accumulation_steps 2 \
                                    --device cuda:0 \
                                    --use_wandb true                                                                                                
# python -m train.last_layer_distillation --teacher_model_name VideoMAE-v2-small \
#                                         --student_model_name i3d \
#                                         --num_classes 50 \
#                                         --num_epochs 100 \
#                                         --learning_rate 0.0005 \
#                                         --batch_size 4 \
#                                         --alpha 0. \
#                                         --temperature 5 \
#                                         --teacher_model_path /workspace/pytorch_gpu/sign_language_code/model_ckpts/VideoMAE-2-small-finetuned/model_epoch_8.pth \
#                                         --student_model_path /workspace/pytorch_gpu/sign_language_code/model_ckpts/i3d/rgb_imagenet.pt \
#                                         --train_data_path_1 /workspace/sign-language-data/video-frames-2.0/validation \
#                                         --validation_data_path_1 /workspace/sign-language-data/video-frames-2.0/validation \
#                                         --save_ckpt_dir /workspace/pytorch_gpu/sign_language_code/model_ckpts/last_layer_distillation \
#                                         --warmup_steps 0.01 \
#                                         --save_ckpt_every 100 \
#                                         --accumulation_steps 6 \
#                                         --device cuda:0 \
#                                         --use_wandb true

python -m train.last_layer_distillation --teacher_model_name VideoMAE-v2-base \
                                        --student_model_name i3d \
                                        --num_classes 83 \
                                        --num_epochs 100 \
                                        --learning_rate 0.0005 \
                                        --batch_size 4 \
                                        --alpha 0. \
                                        --temperature 5 \
                                        --teacher_model_path /workspace/pytorch_gpu/sign_language_code/model_ckpts/VideoMAE-2-base-ss-finetuned/model_epoch_33.pth \
                                        --student_model_path /workspace/pytorch_gpu/sign_language_code/model_ckpts/i3d/rgb_imagenet.pt \
                                        --train_data_path_1 /workspace/sign-language-data/data_cut_full_frames/val \
                                        --validation_data_path_1 /workspace/sign-language-data/data_cut_full_frames/val \
                                        --save_ckpt_dir /workspace/pytorch_gpu/sign_language_code/model_ckpts/last_layer_distillation \
                                        --warmup_steps 0.01 \
                                        --save_ckpt_every 100 \
                                        --accumulation_steps 6 \
                                        --device cuda:0 \
                                        --use_wandb true
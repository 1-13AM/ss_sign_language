# train an I3D model from scratch
python -m train.train_from_scratch --model_name i3d \
                                    --num_classes 131 \
                                    --num_epochs 30 \
                                    --learning_rate 0.0001 \
                                    --batch_size 4 \
                                    --train_data_path_1 /workspace/sign-language-data/data_cut_full_frames/train \
                                    --train_data_path_2 /workspace/sign-language-data/data_50_labels_full_frames/train \
                                    --validation_data_path_1 /workspace/sign-language-data/data_cut_full_frames/val \
                                    --validation_data_path_2 /workspace/sign-language-data/data_50_labels_full_frames/val \
                                    --warmup_steps 0. \
                                    --accumulation_steps 6 \
                                    --save_ckpt_every 50 \
                                    --device cuda:0 \
                                    --class_balance true \
                                    --use_wandb true




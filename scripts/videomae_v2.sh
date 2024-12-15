
python -m train.train_from_scratch --model_name VideoMAE-v2 \
                                --num_classes 129 \
                                --num_epochs 30 \
                                --learning_rate 0.0001 \
                                --batch_size 4 \
                                --scheduler StepLR \
                                --model_path /workspace/pytorch_gpu/sign_language_code/model_ckpts/VideoMAE-2-base-ss-finetuned/model_epoch_33.pth \
                                --train_data_path_1 /workspace/sign-language-data/data_83_labels_full_frames/train \
                                --validation_data_path_1 /workspace/sign-language-data/data_83_labels_full_frames/val \
                                --warmup_steps 0.01 \
                                --save_ckpt_every 100 \
                                --save_ckpt_dir /workspace/pytorch_gpu/sign_language_code/model_ckpts/VideoMAE-2-131-labels-more-data \
                                --accumulation_steps 8 \
                                --class_balance true \
                                --device cuda:0 \
                                --use_wandb true     

                                
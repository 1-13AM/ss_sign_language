
python -m train.train_from_scratch --model_name VideoMAE-v2 \
                                --num_classes 131 \
                                --num_epochs 50 \
                                --learning_rate 0.00015 \
                                --batch_size 4 \
                                --scheduler StepLR \
                                --model_path MODEL_PATH \
                                --train_data_path_1 TRAIN_DATA_PATH_1 \
                                --train_data_path_2 TRAIN_DATA_PATH_2 \
                                --validation_data_path_1 VAL_DATA_PATH_1 \
                                --validation_data_path_2 VAL_DATA_PATH_2 \
                                --warmup_steps 0.01 \
                                --save_ckpt_every 1 \
                                --save_ckpt_dir model_ckpts/VideoMAE-v2-base-sl \
                                --accumulation_steps 8 \
                                --class_balance true \
                                --device cuda:0 \
                                --use_wandb true     

                                
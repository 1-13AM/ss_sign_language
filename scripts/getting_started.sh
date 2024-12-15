# install vit-b-based videomae-v2 checkpoint
wget https://download.openmmlab.com/mmaction/v1.0/recognition/videomaev2/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-3e7f93b2.pth -P model_ckpts/VideoMAE-2-base

# script to rename layer names from source
python -m utils.rename_layer_names --weight_path model_ckpts/VideoMAE-2-base/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-3e7f93b2.pth
                                   --new_weight_path model_ckpts/VideoMAE-2-base/VideoMAE_v2_base.pth

# install vit-b-based videomae-v2 checkpoint
wget https://download.openmmlab.com/mmaction/v1.0/recognition/videomaev2/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-3e7f93b2.pth -P model_ckpts/VideoMAE-2-base-2

pip install transformers==4.45.0 datasets==3.0.1 evaluate==0.4.3 accelerate==0.34.2 --force-reinstall
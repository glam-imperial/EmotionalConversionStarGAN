python classifier_train.py --epochs 100
python train_main.py --recon_only --config ./configs/config_step1.yaml
python train_main.py --checkpoint ./checkpoints/model_step1/200000.ckpt --load_emo ./checkpoints/cls_checkpoint.ckpt --config ./configs/config_step2.yaml --alter

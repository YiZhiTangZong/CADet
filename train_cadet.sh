gpuid=0

CUDA_VISIBLE_DEVICES=${gpuid} python -u train.py --config config/train/cadet_crack500.yaml

gpuid=0

CUDA_VISIBLE_DEVICES=${gpuid} python -u test.py --config config/test/cadet_crack500.yaml

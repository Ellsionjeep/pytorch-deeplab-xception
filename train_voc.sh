CUDA_VISIBLE_DEVICES=1 python train.py --backbone resnet --lr 0.007 --workers 0 --epochs 50 --batch-size 16 --gpu-ids 1 --checkname deeplab-resnet --eval-interval 1 --dataset pascal

import argparse
import os

arg = "CUDA_VISIBLE_DEVICES=3 python train_cifar.py --lsqforward True --weight-decay 5e-5"
print("arg")
for i in [8, 4, 3, 2]:
    os.system("{} --awbits {}".format(arg, i))
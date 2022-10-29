import os

# for epoch in [199, 100, 20, 1, 0]:
#     for bit in [4, 5, 6, 7, 8]:
#         os.makedirs("20221026/cifar10/{}/{}".format(epoch, bit), exist_ok=True)
#
#         os.system("python train_cifar.py --training-bit plt --plt-bit "
#                   "TTT8{}8 --training-strategy checkpoint_full_precision --checkpoint-epoch {}".format(bit, epoch))
#
#     os.makedirs("20221026/cifar10/{}/grad_weight".format(epoch), exist_ok=True)
#     os.system("python train_cifar.py "
#               "--training-bit plt --plt-bit TTT848 --twolayers_gradweight True "
#               "--training-strategy checkpoint_full_precision --checkpoint-epoch {}".format(epoch))
    
for epoch in [89, 75, 20, 1, 0]:
    for bit in [4, 5, 6, 7, 8]:
        os.makedirs("20221026/imagenet/{}/{}".format(epoch, bit), exist_ok=True)

        os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python train_imagenet.py --gpu-num 7 --b 100 --obs 700 --lr 0.7 --master_addr 127.0.0.2 --master_port 66666 "
                  " --arch resnet18 --training-bit plt --plt-bit "
                  "TTT8{}8 --training-strategy checkpoint_full_precision --checkpoint-epoch {}".format(bit, epoch))

    os.makedirs("20221026/imagenet/{}/grad_weight".format(epoch), exist_ok=True)
    os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python train_imagenet.py --gpu-num 7 --b 100 --obs 700 --lr 0.7 --master_addr 127.0.0.2 --master_port 66666 "
              " --arch resnet18 --training-bit plt --plt-bit TTT848 --twolayers_gradweight True "
              "--training-strategy checkpoint_full_precision --checkpoint-epoch {}".format(epoch))


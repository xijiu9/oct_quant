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

        os.system("python train_imagenet.py --b 100 --obs 800 --lr 0.8"
                  " --arch resnet18 --training-bit plt --plt-bit "
                  "TTT8{}8 --training-strategy checkpoint_full_precision --checkpoint-epoch {}".format(bit, epoch))

    os.makedirs("20221026/imagenet/{}/grad_weight".format(epoch), exist_ok=True)
    os.system("python train_imagenet.py --b 100 --obs 800 --lr 0.8"
              " --arch resnet18 --training-bit plt --plt-bit TTT848 --twolayers_gradweight True "
              "--training-strategy checkpoint_full_precision --checkpoint-epoch {}".format(epoch))

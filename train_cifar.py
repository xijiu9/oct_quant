import argparse
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected in train_cifar.')


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--twolayers_gradweight', type=str2bool, default=False, help='use two 4 bit to simulate a 8 bit')
parser.add_argument('--twolayers_gradinputt', type=str2bool, default=False, help='use two 4 bit to simulate a 8 bit')
parser.add_argument('--lsqforward', type=str2bool, default=False, help='apply LSQ')
parser.add_argument('--training-bit', type=str, default='', help='weight number of bits',
                    choices=['full_precision', 'only_weight', 'all4bit', 'forward8', 'forward4'])

parser.add_argument('--training-strategy', default='scratch', type=str, metavar='strategy',
                    choices=['scratch', 'checkpoint', 'gradually', 'checkpoint_from_zero', 'checkpoint_full_precision'])
parser.add_argument('--checkpoint_epoch_full_precision', type=int, default=0, help='full precision')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='N',
                    help='number of total epochs to run')

args = parser.parse_args()

if args.training_bit == 'full_precision':
    bbits, bwbits, awbits = 8, 8, 8
elif args.training_bit == 'only_weight':
    bbits, bwbits, awbits = 8, 4, 4
elif args.training_bit == 'all4bit':
    bbits, bwbits, awbits = 4, 4, 4
elif args.training_bit == 'forward8':
    bbits, bwbits, awbits = 4, 4, 8
elif args.training_bit == 'forward4':
    bbits, bwbits, awbits = 8, 8, 4
else:
    bbits, bwbits, awbits = 0, 0, 0

if args.twolayers_gradweight:
    assert bwbits == 4
if args.twolayers_gradinputt:
    assert bbits == 4

if args.twolayers_gradweight and args.twolayers_gradinputt:
    method = 'twolayer'
elif args.twolayers_gradweight and not args.twolayers_gradinputt:
    method = 'twolayer_weightonly'
elif not args.twolayers_gradweight and not args.twolayers_gradinputt:
    method = 'full'

# model = "results_cifar/{}/models/checkpoint-acc-93.0.pth.tar".format(method)
model = "results_cifar/{}/models/checkpoint-93.pth.tar".format(method)
if args.training_strategy == 'checkpoint_full_precision':
    if args.checkpoint_epoch_full_precision == 0:
        print("should bigger than 0")
        exit(0)
    else:
        model = "results_cifar/full/models/checkpoint-{}.pth.tar".format(args.checkpoint_epoch_full_precision)


workplace = "results_cifar/" + method
if not os.path.exists(workplace):
    os.mkdir(workplace)

# if args.net == 'qat':
#     arg = "-c quantize --qa=True --qw=True --qg=False"
#
#     os.system("python main.py --dataset cifar10 --arch preact_resnet56 --gather-checkpoints --workspace results/{} "
#               "--label-smoothing 0  --warmup 0 "
#               "--weight-decay 1e-4 {} ~/data/cifar10".format(method, arg))
# elif args.net == 'ptq':

arg = " -c quantize --qa=True --qw=True --qg=True --persample=False --hadamard=False"

if args.lsqforward:
    arg = " -c quantize --qa=True --qw=True --qg=True --persample=False --hadamard=False"
    method = 'lsq/lsq_' + str(bbits) + str(bwbits) + str(awbits)
    model = "results_cifar/{}/models/checkpoint-93.3.pth.tar".format(method)
    workplace = "results_cifar/" + method

if args.training_strategy == 'checkpoint' or args.training_strategy == "checkpoint_from_zero" \
        or args.training_strategy == "checkpoint_full_precision":
    os.system("python main.py --dataset cifar10 --arch preact_resnet56 --gather-checkpoints "
              "--workspace {} --twolayers_gradweight {} --twolayers_gradinputt {} "
              "--lsqforward {} {} ~/data/cifar10 --training-strategy {} --epochs {} "
              "--resume {} --bbits {} --bwbits {} --abits {} --wbits {} --weight-decay {}".format(
                workplace, args.twolayers_gradweight, args.twolayers_gradinputt,
                args.lsqforward, arg, args.training_strategy, args.epochs,
                model, bbits, bwbits, awbits, awbits, args.weight_decay))
else:
    os.system("python main.py --dataset cifar10 --arch preact_resnet56 --gather-checkpoints "
              "--workspace {} --twolayers_gradweight {} --twolayers_gradinputt {} "
              "--lsqforward {} {} ~/data/cifar10 --training-strategy {} --epochs {} "
              "--bbits {} --bwbits {} --abits {} --wbits {} --weight-decay {}".format(
                workplace, args.twolayers_gradweight, args.twolayers_gradinputt,
                args.lsqforward, arg, args.training_strategy, args.epochs,
                bbits, bwbits, awbits, awbits, args.weight_decay))
    # {} ~/data/cifar10 --resume {}".format(args.twolayersweight, args.lsqforward, method, arg, model))

# elif args.net == 'psq':
#     arg = "-c quantize --qa=True --qw=True --qg=True --persample=True --hadamard=False --bbits={}".format(args.bbits)
#
#     os.system("python main.py --dataset cifar10 --arch preact_resnet56 --gather-checkpoints --workspace results/{} "
#               "--lr 0.1 --momentum 0.9 --label-smoothing 0  --warmup 0 "
#               "--weight-decay 1e-4 {} ~/data/cifar10 --resume {}".format(method, arg, model))
#               # "--weight-decay 1e-4 {} ~/data/cifar10".format(method, arg))
#
#
# elif args.net == 'bhq':
#     arg = "-c quantize --qa=True --qw=True --qg=True --persample=True --hadamard=True --bbits={}".format(args.bbits)
#
#     os.system("python main.py --dataset cifar10 --arch preact_resnet56 --gather-checkpoints --workspace results/{} "
#               "--lr 0.1 --momentum 0.9 --label-smoothing 0  --warmup 0 "
#               "--weight-decay 1e-4 {} ~/data/cifar10 --resume {}".format(method, arg, model))
#               # "--weight-decay 1e-4 {} ~/data/cifar10".format(method, arg))

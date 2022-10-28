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
        raise argparse.ArgumentTypeError('Boolean value expected in train_imagenet.')

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--twolayers_gradweight', '--2gw', type=str2bool, default=False, help='use two 4 bit to simulate a 8 bit')
parser.add_argument('--twolayers_gradinputt', '--2gi', type=str2bool, default=False, help='use two 4 bit to simulate a 8 bit')
parser.add_argument('--lsqforward', type=str2bool, default=False, help='apply LSQ')

parser.add_argument('--training-bit', type=str, default='', help='weight number of bits',
                    choices=['exact', 'qat', 'all8bit', 'star_weight', 'only_weight', 'weight4', 'all4bit', 'forward8',
                             'forward4', 'plt'])
parser.add_argument('--plt-bit', type=str, default='', help='')
parser.add_argument('--training-strategy', default='scratch', type=str, metavar='strategy',
                    choices=['scratch', 'checkpoint', 'checkpoint_from_zero', 'checkpoint_full_precision'])
parser.add_argument('--checkpoint-epoch', type=int, default=0, help='full precision')
parser.add_argument('--checkpoint_epoch_full_precision', type=int, default=0, help='full precision')
parser.add_argument('--clip-grad', type=float, default=10, help='clip gradient to 0.01(CIFAR)')
parser.add_argument('--amp', action='store_true', help='Run model AMP (automatic mixed precision) mode.')

parser.add_argument('--lr', type=float, default=0.1, help='clip gradient to 0.01(CIFAR)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--warmup', default=20, type=int, metavar='E', help='number of warmup epochs')

args = parser.parse_args()

arg = " -c quantize --qa=True --qw=True --qg=True"

if args.training_bit == 'all8bit':
    bbits, bwbits, awbits = 8, 8, 8
elif args.training_bit == 'exact':
    arg = ''
    bbits, bwbits, awbits = 0, 0, 0
elif args.training_bit == 'qat':
    bbits, bwbits, awbits = 0, 0, 8
    arg = "-c quantize --qa=True --qw=True --qg=False"
elif args.training_bit == 'star_weight':
    bbits, bwbits, awbits = 8, 4, 4
elif args.training_bit == 'only_weight':
    bbits, bwbits, awbits = 8, 4, 8
    arg = "-c quantize --qa=False --qw=False --qg=True"
elif args.training_bit == 'weight4':
    bbits, bwbits, awbits = 8, 4, 8
elif args.training_bit == 'all4bit':
    bbits, bwbits, awbits = 4, 4, 4
elif args.training_bit == 'forward8':
    bbits, bwbits, awbits = 4, 4, 8
elif args.training_bit == 'forward4':
    bbits, bwbits, awbits = 8, 8, 4
else:
    bbits, bwbits, awbits = 0, 0, 0
    if args.training_bit == 'plt':
        arg = "-c quantize --qa={} --qw={} --qg={}".format(args.plt_bit[0], args.plt_bit[1], args.plt_bit[2])
        bbits, bwbits, awbits = args.plt_bit[5], args.plt_bit[4], args.plt_bit[3]

if args.twolayers_gradweight:
    assert int(bwbits) == 4
if args.twolayers_gradinputt:
    assert int(bbits) == 4

if args.twolayers_gradweight and args.twolayers_gradinputt:
    method = 'twolayer'
elif args.twolayers_gradweight and not args.twolayers_gradinputt:
    method = 'twolayer_weightonly'
elif not args.twolayers_gradweight and not args.twolayers_gradinputt:
    method = args.training_bit

arg_epochs = 200
if args.training_strategy == 'checkpoint' or args.training_strategy == 'checkpoint_from_zero':
    model = 'results/cifar/{}/models/checkpoint-{}.pth.tar'.format(args.training_bit, args.checkpoint_epoch)
    # arg_epochs = 1
elif args.training_strategy == 'checkpoint_full_precision':
    model = 'results/cifar/exact/models/saves/checkpoint-{}.pth.tar'.format(args.checkpoint_epoch)
    # arg_epochs = 1
else:
    model = 'pass'

if args.amp:
    amp_control = '--amp --static-loss-scale 128'
else:
    amp_control = ''

os.system("python ./main.py --arch preact_resnet56 --gather-checkpoints --checkpoint-epoch {} \
            --lr {} --resume {} --dataset cifar10 --momentum 0.9 --weight-decay {} --epoch {}\
            --warmup {} {}  ~/data/cifar10 --workspace ./results/cifar/{}/models \
            {} --print-freq 300 --clip-grad {} \
            --bbits {} --bwbits {} --abits {} --wbits {} --lsqforward {} \
            --twolayers-gradweight {} --twolayers-gradinputt {}"
            .format(args.checkpoint_epoch, args.lr, model, args.weight_decay, arg_epochs,
                    args.warmup, arg, args.training_bit,
                    amp_control, args.clip_grad,
                    bbits, bwbits, awbits, awbits, args.lsqforward,
                    args.twolayers_gradweight, args.twolayers_gradinputt))
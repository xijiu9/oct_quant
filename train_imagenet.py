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
parser.add_argument('--luq', type=str2bool, default=False, help='use luq for backward')
parser.add_argument('--lsqforward', type=str2bool, default=False, help='apply LSQ')

parser.add_argument('--training-bit', type=str, default='', help='weight number of bits',
                    choices=['exact', 'qat', 'all8bit', 'star_weight', 'only_weight', 'weight4', 'all4bit', 'forward8',
                             'forward4', 'plt'])
parser.add_argument('--plt-bit', type=str, default='', help='')
parser.add_argument('--training-strategy', default='scratch', type=str, metavar='strategy',
                    choices=['scratch', 'checkpoint', 'checkpoint_from_zero',
                             'checkpoint_full_precision', 'checkpoint_full_precision_from_zero'])
parser.add_argument('--checkpoint-epoch', type=int, default=0, help='full precision')
parser.add_argument('--clip-grad', type=float, default=10, help='clip gradient')
parser.add_argument('--amp', action='store_true', help='Run model AMP (automatic mixed precision) mode.')

parser.add_argument('--arch', type=str, default='resnet50', help='clip gradient to 0.01(CIFAR)')
parser.add_argument('--gpu-num', type=int, default=8, help='clip gradient to 0.01(CIFAR)')
parser.add_argument('--lr', type=float, default=1.024, help='clip gradient to 0.01(CIFAR)')
parser.add_argument('--epochs', type=int, default=90, help='clip gradient to 0.01(CIFAR)')
parser.add_argument('--weight-decay', '--wd', default=3.0517578125e-05, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--batch-size', '--b', type=int, default=128, help='clip gradient to 0.01(CIFAR)')
parser.add_argument('--optimizer-batch-size', '--obs', type=int, default=1024, help='clip gradient to 0.01(CIFAR)')
parser.add_argument('--warmup', default=4, type=int, metavar='E', help='number of warmup epochs')

parser.add_argument("--master_port", default=29500, type=int,
                    help="Master node (rank 0)'s free port that needs to "
                         "be used for communciation during distributed "
                         "training")
parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                    help="Master node (rank 0)'s address, should be either "
                         "the IP address or the hostname of node 0, for "
                         "single node multi-proc training, the "
                         "--master_addr can simply be 127.0.0.1")
args = parser.parse_args()

assert args.gpu_num * args.batch_size == 1000 * args.lr
assert args.gpu_num * args.batch_size == args.optimizer_batch_size

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

arg_epochs = args.epochs
if args.training_strategy == 'checkpoint' or args.training_strategy == 'checkpoint_from_zero':
    model = 'results/imagenet/{}/models/checkpoint-{}.pth.tar'.format(args.training_bit, args.checkpoint_epoch)
    arg_epochs = 1
elif args.training_strategy == 'checkpoint_full_precision' or args.training_strategy == 'checkpoint_full_precision_from_zero':
    model = 'results/imagenet/exact/models/saves/checkpoint-{}.pth.tar'.format(args.checkpoint_epoch)
    arg_epochs = 1
else:
    model = 'pass'

if args.amp:
    amp_control = '--amp --static-loss-scale 128'
else:
    amp_control = ''

os.system("python ./multiproc.py --nnodes 1 --node_rank 0 --master_addr {} --master_port {} \
            --nproc_per_node {} ./main.py --arch {} --gather-checkpoints --checkpoint-epoch {} --training-strategy {} \
            --batch-size {} --lr {} --optimizer-batch-size {} --resume {}\
            --warmup {} {}  /data/LargeData/Large/ImageNet --workspace ./results/imagenet/{}/models \
            {} --print-freq 400 --clip-grad {} --epochs {}\
            --bbits {} --bwbits {} --abits {} --wbits {} --weight-decay {} --lsqforward {} \
            --twolayers-gradweight {} --twolayers-gradinputt {} --luq {}"
          .format(args.master_addr, args.master_port,
                  args.gpu_num, args.arch, args.checkpoint_epoch, args.training_strategy, 
                  args.batch_size, args.lr, args.optimizer_batch_size, model,
                  args.warmup, arg, args.training_bit,
                  amp_control, args.clip_grad, arg_epochs,
                  bbits, bwbits, awbits, awbits, args.weight_decay, args.lsqforward,
                  args.twolayers_gradweight, args.twolayers_gradinputt, args.luq))

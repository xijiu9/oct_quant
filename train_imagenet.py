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
parser.add_argument('--twolayers_gradweight', type=str2bool, default=False, help='use two 4 bit to simulate a 8 bit')
parser.add_argument('--twolayers_gradinputt', type=str2bool, default=False, help='use two 4 bit to simulate a 8 bit')
parser.add_argument('--lsqforward', type=str2bool, default=False, help='apply LSQ')
parser.add_argument('--training-bit', type=str, default='', help='weight number of bits',
                    choices=['full_precision', 'only_weight', 'all4bit', 'forward8'])

parser.add_argument('--training-strategy', default='scratch', type=str, metavar='strategy',
                    choices=['scratch', 'checkpoint', 'gradually', 'checkpoint_from_zero', 'checkpoint_full_precision'])
parser.add_argument('--checkpoint_epoch_full_precision', type=int, default=0, help='full precision')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup', type=int, default=4, help='freeze or not the step size update')
parser.add_argument('--lr', type=float, default=0.4, help='freeze or not the step size update')
parser.add_argument('--batch-size', type=int, default=50, help='freeze or not the step size update')

args = parser.parse_args()

if args.training-bit == 'full_precision':
    bbits, bwbits, awbits = 8, 8, 8
elif args.training-bit == 'only_weight':
    bbits, bwbits, awbits = 8, 4, 4
elif args.training-bit == 'all4bit':
    bbits, bwbits, awbits = 4, 4, 4
elif args.training-bit == 'forward8':
    bbits, bwbits, awbits = 4, 4, 8
else:
    bbits, bwbits, awbits = 0, 0, 0


if args.twolayers_gradweight and args.twolayers_gradinputt:
    method = 'twolayer'
elif args.twolayers_gradweight and not args.twolayers_gradinputt:
    method = 'twolayer_weightonly'
elif not args.twolayers_gradweight and not args.twolayers_gradinputt:
    method = 'full'

arg = "-c quantize --qa=True --qw=True --qg=True --persample=False --hadamard=False"

if args.training_strategy == 'checkpoint' or args.training_strategy == "checkpoint_from_zero" \
        or args.training_strategy == "checkpoint_full_precision":
    os.system("python ./multiproc.py --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 "
              "--nproc_per_node 4 ./main.py --arch resnet18 --gather-checkpoints "
              "--workspace /store/results/{} --batch-size {} --lr {} --warmup {} {} {} "
              "--bbits {} --bwbits {} --abits --wbits --resume {} "
              "--twolayers_gradweight {} --twolayers_gradinputt {} --training-strategy {} ".format(
                method, args.batch_size, args.lr, args.warmup, arg, ???
                bbits, bwbits, awbits, awbits, model,
                args.twolayers_gradweight, args.twolayers_gradinputt, args.training_strategy))

else:
    os.system("python ./multiproc.py --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 "
              "--nproc_per_node 4 ./main.py --arch resnet18 --gather-checkpoints "
              "--workspace /store/results/{} --batch-size {} --lr {} --warmup {} {} {} "
              "--bbits {} --bwbits {} --abits --wbits "
              "--twolayers_gradweight {} --twolayers_gradinputt {} --training-strategy {} ".format(
                method, args.batch_size, args.lr, args.warmup, arg, ???
                bbits, bwbits, awbits, awbits,
                args.twolayers_gradweight, args.twolayers_gradinputt, args.training_strategy))
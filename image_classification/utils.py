import os
import numpy as np
import torch
import shutil
import torch.distributed as dist
from torch import tensor
import random

from image_classification.preconditioner import ScalarPreconditioner, DiagonalPreconditioner, \
    BlockwiseHouseholderPreconditioner, ScalarPreconditionerAct, TwoLayerWeightPreconditioner, lsq_per_tensor
from matplotlib import pyplot as plt

class QuantizationConfig:
    def __init__(self):
        self.quantize_activation = True
        self.quantize_weights = True
        self.quantize_gradient = True
        self.activation_num_bits = 8
        self.weight_num_bits = 8
        self.bias_num_bits = 16
        self.backward_num_bits = 8
        self.bweight_num_bits = 8
        self.backward_persample = False
        self.biased = False
        self.grads = None
        self.acts = None
        self.biprecision = True
        self.freeze_step = 0
        self.twolayer_weight = False
        self.twolayer_inputt = False
        self.lsqforward = False

        self.epoch = 0
        self.debug = False
        self.args = None
        self.valid_history = []


    def activation_preconditioner(self):
        return lambda x: ScalarPreconditionerAct(x, self.activation_num_bits)

    def weight_preconditioner(self):
        return lambda x: ScalarPreconditioner(x, self.weight_num_bits)

    def bias_preconditioner(self):
        return lambda x: ScalarPreconditioner(x, self.bias_num_bits)

    def activation_gradient_preconditioner(self):
        if self.twolayer_inputt:
            return lambda x: TwoLayerWeightPreconditioner(x, self.backward_num_bits)
        else:
            return lambda x: ScalarPreconditioner(x, self.backward_num_bits)

    def weight_gradient_preconditioner(self):
        if self.twolayer_weight:
            return lambda x: TwoLayerWeightPreconditioner(x, self.bweight_num_bits)
        return lambda x: ScalarPreconditioner(x, self.bweight_num_bits)

config = QuantizationConfig()

def checkNAN(x, s=''):
    N = torch.isnan(x)
    cN = torch.count_nonzero(N)
    if cN != 0:

        print("NAN!!!{}".format(s))
        print(cN)
        print(x.shape)
        print(x)
        print(config.valid_history)

def should_backup_checkpoint(args):
    def _sbc(epoch):
        return args.gather_checkpoints  # and (epoch < 10 or epoch % 10 == 0)

    return _sbc

def set_seed_epoch(seed, epoch):
    torch.manual_seed(seed+epoch)
    torch.cuda.manual_seed(seed+epoch)
    np.random.seed(seed=seed+epoch)
    random.seed(seed+epoch)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', checkpoint_dir='./', backup_filename=None):
    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        filename = os.path.join(checkpoint_dir, filename)
        # print("SAVING {}".format(filename))
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(checkpoint_dir, 'model_best.pth.tar'))
        if backup_filename is not None:
            shutil.copyfile(filename, os.path.join(checkpoint_dir, backup_filename))


def timed_generator(gen):
    start = time.time()
    for g in gen:
        end = time.time()
        t = end - start
        yield g, t
        start = time.time()


def timed_function(f):
    def _timed_function(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        return ret, time.time() - start

    return _timed_function


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    return rt


def dict_add(x, y):
    if x is None:
        return y
    return {k: x[k] + y[k] for k in x}


def dict_minus(x, y):
    return {k: x[k] - y[k] for k in x}


def dict_sqr(x):
    return {k: x[k] ** 2 for k in x}


def dict_sqrt(x):
    return {k: torch.sqrt(x[k]) for k in x}


def dict_mul(x, a):
    return {k: x[k] * a for k in x}


def dict_clone(x):
    return {k: x[k].clone() for k in x}


def twolayer_linearsample_weight(m1, m2):
    m2 = torch.cat([m2, m2], dim=0)
    m1_len = torch.linalg.norm(m1, dim=1)
    m2_len = torch.linalg.norm(m2, dim=1)
    vec_norm = m1_len.mul(m2_len)

    index, norm_x = sample_index_from_bernouli(vec_norm)
    m1 = m1 / norm_x.unsqueeze(1)

    m1, m2 = m1[index, :], m2[index, :]

    return m1, m2


def twolayer_convsample_weight(m1, m2):
    # print(m1.mean(), m1.max(), m1.min(), m2.mean(), m2.max(), m2.min())
    m1_len, m2_len = m1.mean(dim=(2, 3)).square().sum(dim=1), m2.sum(dim=(2, 3)).square().sum(dim=1)
    vec_norm = m1_len.mul(m2_len)

    checkNAN(vec_norm, 'vec_norm')
    index, norm_x = sample_index_from_bernouli(vec_norm)
    m1 = m1 / norm_x.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    m1, m2 = m1[index, :], m2[index, :]

    return m1, m2


def sample_index_from_bernouli(x):
    # if torch.isnan(x[0]):
    #     print("aduiduidui")
    #     exit(0)
    # print(x.max(), x.min(), x.mean())
    len_x = len(x)
    norm_x = x * len_x / (2 * x.sum())
    # print(norm_x)
    typeflag = 'NoNoNo'
    randflag = torch.rand(1)

    cnt = 0
    while norm_x.max() > 1 and cnt < len_x / 2:
        small_index = torch.nonzero((norm_x < 1)).squeeze()
        small_value = norm_x[small_index]
        cnt = len_x - len(small_index)
        norm_x = torch.clamp(norm_x, 0, 1)
        if small_value.max() == 0 and small_value.min() == 0:
            break
        # print(len(x), cnt)
        small_value = small_value * (len_x // 2 - cnt) / small_value.sum()
        norm_x[small_index] = small_value

        # print("small index is {}, \n small value is {}, cnt is {}".format(small_index, small_value, cnt))
        # print("norm x is {}".format(norm_x))
        # print("sorted norm x is {}".format(norm_x.sort()[0]))
        # print("small index is {}, \n small value is {}, cnt is {}".format(small_index, small_value, cnt))
        # print("sum up to {}".format(norm_x.sum()))
        # print("cnt is {}".format(cnt))
        # print("_______________________________________________________________________________________________________")
        # exit(0)
    # if norm_x.max() > 1 or norm_x.min() < 0:
    #     # if torch.isnan(norm_x[0]):
    #     typeflag = 'debug'
    #     print("We change it to debug mode because of the Bernoulli")
    # if typeflag == 'debug':
    #     with open("debug.txt", "a") as f:
    #         f.write("raw {} is {}\n".format(randflag, x))
    #     with open("debug.txt", "a") as f:
    #         f.write("the after norm {} is {}\n".format(randflag, norm_x))
    # # print("norm x is {}".format(norm_x))

    checkNAN(norm_x, 'norm_x')
    sample_index = torch.bernoulli(norm_x)
    # print("sample_index is {}
    # if typeflag == 'debug':
    #     with open("debug.txt", "a") as f:
    #         f.write("sample index {} is {}\n".format(randflag, sample_index))
    # # index = [x for x in range(len(sample_index)) if sample_index[x] == 1]
    # # try:
    # if sample_index.max() > 1 or sample_index.min() < 0:
    #     print(sample_index)
    #     print(x)

    index = torch.nonzero((sample_index == 1)).squeeze()
    # if typeflag == 'debug':
    #     with open("debug.txt", "a") as f:
    #         f.write("index {} is {}\n".format(randflag, index))
    # print("bernoulli", x, '\n', index, '\n', norm_x, '\n', len(index))
    return index, norm_x


def twolayer_linearsample_input(m1, m2):

    m2 = torch.cat([m2, m2], dim=0)
    m1_len = torch.linalg.norm(m1, dim=1)
    m2_len = torch.linalg.norm(m2, dim=1)
    vec_norm = m1_len.mul(m2_len)

    index, norm_x = sample_index_from_bernouli(vec_norm)
    norm_x[norm_x == 0] = 1e-8
    m1 = m1 / norm_x.unsqueeze(1)

    Ind = torch.zeros_like(m1)
    Ind[index] = 1
    m1 = m1.mul(Ind)
    m1 = m1[0:m1.shape[0] // 2] + m1[m1.shape[0] // 2:]

    return m1


def twolayer_convsample_input(m1, config):
    m1_len = m1.square().mean(dim=(2, 3)).sum(dim=1)
    vec_norm = m1_len

    index, norm_x = sample_index_from_bernouli(vec_norm)
    norm_x[norm_x == 0] = 1e-8
    m1 = m1 / norm_x.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    Ind = torch.zeros_like(m1)
    Ind[index] = 1
    m1 = m1.mul(Ind)
    m1up = m1[0:m1.shape[0] // 2]
    m1down = m1[m1.shape[0] // 2:]
    m1 = m1up + m1down

    return m1

def checkAbsmean(s, x):
    print(s, x.mean(), x.max(), x.min(), x.abs().mean())

def draw_maxmin(plt_list, cnt_plt, info):
    plt_path = os.path.join(config.args.workspace, 'plt_list', '{}'.format(config.epoch), '{}.png'.format(info))
    plt_path_log = os.path.join(config.args.workspace, 'plt_list', '{}'.format(config.epoch), '{}_log.png'.format(info))
    plt.figure(0)
    for idx, plst in enumerate(plt_list[info]):
        maxx, minn = plst[0].cpu(), plst[1].cpu()
        plt.scatter(maxx, 2 - idx / len(plt_list[info]), s=1, c='red')
        plt.scatter(minn, 0 + idx / len(plt_list[info]), s=1, c='blue')
    plt.title('{} {}'.format(config.epoch, info))
    plt.savefig(plt_path)

    plt.figure(1)
    for idx, plst in enumerate(plt_list[info]):
        maxx, minn = plst[0].cpu(), plst[1].cpu()
        plt.scatter(np.log10(maxx.abs() + 1e-10), 2 - idx / len(plt_list[info]), s=1, c='red')
        plt.scatter(-np.log10(minn.abs() + 1e-10), 0 + idx / len(plt_list[info]), s=1, c='blue')
    plt.savefig(plt_path_log)

    print("{} finish!".format(info))

cnt_plt = {'conv_weight': 0, 'conv_active': 0, 'linear_weight': 0, 'linear_active': 0}
list_plt = {'conv_weight': [], 'conv_active': [], 'linear_weight': [], 'linear_active': []}

if __name__ == '__main__':
    torch.set_printoptions(linewidth=160)
    m1 = torch.load('ckpt/twolayer_input.pt')
    m0 = m1[0:m1.shape[0] // 2] + m1[m1.shape[0] // 2:]
    m2 = twolayer_convsample_input(m1)
    print(m0.shape, m1.shape, m2.shape)

    checkAbsmean("m0", m0)
    checkAbsmean("m1", m1)
    checkAbsmean("m2", m2)
    diff_m = m0 - m2
    print(m0.abs().mean(), m2.abs().mean(), diff_m.abs().mean())

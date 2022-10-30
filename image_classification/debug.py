import os.path

import torch
from .quantize import config
from .utils import *
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import numpy as np
import pickle
from matplotlib.colors import LogNorm
import time

def get_error_grad(m):
    grad_dict = {}

    if hasattr(m, 'layer4'):
        layers = [m.layer1, m.layer2, m.layer3, m.layer4]
    else:
        layers = [m.layer1, m.layer2, m.layer3]

    for lid, layer in enumerate(layers):
        for bid, block in enumerate(layer):
            clayers = [block.conv1_in, block.conv2_in]
            if hasattr(block, 'conv3'):
                clayers.extend([block.conv3_in])

            for cid, clayer in enumerate(clayers):
                layer_name = 'conv_{}_{}_{}_error'.format(lid + 1, bid + 1, cid + 1)
                grad_dict[layer_name] = clayer.grad.detach().cpu()

    return grad_dict


def get_grad(m):
    grad_dict = {}

    if hasattr(m, 'layer4'):
        layers = [m.layer1, m.layer2, m.layer3, m.layer4]
    else:
        layers = [m.layer1, m.layer2, m.layer3]

    for lid, layer in enumerate(layers):
        for bid, block in enumerate(layer):
            clayers = [block.conv1, block.conv2, block.conv3] if hasattr(block, 'conv3') \
                else [block.conv1, block.conv2]

            for cid, clayer in enumerate(clayers):
                layer_name = 'conv_{}_{}_{}_weight'.format(lid + 1, bid + 1, cid + 1)
                grad_dict[layer_name] = clayer.weight.detach().cpu()
                layer_name = 'conv_{}_{}_{}_grad'.format(lid + 1, bid + 1, cid + 1)
                grad_dict[layer_name] = clayer.weight.grad.detach().cpu()

    return grad_dict


def get_batch_grad(model_and_loss, optimizer, val_loader, ckpt_name):
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    m.set_debug(True)
    data_iter = enumerate(val_loader)
    optimizer.zero_grad()
    cnt = 0
    for i, (input, target) in data_iter:
        loss, output = model_and_loss(input, target)
        loss.backward()
        torch.cuda.synchronize()
        cnt += 1

    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            param.grad /= cnt

    grad = get_grad(m)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        torch.save(grad, ckpt_name)
    return get_grad(m)


def get_grad_bias_std(model_and_loss, optimizer, val_loader, mean_grad, ckpt_name, num_epochs=1):
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    m.set_debug(True)
    data_iter = enumerate(val_loader)
    var_grad = None
    empirical_mean_grad = None
    cnt = 0
    for i, (input, target) in data_iter:
        for e in range(num_epochs):
            optimizer.zero_grad()
            loss, output = model_and_loss(input, target)
            loss.backward()
            torch.cuda.synchronize()

            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                grad_dict = get_grad(m)

                e_grad = dict_sqr(dict_minus(grad_dict, mean_grad))
                if var_grad is None:
                    var_grad = e_grad
                else:
                    var_grad = dict_add(var_grad, e_grad)

                if empirical_mean_grad is None:
                    empirical_mean_grad = grad_dict
                else:
                    empirical_mean_grad = dict_add(empirical_mean_grad, grad_dict)

            cnt += 1

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        std_grad = dict_sqrt(dict_mul(var_grad, 1.0 / cnt))
        bias_grad = dict_minus(dict_mul(empirical_mean_grad, 1.0 / cnt), mean_grad)
        torch.save(std_grad, ckpt_name)
        return bias_grad, std_grad


def get_grad_std_naive(model_and_loss, optimizer, val_loader, num_epochs=1):
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    m.set_debug(True)
    data_iter = enumerate(val_loader)
    var_grad = None
    cnt = 0
    for i, (input, target) in data_iter:
        config.quantize_gradient = False
        optimizer.zero_grad()
        loss, output = model_and_loss(input, target)
        loss.backward()
        torch.cuda.synchronize()
        mean_grad = dict_clone(get_grad(m))

        config.quantize_gradient = True
        for e in range(num_epochs):
            optimizer.zero_grad()
            loss, output = model_and_loss(input, target)
            loss.backward()
            torch.cuda.synchronize()
            grad_dict = get_grad(m)

            e_grad = dict_sqr(dict_minus(grad_dict, mean_grad))
            if var_grad is None:
                var_grad = e_grad
            else:
                var_grad = dict_add(var_grad, e_grad)

            cnt += 1

    std_grad = dict_sqrt(dict_mul(var_grad, 1.0 / cnt))
    return std_grad


def debug_bias(model_and_loss, optimizer, val_loader):
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    m.set_debug(True)
    data_iter = enumerate(val_loader)
    var_grad = None
    empirical_mean_grad = None
    cnt = 0
    for i, (input, target) in data_iter:
        break

    config.quantize_gradient = False
    optimizer.zero_grad()
    loss, output = model_and_loss(input, target)
    loss.backward()
    torch.cuda.synchronize()

    exact_grad = get_grad(m)
    empirical_mean_grad = None
    config.quantize_gradient = True
    for e in range(100):
        optimizer.zero_grad()
        loss, output = model_and_loss(input, target)
        loss.backward()
        torch.cuda.synchronize()

        cnt += 1
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            grad_dict = get_grad(m)

            if empirical_mean_grad is None:
                empirical_mean_grad = grad_dict
            else:
                empirical_mean_grad = dict_add(empirical_mean_grad, grad_dict)

            bias_grad = dict_minus(dict_mul(empirical_mean_grad, 1.0 / cnt), exact_grad)
            print(e, bias_grad['conv_1_1_1_grad'].abs().mean())


def get_gradient(model_and_loss, optimizer, input, target, prefix):
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    m.set_debug(True)

    loss, output = model_and_loss(input, target)

    optimizer.zero_grad()
    loss.backward()
    torch.cuda.synchronize()

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        grad_dict = get_grad(m)
        ckpt_name = "{}_weight.grad".format(prefix)
        torch.save(grad_dict, ckpt_name)

    grad_dict = get_error_grad(m)
    if not torch.distributed.is_initialized():
        rank = 0
    else:
        rank = torch.distributed.get_rank()

    ckpt_name = "{}_{}_error.grad".format(prefix, rank)
    torch.save(grad_dict, ckpt_name)


def dump(model_and_loss, optimizer, val_loader, checkpoint_dir):
    config.quantize_gradient = False
    print("Computing batch gradient...")
    grad = get_batch_grad(model_and_loss, optimizer, val_loader, checkpoint_dir + "/grad_mean.grad")

    # print("Computing gradient std...")
    # get_grad_std(model_and_loss, optimizer, val_loader, grad, checkpoint_dir + "/grad_std.grad")

    print("Computing quantization noise...")
    data_iter = enumerate(val_loader)
    for i, (input, target) in data_iter:
        break

    input = input[:128]
    target = target[:128]

    get_gradient(model_and_loss, optimizer, input, target, checkpoint_dir + "/exact")

    config.quantize_gradient = True
    for i in range(10):
        print(i)
        get_gradient(model_and_loss, optimizer, input, target, checkpoint_dir + "/sample_{}".format(i))

    # print("Computing quantized gradient std...")
    # get_grad_std(model_and_loss, optimizer, val_loader, grad, checkpoint_dir + "/grad_std_quan.grad")


def key(a):
    return [int(i) for i in a.split('_')[1:4]]


def fast_dump(model_and_loss, optimizer, val_loader, checkpoint_dir):
    # debug_bias(model_and_loss, optimizer, val_loader)
    # exit(0)

    config.quantize_gradient = False
    print("Computing batch gradient...")
    grad = get_batch_grad(model_and_loss, optimizer, val_loader, checkpoint_dir + "/grad_mean.grad")

    print("Computing gradient std...")
    g_outputs = get_grad_bias_std(model_and_loss, optimizer, val_loader, grad, checkpoint_dir + "/grad_std.grad",
                                  num_epochs=1)

    config.quantize_gradient = True
    q_outputs = get_grad_bias_std(model_and_loss, optimizer, val_loader, grad, checkpoint_dir + "/grad_std_quan.grad",
                                  num_epochs=1)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        bias_grad, std_grad = g_outputs
        bias_quan, std_quan = q_outputs
        weight_names = list(grad.keys())
        weight_names = [n.replace('_grad', '').replace('_weight', '') for n in weight_names]
        weight_names = list(set(weight_names))
        weight_names.sort(key=key)
        for k in weight_names:
            grad_mean = grad[k + '_grad']
            sg = std_grad[k + '_grad']
            bg = bias_grad[k + '_grad']
            sq = std_quan[k + '_grad']
            bq = bias_quan[k + '_grad']

            print('{}, batch grad mean={}, sample std={}, sample bias={}, overall std={}, overall bias={}'.format(
                k, grad_mean.abs().mean(), sg.mean(), bg.abs().mean(), sq.mean(), bq.abs().mean()))


def fast_dump_2(model_and_loss, optimizer, val_loader, checkpoint_dir):
    config.quantize_gradient = False
    print("Computing batch gradient...")
    grad = get_batch_grad(model_and_loss, optimizer, val_loader, checkpoint_dir + "/grad_mean.grad")

    print("Computing gradient std...")
    g_outputs = get_grad_bias_std(model_and_loss, optimizer, val_loader, grad, checkpoint_dir + "/grad_std.grad",
                                  num_epochs=1)

    # config.quantize_gradient = True
    # q_outputs = get_grad_bias_std(model_and_loss, optimizer, val_loader, grad, checkpoint_dir + "/grad_std_quan.grad", num_epochs=3)
    std_quan = get_grad_std_naive(model_and_loss, optimizer, val_loader, num_epochs=10)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        bias_grad, std_grad = g_outputs
        # bias_quan, std_quan = q_outputs
        weight_names = list(grad.keys())
        weight_names = [n.replace('_grad', '').replace('_weight', '') for n in weight_names]
        weight_names = list(set(weight_names))
        weight_names.sort(key=key)

        sample_var = 0.0
        quant_var = 0.0
        for k in weight_names:
            grad_mean = grad[k + '_grad']
            sg = std_grad[k + '_grad']
            sq = std_quan[k + '_grad']

            print('{}, batch grad norm={}, sample var={}, quantization var={}, overall var={}'.format(
                k, grad_mean.norm() ** 2, sg.norm() ** 2, sq.norm() ** 2, sq.norm() ** 2 + sg.norm() ** 2))

            sample_var += sg.norm() ** 2
            quant_var += sq.norm() ** 2

        print('SampleVar = {}, QuantVar = {}, OverallVar = {}'.format(
            sample_var, quant_var, sample_var + quant_var))


def leverage_score(args):
    import actnn.cpp_extension.backward_func as ext_backward_func
    from image_classification.quantize import quantize
    from image_classification.utils import twolayer_convsample_weight
    from image_classification.preconditioner import ScalarPreconditioner, TwoLayerWeightPreconditioner

    torch.set_printoptions(profile="full", linewidth=160)

    if config.args.twolayers_gradweight:
        load_dir = '20221025/{}/{}/grad_weight'.format(args.dataset, args.checkpoint_epoch)
    else:
        load_dir = '20221025/{}/{}/{}'.format(args.dataset, args.checkpoint_epoch,
                                              args.bwbits)
    save_file = open(os.path.join(load_dir, 'leverage.txt'), 'a')
    clear_file = open(os.path.join(load_dir, 'leverage.txt'), 'w')

    PT = torch.load(os.path.join(load_dir, 'tensor.pt'))
    grad_output, grad_input, grad_weight, saved, other_args = PT["grad output"], PT["grad input"], PT["grad weight"], \
                                                              PT["saved"], PT["other args"]
    inputt, weight, bias = saved
    stride, padding, dilation, groups = other_args

    _, full_grad_weight = ext_backward_func.cudnn_convolution_backward(
        inputt, grad_output, weight, padding, stride, dilation, groups,
        True, False, False,  # ?
        [False, True])
    grad_output_8_sum = None
    grad_output_4_sum = None
    grad_output_2_sum = None
    grad_weight_8_sum = None
    grad_weight_4_sum = None
    grad_weight_2_sum = None
    num_sample = 5
    for _ in trange(num_sample):
        grad_output_8 = quantize(grad_output, lambda x: ScalarPreconditioner(x, 8), stochastic=True)
        grad_output_4 = quantize(grad_output, lambda x: ScalarPreconditioner(x, 4), stochastic=True)
        grad_output_2 = quantize(grad_output, lambda x: TwoLayerWeightPreconditioner(x, 4), stochastic=True)

        input_sample, grad_output_weight_condi_sample = twolayer_convsample_weight(torch.cat([inputt, inputt], dim=0),
                                                                                   grad_output_2)
        vec_norm, index, norm = twolayer_convsample_weight(torch.cat([inputt, inputt], dim=0),
                                                           grad_output_2, debug=True)
        # grad_weight_2 = grad_output_2.t().mm(torch.cat([inputs, inputs], dim=0))
        _, grad_weight_2 = ext_backward_func.cudnn_convolution_backward(
            input_sample, grad_output_weight_condi_sample, weight, padding, stride, dilation, groups,
            True, False, False,  # ?
            [False, True])

        # brute force
        # input_2 = torch.cat([inputt, inputt], dim=0)
        # new_grad_weight = full_grad_weight.unsqueeze(0).repeat(input_2.shape[0], 1, 1, 1, 1)
        # for i in range(input_2.shape[0]):
        #     g2i, i2i = grad_output_2[i].unsqueeze(0), input_2[i].unsqueeze(0)
        #     _, grad_weight_2_i = ext_backward_func.cudnn_convolution_backward(
        #         i2i, g2i, weight, padding, stride, dilation, groups,
        #         True, False, False,  # ?
        #         [False, True])
        #     new_grad_weight[i] = grad_weight_2_i
        #
        # new_norm, new_abs_norm = new_grad_weight.sum(dim=(1, 2, 3, 4)), new_grad_weight.abs().sum(dim=(1, 2, 3, 4))
        # index = new_abs_norm.sort()[1]
        # # index = index[input_2.shape[0] // 2:]
        # index = index[100:]
        # grad_weight_2 = new_grad_weight[index].sum(dim=0)
        #
        # print(new_norm.sort()[0], new_abs_norm.sort()[0])
        # exit(0)

        _, grad_weight_8 = ext_backward_func.cudnn_convolution_backward(
            inputt, grad_output_8, weight, padding, stride, dilation, groups,
            True, False, False,  # ?
            [False, True])

        _, grad_weight_4 = ext_backward_func.cudnn_convolution_backward(
            inputt, grad_output_4, weight, padding, stride, dilation, groups,
            True, False, False,  # ?
            [False, True])

        try:
            grad_weight_2_sum += grad_weight_2 / num_sample
            grad_weight_4_sum += grad_weight_4 / num_sample
            grad_weight_8_sum += grad_weight_8 / num_sample
            grad_output_2_sum += grad_output_2 / num_sample
            grad_output_4_sum += grad_output_4 / num_sample
            grad_output_8_sum += grad_output_8 / num_sample
        except:
            grad_weight_2_sum = grad_weight_2 / num_sample
            grad_weight_4_sum = grad_weight_4 / num_sample
            grad_weight_8_sum = grad_weight_8 / num_sample
            grad_output_2_sum = grad_output_2 / num_sample
            grad_output_4_sum = grad_output_4 / num_sample
            grad_output_8_sum = grad_output_8 / num_sample

        del grad_weight_2, grad_weight_4, grad_weight_8

    time_tuple = time.localtime(time.time())
    print('Time {}/{:02d}/{:02d} {:02d}:{:02d}:{:02d}:'
          .format(time_tuple[0], time_tuple[1], time_tuple[2], time_tuple[3],
                  time_tuple[4], time_tuple[5]), file=clear_file)
    # print("fake gradient: ", grad_weight_fake.mean(), grad_weight_fake.abs().mean())
    print("full gradient: ", full_grad_weight.mean().detach().cpu().numpy(),
          full_grad_weight.abs().mean().detach().cpu().numpy(), file=save_file)
    print("grad_output:   ", grad_output.mean().detach().cpu().numpy(), grad_output.abs().mean().detach().cpu().numpy(),
          file=save_file)
    print("inputs:        ", inputt.mean().detach().cpu().numpy(), inputt.abs().mean().detach().cpu().numpy(),
          file=save_file)
    bias_weight_8 = grad_weight_8_sum - full_grad_weight
    bias_output_8 = grad_output_8_sum - grad_output
    print("bias_weight_8  ", bias_weight_8.mean().detach().cpu().numpy(),
          bias_weight_8.abs().mean().detach().cpu().numpy(), file=save_file)
    print("bias_output_8  ", bias_output_8.mean().detach().cpu().numpy(),
          bias_output_8.abs().mean().detach().cpu().numpy(), file=save_file)
    print("_________________________________________________________________________________")
    bias_weight_4 = grad_weight_4_sum - full_grad_weight
    bias_output_4 = grad_output_4_sum - grad_output
    print("bias_weight_4  ", bias_weight_4.mean().detach().cpu().numpy(),
          bias_weight_4.abs().mean().detach().cpu().numpy(), file=save_file)
    print("bias_output_4  ", bias_output_4.mean().detach().cpu().numpy(),
          bias_output_4.abs().mean().detach().cpu().numpy(), file=save_file)
    print("_________________________________________________________________________________")
    bias_weight_2 = grad_weight_2_sum - full_grad_weight
    bias_output_2 = grad_output_2_sum[:args.batch_size] + grad_output_2_sum[args.batch_size:] - grad_output
    print("bias_weight_2  ", bias_weight_2.mean().detach().cpu().numpy(),
          bias_weight_2.abs().mean().detach().cpu().numpy(), file=save_file)
    print("bias_output_2  ", bias_output_2.mean().detach().cpu().numpy(),
          bias_output_2.abs().mean().detach().cpu().numpy(), file=save_file)

    print("index: \n", index, 'norm\n', norm, norm.sum(), file=save_file)

    vec_norm = vec_norm.sort()[0].detach().cpu().numpy()[::-1]
    sum_norm = [vec_norm[:i].sum() / vec_norm.sum() for i in range(len(vec_norm))]

    plt.figure(1)
    plt.title("{}".format(args.checkpoint_epoch))
    plt.plot(np.arange(len(vec_norm)), vec_norm)
    plt.savefig(os.path.join(load_dir, 'leverage_score.png'))

    plt.figure(2)
    plt.title("{}".format(args.checkpoint_epoch))
    plt.plot(np.arange(len(vec_norm)), sum_norm)
    plt.savefig(os.path.join(load_dir, 'sum_norm.png'))

    print("calculate over!")


def plot_bin_hist(model_and_loss, optimizer, val_loader, args):
    config.grads = []
    config.acts = []
    data_iter = enumerate(val_loader)
    for i, (input, target) in data_iter:
        break

    input = input[:128]
    target = target[:128]

    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    loss, output = model_and_loss(input, target)

    optimizer.zero_grad()
    loss.backward()
    torch.cuda.synchronize()

    # fig, ax = plt.subplots(figsize=(5, 5))
    g = config.grads[10]
    # ax.hist(g.cpu().numpy().ravel(), bins=2**config.backward_num_bits-1)
    # ax.set_yscale('log')
    # fig.savefig('grad_output_hist.pdf')

    num_bins = 2 ** args.bwbits - 1

    if args.twolayers_gradweight:
        save_dir = "20221026/{}/{}/grad_weight".format(args.dataset, args.checkpoint_epoch)
    else:
        save_dir = "20221026/{}/{}/{}".format(args.dataset, args.checkpoint_epoch, args.bwbits)

    for i in [1, 2]:
        mxthres = g[i].abs().max().cpu().numpy() * 1.2
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.hist(g[i].cpu().numpy().ravel(), bins=num_bins, range=[-mxthres, mxthres])
        ax.set_yscale('log')
        ax.set_xlim([-mxthres, mxthres])
        ax.set_xticks([-mxthres, 0, mxthres])
        ax.set_xticklabels(['-{:.2e}'.format(mxthres), '$0$', '{:.2e}'.format(mxthres)])
        l, b, w, h = ax.get_position().bounds
        ax.set_position([l + 0.05 * w, b, 0.95 * w, h])
        fig.savefig(os.path.join(save_dir, '{}_hist.png'.format(i)), transparent=True)

    from image_classification.quantize import quantize

    def plot_each(preconditioner, Preconditioner, name, g):
        # input = g
        # prec = preconditioner(g, num_bits=config.backward_num_bits)

        prec = Preconditioner(g)
        g = prec.forward()

        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.hist(g.cpu().numpy().ravel(), bins=num_bins, range=[0, num_bins])
        ax.set_yscale('log')
        ax.set_ylim([1, 1e6])
        ax.set_xlim([0, num_bins])
        ax.set_xticks([0, num_bins])
        l, b, w, h = ax.get_position().bounds
        ax.set_position([l + 0.05 * w, b, 0.95 * w, h])
        fig.savefig(os.path.join(save_dir, '{}_hist.png'.format(name)), transparent=True)

        # prec.zero_point *= 0
        # bin_sizes = []
        # for i in range(128):
        #     bin_sizes.append(float(prec.inverse_transform(torch.eye(128)[:, i:i + 1].cuda()).sum()))
        # print(bin_sizes)
        # fig, ax = plt.subplots(figsize=(2.5, 2.5))
        # ax.hist(bin_sizes, bins=50, range=[0, 1e-5])
        # # ax.set_yscale('log')
        # ax.set_xlim([0, 1e-5])
        # ax.set_xticks([0, 1e-5])
        # ax.set_xticklabels(['$0$', '$10^{-5}$'])
        # l, b, w, h = ax.get_position().bounds
        # ax.set_position([l + 0.05 * w, b, 0.95 * w, h])
        # # ax.set_ylim([0, 128])
        # fig.savefig('{}_bin_size_hist.pdf'.format(name), transparent=True)
        #
        # gs = []
        # for i in range(10):
        #     grad = quantize(input, Preconditioner, stochastic=True)
        #     gs.append(grad.cpu().numpy())
        # var = np.stack(gs).var(0).sum()
        # print(var)

    from image_classification.preconditioner import ScalarPreconditionerAct

    plot_each(ScalarPreconditionerAct, lambda x: ScalarPreconditionerAct(x, config.bweight_num_bits), 'PTQ', g)
    # plot_each(DiagonalPreconditioner, lambda x: DiagonalPreconditioner(x, config.backward_num_bits), 'PSQ', g)
    # plot_each(BlockwiseHouseholderPreconditioner,
    #           lambda x: BlockwiseHouseholderPreconditioner(x, config.backward_num_bits), 'BHQ', g)

    # R = g.max(1)[0] - g.min(1)[0]
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.hist(R.cpu().numpy().ravel(), bins=2 ** config.backward_num_bits - 1)
    # fig.savefig('dyn_range_hist.pdf')

    # prec = BlockwiseHouseholderPreconditioner(g, num_bits=config.backward_num_bits)
    # gH = prec.T @ g
    # R = gH.max(1)[0] - gH.min(1)[0]
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.hist(R.cpu().numpy().ravel(), bins=2 ** config.backward_num_bits - 1)
    # fig.savefig('bH_dyn_range_hist.pdf')

    # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    #     num_grads = len(config.grads)
    #     fig, ax = plt.subplots(num_grads, figsize=(5, 5*num_grads))
    #     for i in range(num_grads):
    #         g = config.grads[i]
    #         ax[i].hist(g.cpu().numpy().ravel(), bins=2**config.backward_num_bits)
    #         ax[i].set_title(str(i))
    #         print(i, g.shape)
    #
    #     fig.savefig('grad_hist.pdf')

    # np.savez('errors.pkl', *config.grads)
    # np.savez('acts.pkl', *config.acts)


def plot_weight_hist(model_and_loss, optimizer, val_loader):
    config.grads = []
    config.acts = []
    data_iter = enumerate(val_loader)
    for i, (input, target) in data_iter:
        break

    input = input[:32]
    target = target[:32]

    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    m.set_debug(True)
    loss, output = model_and_loss(input, target)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        weights = []
        exact_weights = []
        acts = []
        names = []
        ins = []

        if hasattr(m, 'layer4'):
            layers = [m.layer1, m.layer2, m.layer3, m.layer4]
        else:
            layers = [m.layer1, m.layer2, m.layer3]

        print(m.layer1[0].conv1_out[0, 10])
        print(m.layer1[0].conv1_bn_out[0, 10])
        print(m.layer1[0].conv1_relu_out[0, 10])
        print(m.layer1[0].conv2_in[0, 10])
        print(m.layer1[0].bn1.running_mean, m.layer1[0].bn1.running_var)

        for lid, layer in enumerate(layers):
            for bid, block in enumerate(layer):
                clayers = [block.conv1, block.conv2, block.conv3] if hasattr(block, 'conv3') \
                    else [block.conv1, block.conv2]

                for cid, clayer in enumerate(clayers):
                    layer_name = 'conv_{}_{}_{}'.format(lid + 1, bid + 1, cid + 1)
                    names.append(layer_name)
                    exact_weights.append(clayer.weight.detach().cpu().numpy())
                    weights.append(clayer.qweight.detach().cpu().numpy())
                    acts.append(clayer.act.detach().cpu().numpy())
                    ins.append(clayer.iact.detach().cpu().numpy())

        num_weights = len(weights)
        fig, ax = plt.subplots(num_weights, figsize=(5, 5 * num_weights))
        for i in range(num_weights):
            weight = weights[i]
            ax[i].hist(weight.ravel(), bins=2 ** config.backward_num_bits)
            ax[i].set_title(names[i])
            print(i, weight.min(), weight.max())

        fig.savefig('weight_hist.pdf')
        np.savez('acts.pkl', *acts)
        np.savez('exact_weights.pkl', *exact_weights)
        np.savez('weights.pkl', *weights)
        np.savez('iacts.pkl', *ins)
        with open('layer_names.pkl', 'wb') as f:
            pickle.dump(names, f)

    config.quantize_weights = False
    loss, output = model_and_loss(input, target)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        acts = []
        ins = []

        if hasattr(m, 'layer4'):
            layers = [m.layer1, m.layer2, m.layer3, m.layer4]
        else:
            layers = [m.layer1, m.layer2, m.layer3]

        for lid, layer in enumerate(layers):
            for bid, block in enumerate(layer):
                clayers = [block.conv1, block.conv2, block.conv3] if hasattr(block, 'conv3') \
                    else [block.conv1, block.conv2]

                for cid, clayer in enumerate(clayers):
                    layer_name = 'conv_{}_{}_{}'.format(lid + 1, bid + 1, cid + 1)
                    acts.append(clayer.act.detach().cpu().numpy())
                    ins.append(clayer.iact.detach().cpu().numpy())

        np.savez('exact_acts.pkl', *acts)
        np.savez('exact_iacts.pkl', *ins)


def write_errors(model_and_loss, optimizer, val_loader):
    data_iter = enumerate(val_loader)
    for i, (input, target) in data_iter:
        break

    input = input[:128]
    target = target[:128]

    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    for iter in range(10):
        print(iter)
        config.grads = []
        loss, output = model_and_loss(input, target)
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            np.savez('errors_{}.pkl'.format(iter), *config.grads)


def variance_profile(model_and_loss, optimizer, val_loader, prefix='.', num_batches=10000):
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    # Get top 10 batches
    m.set_debug(True)
    m.set_name()
    weight_names = [layer.layer_name for layer in m.linear_layers]

    data_iter = enumerate(val_loader)
    inputs = []
    targets = []
    batch_grad = None
    quant_var = None

    def bp(input, target):
        optimizer.zero_grad()
        loss, output = model_and_loss(input, target)
        loss.backward()
        torch.cuda.synchronize()
        grad = {layer.layer_name: layer.weight.grad.detach().cpu() for layer in m.linear_layers}
        return grad

    cnt = 0
    for i, (input, target) in tqdm(data_iter):
        cnt += 1

        inputs.append(input.clone())
        targets.append(target.clone())

        # Deterministic
        config.quantize_gradient = False
        mean_grad = bp(input, target)
        batch_grad = dict_add(batch_grad, mean_grad)

        if cnt == num_batches:
            break

    num_batches = cnt
    batch_grad = dict_mul(batch_grad, 1.0 / num_batches)

    def get_variance():
        total_var = None
        for i, input, target in tqdm(zip(range(num_batches), inputs, targets)):
            grad = bp(input, target)
            total_var = dict_add(total_var, dict_sqr(dict_minus(grad, batch_grad)))

        grads = [total_var[k].sum() / num_batches for k in weight_names]
        print(grads)
        return grads

    config.quantize_gradient = True
    grads = [get_variance()]
    for layer in tqdm(m.linear_layers):
        layer.exact = True
        grads.append(get_variance())

    grads = np.array(grads)

    for i in range(grads.shape[0] - 1):
        grads[i] -= grads[i + 1]

    np.save(prefix + '/error_profile.npy', grads)
    with open(prefix + '/layer_names.pkl', 'wb') as f:
        pickle.dump(weight_names, f)

    grads = np.maximum(grads, 0)
    # grads = np.minimum(grads, 1)
    for i in range(grads.shape[0]):
        for j in range(grads.shape[1]):
            if j > i:
                grads[i, j] = 0

    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(grads, cmap='Blues', norm=LogNorm(vmin=1e-7, vmax=1e-3))
    ax.set_xticks(np.arange(len(weight_names)))
    ax.set_yticks(np.arange(len(weight_names)))
    ax.set_xticklabels(weight_names)
    ax.set_yticklabels(weight_names)
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")
    cbar = ax.figure.colorbar(im, ax=ax)

    for i in range(grads.shape[0]):
        for j in range(grads.shape[1]):
            text = ax.text(j, i, int(grads[i, j] * 10),
                           ha="center", va="center")

    fig.savefig('variance_profile.pdf')


#
# def get_var(model_and_loss, optimizer, val_loader, num_batches=10000):
#     if hasattr(model_and_loss.model, 'module'):
#         m = model_and_loss.model.module
#     else:
#         m = model_and_loss.model
#
#     # Get top 10 batches
#     m.set_debug(True)
#     m.set_name()
#     weight_names = [layer.layer_name for layer in m.linear_layers]
#
#     data_iter = enumerate(val_loader)
#     inputs = []
#     targets = []
#     batch_grad = None
#     quant_var = None
#
#     def bp(input, target):
#         optimizer.zero_grad()
#         loss, output = model_and_loss(input, target)
#         loss.backward()
#         torch.cuda.synchronize()
#         grad = {layer.layer_name : layer.weight.grad.detach().cpu() for layer in m.linear_layers}
#         return grad
#
#     cnt = 0
#     for i, (input, target) in tqdm(data_iter):
#         cnt += 1
#
#         inputs.append(input.clone())
#         targets.append(target.clone())
#
#         # Deterministic
#         config.quantize_gradient = False
#         mean_grad = bp(input, target)
#         batch_grad = dict_add(batch_grad, mean_grad)
#
#         if cnt == num_batches:
#             break
#
#     num_batches = cnt
#     batch_grad = dict_mul(batch_grad, 1.0 / num_batches)
#
#     def get_variance():
#         total_var = None
#         for i, input, target in tqdm(zip(range(num_batches), inputs, targets)):
#             grad = bp(input, target)
#             total_var = dict_add(total_var, dict_sqr(dict_minus(grad, batch_grad)))
#
#         grads = [total_var[k].sum() / num_batches for k in weight_names]
#         return grads
#
#     config.quantize_gradient = True
#     q_grads = get_variance()
#     config.quantize_gradient = False
#     s_grads = get_variance()
#
#     all_qg = 0
#     all_sg = 0
#     for i, k in enumerate(weight_names):
#         qg = q_grads[i].sum()
#         sg = s_grads[i].sum()
#         all_qg += qg
#         all_sg += sg
#         print('{}, overall var = {}, quant var = {}, sample var = {}'.format(k, qg, qg-sg, sg))
#
#     print('Overall Var = {}, Quant Var = {}, Sample Var = {}'.format(all_qg, all_qg - all_sg, all_sg))

def get_var(model_and_loss, optimizer, val_loader, args=None):
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    if args.twolayers_gradweight:
        fopen = open("20221025/{}/{}/grad_weight/var.txt".format(args.dataset, args.checkpoint_epoch), 'a')
    else:
        fopen = open("20221025/{}/{}/{}/var.txt".format(args.dataset, args.checkpoint_epoch, args.bwbits), 'a')
    # Get top 10 batches
    m.set_debug(True)
    m.set_name()
    weight_names = [layer.layer_name for layer in m.linear_layers]

    data_iter = enumerate(val_loader)

    def bp(input, target):
        optimizer.zero_grad()
        loss, output = model_and_loss(input, target)
        loss.backward()
        torch.cuda.synchronize()
        grad = {layer.layer_name: layer.weight.grad.detach().cpu() for layer in m.linear_layers}
        return grad

    cnt = 0
    num_samples = 1
    all_var = None
    all_bias = None
    for (input, target) in tqdm(val_loader):
        cnt += 1

        config.quantize_gradient = False
        full_grad = bp(input, target)

        config.quantize_gradient = True
        cur_var = None
        cur_bias = None
        for _ in range(num_samples):
            grad = bp(input, target)
            cur_var = dict_add(cur_var, dict_sqr(dict_minus(grad, full_grad)))
            cur_bias = dict_add(cur_bias, dict_minus(grad, full_grad))

        var = [cur_var[k].sum() / num_samples for k in weight_names]
        bias = [cur_bias[k].sum() / num_samples for k in weight_names]

        try:
            all_var = all_var + var
            all_bias = all_bias + bias
        except:
            all_var = var
            all_bias = bias
    if args.local_rank == 0:
        print("cnt is {}".format(cnt), file=fopen)
    all_var, all_bias = [x / cnt for x in all_var], [x / cnt for x in all_bias]

    all_qg, all_qb = 0, 0
    for i, k in enumerate(weight_names):
        qg = all_var[i]
        qb = all_bias[i]
        all_qg += qg
        all_qb += qb
        if args.local_rank == 0:
            print('{}, quant var = {:.3e}, quant bias = {:.3e}'.format(k, qg, qb), file=fopen)
    if args.local_rank == 0:
        print('Overall Quant Var = {:.3e}, Quant bias = {:.3e}'
              .format(all_qg, all_qb), file=fopen)
    # print("hello world", file=open("20221025/{}/{}/var.txt".format(args.checkpoint_epoch, args.bwbits), 'a'))

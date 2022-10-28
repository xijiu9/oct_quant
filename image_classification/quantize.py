from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd.function import InplaceFunction, Function

import matplotlib.pyplot as plt
import actnn.cpp_extension.backward_func as ext_backward_func
import numpy as np
from tqdm import trange
from image_classification.utils import config

try:
    from image_classification.preconditioner import ScalarPreconditioner, ScalarPreconditionerAct, \
        TwoLayerWeightPreconditioner, lsq_per_tensor
    from image_classification.utils import twolayer_linearsample_weight, twolayer_convsample_weight, \
        twolayer_linearsample_input, twolayer_convsample_input, checkNAN, checkAbsmean, cnt_plt, list_plt, draw_maxmin
except:
    from utils import twolayer_linearsample_weight, twolayer_convsample_weight, sample_index_from_bernouli, \
        twolayer_linearsample_input, twolayer_convsample_input, checkNAN, checkAbsmean
    from preconditioner import ScalarPreconditioner, DiagonalPreconditioner, \
        BlockwiseHouseholderPreconditioner, ScalarPreconditionerAct, TwoLayerWeightPreconditioner, lsq_per_tensor

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_GRAD = (0, -1)


class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, Preconditioner, stochastic=False, inplace=False, debug=False, info=''):

        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        #     print('---')
        #     print(input.view(-1)[:10], input.min(), input.max())
        with torch.no_grad():
            preconditioner = Preconditioner(output)
            output = preconditioner.forward()

            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            # quantize
            output.clamp_(0.0, preconditioner.num_bins).round_()

            inverse_output = preconditioner.inverse(output)

            # threshold = {'conv_weight': 1100, 'conv_active': 1100, 'linear_weight': 20, 'linear_active': 20}
            # if info != '' and cnt_plt[info] < threshold[info]:
            #     cnt_plt[info] += 1
            #     list_plt[info].append([inverse_output.max(), inverse_output.min()])
            #
            # if info != '' and cnt_plt[info] == threshold[info]:
            #     cnt_plt[info] += 1
            #     draw_maxmin(list_plt, cnt_plt, info)

        # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        #     print(output.view(-1)[:10])
        if not debug:
            return inverse_output
        return output, inverse_output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None


def quantize(x, Preconditioner, stochastic=False, inplace=False, debug=False, info=''):
    return UniformQuantize().apply(x, Preconditioner, stochastic, inplace, debug, info)


class conv2d_act(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        ctx.saved = input, weight, bias
        ctx.other_args = stride, padding, dilation, groups
        ctx.inplace = False
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        if config.grads is not None:
            config.grads.append(grad_output.detach())
        # checkNAN(grad_output, 'grad_out')
        # if torch.isnan(grad_output[0, 0, 0, 0]):
        #     print("Linear De")
        # torch.save(grad_output, 'image_classification/ckpt/grad_output_conv_180.pt')
        # print('*'*100)
        # checkAbsmean("conv start", grad_output)
        grad_output_weight_condi = quantize(grad_output, config.weight_gradient_preconditioner(), stochastic=True,
                                            info='conv_weight')

        grad_output_active_condi = quantize(grad_output, config.activation_gradient_preconditioner(),
                                            stochastic=True, info='conv_active')

        input, weight, bias = ctx.saved
        stride, padding, dilation, groups = ctx.other_args

        # torch.save(
        #     {"input": input, "weight": weight, "bias": bias, "stride": stride, "padding": padding, "dilation": dilation
        #         , "groups": groups}, 'image_classification/ckpt/inputs_conv_180.pt')
        if config.twolayer_weight:
            input_sample, grad_output_weight_condi_sample = twolayer_convsample_weight(
                torch.cat([input, input], dim=0), grad_output_weight_condi)

            _, grad_weight = ext_backward_func.cudnn_convolution_backward(
                input_sample, grad_output_weight_condi_sample, weight, padding, stride, dilation, groups,
                True, False, False,  # ?
                [False, True])
            # torch.save(grad_weight, 'image_classification/ckpt/grad_weight_conv_180.pt')
            # exit(0)
        else:
            _, grad_weight = ext_backward_func.cudnn_convolution_backward(
                input, grad_output_weight_condi, weight, padding, stride, dilation, groups,
                True, False, False,  # ?
                [False, True])

        if config.twolayer_inputt:
            # checkAbsmean("grad_active_condi", grad_output_active_condi)
            grad_output_active_sample = twolayer_convsample_input(grad_output_active_condi, config)
            # checkAbsmean("grad_active_sample", grad_output_active_sample)
            grad_input, _ = ext_backward_func.cudnn_convolution_backward(
                input, grad_output_active_sample, weight, padding, stride, dilation, groups,
                True, False, False,  # ?
                [True, False])

        else:
            # checkAbsmean("grad_active_condi", grad_output_active_condi)
            grad_input, _ = ext_backward_func.cudnn_convolution_backward(
                input, grad_output_active_condi, weight, padding, stride, dilation, groups,
                True, False, False,  # ?
                [True, False])
            # checkAbsmean("grad_input", grad_input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum([0, 2])
        else:
            grad_bias = None

        # if config.args.twolayers_gradweight:
        #     save_dir = '20221025/{}/{}/grad_weight/tensor.pt'.format(config.args.dataset, config.args.checkpoint_epoch)
        # else:
        #     save_dir = '20221025/{}/{}/{}/tensor.pt'.format(config.args.dataset, config.args.checkpoint_epoch, config.args.bwbits)
        # torch.save({"grad output": grad_output,
        #             "grad input": grad_input,
        #             "grad weight": grad_weight,
        #             "saved": ctx.saved,
        #             "other args": ctx.other_args},
        #            save_dir)
        # print("already saved")
        # exit(0)
        # checkNAN(grad_input, 'grad_input')
        return grad_input, grad_weight, grad_bias, None, None, None, None


class linear_act(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.saved = input, weight, bias
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        # print("linear", grad_output.mean(), grad_output.max(), grad_output.min(), grad_output.abs().mean())
        # torch.set_printoptions(profile="full")
        # print(grad_output[:, :])
        # exit(0)
        torch.set_printoptions(profile="full", linewidth=160)
        # torch.save(grad_output, 'image_classification/ckpt/grad_output_linear_0.pt')
        grad_output_weight_conditioner = quantize(grad_output, config.weight_gradient_preconditioner(), stochastic=True,
                                                  info='linear_weight')

        grad_output_active_conditioner = quantize(grad_output, config.activation_gradient_preconditioner(),
                                                  stochastic=True, info='linear_active')

        # exit(0)
        input, weight, bias = ctx.saved
        # torch.save(input, 'image_classification/ckpt/input_linear_0.pt')
        C_in = input.shape[-1]
        C_out = grad_output.shape[-1]
        # rank = len(grad_output.shape)

        grad_output_flatten = grad_output.view(-1, C_out)
        grad_output_flatten_weight = grad_output_weight_conditioner.view(-1, C_out)
        grad_output_flatten_active = grad_output_active_conditioner.view(-1, C_out)
        input_flatten = input.view(-1, C_in)

        # print(torch.linalg.norm(grad_output_flatten_weight, dim=1), len(torch.linalg.norm(grad_output_flatten_weight, dim=1)))
        # print(grad_output_flatten_weight[:, :5], grad_output_flatten_weight.shape)
        if config.twolayer_weight:
            m1, m2 = twolayer_linearsample_weight(grad_output_flatten_weight, input_flatten)
            grad_weight = m1.t().mm(m2)
        else:
            grad_weight = grad_output_flatten_weight.t().mm(input_flatten)

        if config.twolayer_inputt:
            I = torch.eye(input.shape[0], device="cuda")
            grad_input = twolayer_linearsample_input(grad_output_flatten_active, I)
            #
            # N = torch.isnan(grad_input)
            # cN = torch.count_nonzero(N)
            # if cN != 0:
            #     print("Linear Input")
            #     print(cN)
            #     print(grad_input.shape)
            #     print(grad_input)
            #     exit(0)

            grad_input = grad_input.mm(weight)
        else:
            grad_input = grad_output_flatten_active.mm(weight)
        # print(grad_weight.shape, weight.shape)
        if bias is not None:
            grad_bias = grad_output_flatten.sum(0)
        else:
            grad_bias = None
        #
        # N = torch.isnan(grad_input)
        # cN = torch.count_nonzero(N)
        # if cN != 0:
        #     print("Linear Linear")
        #     print(cN)
        #     print(grad_input.shape)
        #     print(grad_input)
        #     exit(0)
        return grad_input, grad_weight, grad_bias


class LSQPerTensor(nn.Module):
    def __init__(self, bits, symm=True, inputtype=''):
        super(LSQPerTensor, self).__init__()
        self.bits = bits
        self.symm = symm
        self.step_size = Parameter(torch.tensor(1.0), requires_grad=config.lsqforward)
        self.initialized = False
        self.inputtype = inputtype

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                num_bins = 2 ** self.bits - 1
                step_size = 2 * x.abs().mean() / np.sqrt(num_bins)
                self.step_size.copy_(step_size)  # LSQ type

                self.initialized = True
        if x.min() < 0 and x.max() > 0 and not self.symm:
            print("!!!!!!!!!!!!!")
            print(x)
            exit(0)
        return lsq_per_tensor().apply(x, self.step_size, config, self.bits, self.symm, self.inputtype)

    def quantize_MSE(self, input, scale, bits, symm):
        num_bins = 2 ** bits - 1
        bias = -num_bins / 2 if symm else 0

        # Forward
        eps = 1e-7
        scale = scale + eps
        transformed = input / scale - bias
        vbar = torch.clamp(transformed, 0.0, num_bins).round()
        quantized = (vbar + bias) * scale

        MSE = (quantized - input).square().sum()
        return MSE


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, inplace=False, stochastic=False):
        super(QuantMeasure, self).__init__()
        self.stochastic = stochastic
        self.inplace = inplace

    def forward(self, input):
        q_input = quantize(input, config.activation_preconditioner(),
                           stochastic=self.stochastic, inplace=self.inplace)
        return q_input


class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, symm=True):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.quantize_input = QuantMeasure()
        self.symm = symm
        if config.lsqforward:
            self.lsqweight = LSQPerTensor(config.weight_num_bits, inputtype="weight")
            self.lsqactive = LSQPerTensor(config.activation_num_bits, symm=symm, inputtype="activation")

    def forward(self, input):
        # randflag = torch.rand(1)
        # print(randflag)
        # torch.save(input, 'image_classification/debug_tensor/input.pt')
        if config.acts is not None:
            config.acts.append(input.detach().cpu().numpy())

        if config.quantize_activation:
            if config.lsqforward:
                qinput = self.lsqactive(input)
            else:
                qinput = self.quantize_input(input)
                # torch.save(qinput, 'image_classification/debug_tensor/qinput.pt')
        else:
            qinput = input
        # torch.save(self.weight, 'image_classification/debug_tensor/weight.pt')
        if config.quantize_weights:  # TODO weight quantization scheme...
            if config.lsqforward:
                qweight = self.lsqweight(self.weight)
            else:
                qweight = quantize(self.weight, config.weight_preconditioner())
                # torch.save(qweight, 'image_classification/debug_tensor/qweight.pt')
                # if randflag < 0.1:
                #     exit(0)
            qbias = self.bias
        else:
            qweight = self.weight
            qbias = self.bias

        if hasattr(self, 'exact') or not config.quantize_gradient:
            output = F.conv2d(qinput, qweight, qbias, self.stride,
                              self.padding, self.dilation, self.groups)
        else:
            output = conv2d_act.apply(qinput, qweight, qbias, self.stride,
                                      self.padding, self.dilation, self.groups)

        self.act = output

        return output


class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, symm=True):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.quantize_input = QuantMeasure()
        if config.lsqforward:
            self.lsqweight = LSQPerTensor(config.weight_num_bits, inputtype="weight")
            self.lsqactive = LSQPerTensor(config.activation_num_bits, symm=symm, inputtype="activation")

    def forward(self, input):
        print("QLinear should not be used!")
        if config.quantize_activation:
            if config.lsqforward:
                qinput = self.lsqactive(input)
            else:
                qinput = self.quantize_input(input)
        else:
            qinput = input

        if config.quantize_weights:  # TODO weight quantization scheme...
            if config.lsqforward:
                qweight = self.lsqweight(self.weight)
            else:
                qweight = quantize(self.weight, config.weight_preconditioner())
            qbias = self.bias
        else:
            qweight = self.weight
            qbias = self.bias

        if hasattr(self, 'exact') or not config.quantize_gradient:
            output = F.linear(qinput, qweight, qbias)
        else:
            output = linear_act.apply(qinput, qweight, qbias)

        return output


class QBatchNorm2D(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(QBatchNorm2D, self).__init__(num_features)
        self.quantize_input = QuantMeasure()

    def forward(self, input):  # TODO: weight is not quantized
        self._check_input_dim(input)
        # if config.quantize_activation:
        #     qinput = self.quantize_input(input)
        # else:
        qinput = input

        qweight = self.weight
        qbias = self.bias

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        return F.batch_norm(
            input, self.running_mean, self.running_var, qweight, qbias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)


# if __name__ == '__main__':
#
#     type = "100"
#     torch.set_printoptions(profile="full", linewidth=160)
#     grad_output = torch.load("ckpt/grad_output_linear_{}.pt".format(type))
#     inputs = torch.load("ckpt/inputs_linear_{}.pt".format(type))
#
#     full_grad_weight = grad_output.t().mm(inputs)
#     grad_output_8_sum = None
#     grad_output_2_sum = None
#     grad_weight_8_sum = None
#     grad_weight_2_sum = None
#     num_sample = 10000
#     for i in trange(num_sample):
#         grad_output_8 = quantize(grad_output, lambda x: ScalarPreconditioner(x, 8), stochastic=True)
#         grad_output_2 = quantize(grad_output, lambda x: TwoLayerWeightPreconditioner(x, 4), stochastic=True)
#
#         # grad_weight_2 = grad_output_2.t().mm(torch.cat([inputs, inputs], dim=0))
#         m1, m2 = twolayer_linearsample(grad_output_2, inputs, epoch=0)
#         grad_weight_2 = m1.t().mm(m2)
#         grad_weight_8 = grad_output_8.t().mm(inputs)
#         try:
#             grad_weight_2_sum += grad_weight_2 / num_sample
#             grad_weight_8_sum += grad_weight_8 / num_sample
#             grad_output_2_sum += grad_output_2 / num_sample
#             grad_output_8_sum += grad_output_8 / num_sample
#         except:
#             grad_weight_2_sum = grad_weight_2 / num_sample
#             grad_weight_8_sum = grad_weight_8 / num_sample
#             grad_output_2_sum = grad_output_2 / num_sample
#             grad_output_8_sum = grad_output_8 / num_sample
#
#     print("full gradient: ", full_grad_weight.mean(), full_grad_weight.abs().mean())
#     print("grad_output:   ", grad_output.mean(), grad_output.abs().mean())
#     print("inputs:        ", inputs.mean(), inputs.abs().mean())
#     bias_weight_8 = grad_weight_8_sum - full_grad_weight
#     bias_output_8 = grad_output_8_sum - grad_output
#     print("bias_weight_8  ", bias_weight_8.mean(), bias_weight_8.abs().mean())
#     print("bias_output_8  ", bias_output_8.mean(), bias_output_8.abs().mean())
#     print("_________________________________________________________________________________")
#     bias_weight_2 = grad_weight_2_sum - full_grad_weight
#     bias_output_2 = grad_output_2_sum[:128] + grad_output_2_sum[128:] - grad_output
#     print("bias_weight_2  ", bias_weight_2.mean(), bias_weight_2.abs().mean())
#     print("bias_output_2  ", bias_output_2.mean(), bias_output_2.abs().mean())

if __name__ == '__main__':

    the_type = "180"
    torch.set_printoptions(profile="full", linewidth=160)
    
    if config.args.twolayers_gradweight:
        save_dir = '20221025/{}/{}/grad_weight/tensor.pt'.format(config.args.dataset, config.args.checkpoint_epoch)
    else:
        save_dir = '20221025/{}/{}/{}/tensor.pt'.format(config.args.dataset, config.args.checkpoint_epoch, config.args.bwbits)

    PT = torch.load("")

    grad_output, grad_input, grad_weight, saved, other_args = PT["grad output"], PT["grad input"], PT["grad weight"], \
                                                              PT["saved"]. PT["other args"]

    input, weight, bias = ctx.saved
    stride, padding, dilation, groups = ctx.other_args

    _, full_grad_weight = ext_backward_func.cudnn_convolution_backward(
        inputt, grad_output, weight, padding, stride, dilation, groups,
        True, False, False,  # ?
        [False, True])
    grad_output_8_sum = None
    grad_output_2_sum = None
    grad_weight_8_sum = None
    grad_weight_2_sum = None
    num_sample = 1
    for i in trange(num_sample):
        grad_output_8 = quantize(grad_output, lambda x: ScalarPreconditioner(x, 8), stochastic=True)
        grad_output_2 = quantize(grad_output, lambda x: TwoLayerWeightPreconditioner(x, 4), stochastic=True)

        input_sample, grad_output_weight_condi_sample = twolayer_convsample(torch.cat([inputt, inputt], dim=0),
                                                                            grad_output_2, epoch=0)
        input_sample_debug, grad_output_weight_condi_sample_debug = twolayer_convsample_debug(
            torch.cat([inputt, inputt], dim=0),
            grad_output_2, epoch=int(the_type))
        # grad_weight_2 = grad_output_2.t().mm(torch.cat([inputs, inputs], dim=0))
        _, grad_weight_2 = ext_backward_func.cudnn_convolution_backward(
            input_sample, grad_output_weight_condi_sample, weight, padding, stride, dilation, groups,
            True, False, False,  # ?
            [False, True])
        _, grad_weight_2_d = ext_backward_func.cudnn_convolution_backward(
            input_sample_debug, grad_output_weight_condi_sample_debug, weight, padding, stride, dilation, groups,
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
        try:
            grad_weight_2_sum += grad_weight_2 / num_sample
            grad_weight_8_sum += grad_weight_8 / num_sample
            grad_output_2_sum += grad_output_2 / num_sample
            grad_output_8_sum += grad_output_8 / num_sample
        except:
            grad_weight_2_sum = grad_weight_2 / num_sample
            grad_weight_8_sum = grad_weight_8 / num_sample
            grad_output_2_sum = grad_output_2 / num_sample
            grad_output_8_sum = grad_output_8 / num_sample

    # print("fake gradient: ", grad_weight_fake.mean(), grad_weight_fake.abs().mean())
    print("full gradient: ", full_grad_weight.mean(), full_grad_weight.abs().mean())
    print("grad_output:   ", grad_output.mean(), grad_output.abs().mean())
    print("inputs:        ", inputt.mean(), inputt.abs().mean())
    bias_weight_8 = grad_weight_8_sum - full_grad_weight
    bias_output_8 = grad_output_8_sum - grad_output
    print("bias_weight_8  ", bias_weight_8.mean(), bias_weight_8.abs().mean())
    print("bias_output_8  ", bias_output_8.mean(), bias_output_8.abs().mean())
    print("_________________________________________________________________________________")
    bias_weight_2 = grad_weight_2_sum - full_grad_weight
    bias_output_2 = grad_output_2_sum[:128] + grad_output_2_sum[128:] - grad_output
    print("bias_weight_2  ", bias_weight_2.mean(), bias_weight_2.abs().mean())
    print("bias_output_2  ", bias_output_2.mean(), bias_output_2.abs().mean())
    bias_weight_2_d = grad_weight_2_d - full_grad_weight
    print("bias_weight_2d ", bias_weight_2_d.mean(), bias_weight_2_d.abs().mean())

# if __name__ == '__main__':
#     print(1)
#     input = torch.load("debug_tensor/input.pt")
#     qinput = torch.load("debug_tensor/qinput.pt")
#     weight = torch.load("debug_tensor/weight.pt")
#     qweight = torch.load("debug_tensor/qweight.pt")
#
#     # print(input.shape, qinput.shape, weight.shape, qweight.shape)
#     # print(input.min(), input.max())
#     # print(qinput.min(), qinput.max())
#     print(qweight.min(), qweight.max())
#     print(weight.min(), weight.max())
#
#
#     # print(qweight)
#     def toint(x):
#         minn = x.min()
#         maxx = x.max()
#         print(minn, maxx)
#         print(x - minn)
#         scale = 14 / (maxx - minn)
#         return (x - minn) * scale
#
#
#     # print(toint(qinput))
#     print(toint(qweight))

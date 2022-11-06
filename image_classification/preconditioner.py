import torch
import math
import time
import numpy as np


def checkNAN(x, s=''):
    N = torch.isnan(x)
    cN = torch.count_nonzero(N)
    if cN != 0:
        print("NAN!!!{}".format(s))
        print(cN)
        print(x.shape)
        print(x)


def householder(src, tar):
    N = src.shape[0]
    v = tar - src
    v = v / v.norm()
    return torch.eye(N) - 2 * v.view(N, 1) @ v.view(1, N)


Qs = [[], [torch.ones(1), 1.0]]
Qqs = [torch.tensor(1.0), torch.ones(1)]
Qmax = [1.0, 1.0]


def init(max_bs):
    for i in range(2, max_bs + 1):
        e1 = torch.zeros(i)
        e1[0] = 1
        ones = torch.ones(i) / math.sqrt(i)
        H = householder(e1, ones)
        Hmax = H.abs().max()
        Qs.append([H, Hmax])
        Qqs.append(H)
        Qmax.append(Hmax)


class Preconditioner:
    def __init__(self, x, num_bits, left=True):
        self.left = left
        self.x_shape = x.shape
        self.num_bins = 2 ** num_bits - 1
        self.num_bits = num_bits

        self.x = self.flatten(x)
        self.Tx = self.transform(self.x)

    def flatten(self, x):
        self.x_shape2 = x.shape
        self.x_shape_double = torch.cat([x, x], dim=0).shape
        return x.view(x.shape[0], -1)

    def deflatten(self, Tx):
        try:
            x = Tx.view(*self.x_shape2)
        except:
            x = Tx.view(*self.x_shape_double)
        return x

    def forward(self):
        return self.Tx

    def inverse(self, Tx):
        x = self.inverse_transform(Tx)
        return self.deflatten(x)


class ScalarPreconditioner(Preconditioner):
    # y = (x - z) * scale
    # x = y / scale + z
    def __init__(self, x, num_bits, left=True):
        super(ScalarPreconditioner, self).__init__(x, num_bits, left)

    def transform(self, x):
        with torch.no_grad():
            mn = min(x.min() - 1e-8, 0)
            mx = max(x.max() + 1e-8, 0)

        self.zero_point = mn
        self.scale = self.num_bins / (mx - mn)

        qzero = -self.zero_point * self.scale
        iqzero = torch.floor(qzero)
        mx = (iqzero - self.num_bins) * mn / iqzero
        self.scale = self.num_bins / (mx - mn)

        return (x - self.zero_point) * self.scale

    def inverse_transform(self, x):
        return x / self.scale + self.zero_point


class ScalarPreconditionerAct(Preconditioner):
    # y = (x - z) * scale
    # x = y / scale + z
    def __init__(self, x, num_bits, left=True):
        super(ScalarPreconditionerAct, self).__init__(x, num_bits, left)

    def transform(self, x):
        with torch.no_grad():
            mn = x.min() - 1e-8
            mx = x.max() + 1e-8

        self.zero_point = mn
        self.scale = self.num_bins / (mx - mn)

        return (x - self.zero_point) * self.scale

    def inverse_transform(self, x):
        return x / self.scale + self.zero_point


class TwoLayerWeightPreconditioner(Preconditioner):
    # Y = D (Y - z 1^\top)
    # X = D^-1 Y + z 1^\top
    def __init__(self, x, num_bits, left=True):
        super(TwoLayerWeightPreconditioner, self).__init__(x, num_bits, left)

    def transform(self, x, debug=False):
        with torch.no_grad():
            mn = min(x.min() - 1e-8, 0)
            mx = max(x.max() + 1e-8, 0)

        if debug:
            print("mn is {}, mx is {}".format(mn, mx))
            print(x.view(-1).topk(3), (-x).view(-1).topk(3))

        self.zero_point1 = mn
        self.scale1 = self.num_bins / (mx - mn)

        qzero = -self.zero_point1 * self.scale1
        iqzero = torch.floor(qzero)

        if debug:
            print(qzero, iqzero, "qzero, iqzero")

        if iqzero <= 0:
            # torch.save(x, 'image_classification/ckpt/precon1x.pt')
            print("save for 1")
            print("part 1 break, x is {}, iqzero is {} \n".format(x, iqzero))

        if iqzero > 0:
            mx = (iqzero - self.num_bins) * mn / iqzero
        elif iqzero == 0:
            self.zero_point1, mn = 0, 0

        self.scale1 = self.num_bins / (mx - mn)

        if debug:
            print(mx, self.scale1, "mx, scale1")

        if torch.isnan(self.scale1):
            torch.save(x, 'image_classification/ckpt/precon1x.pt')
            print("save for 1")

        first_transform = (x - self.zero_point1) * self.scale1
        first_transform.clamp_(0.0, self.num_bins).round_()
        first_quantize = first_transform / self.scale1 + self.zero_point1

        residual = x - first_quantize

        with torch.no_grad():
            mn = min(residual.min() - 1e-8, 0)
            mx = max(residual.max() + 1e-8, 0)

        self.zero_point2 = mn
        self.scale2 = self.num_bins / (mx - mn)

        if debug:
            print("mn is {}, mx is {}".format(mn, mx))
            print(residual.view(-1).topk(3), (-residual).view(-1).topk(3))

        qzero = -self.zero_point2 * self.scale2
        iqzero = torch.floor(qzero)

        if iqzero <= 0:
            torch.save(x, 'image_classification/ckpt/precon2x.pt')
            print("save for 2")
            print("part 2 break, x is {}, iqzero is {} \n".format(x, iqzero))

        if iqzero > 0:
            mx = (iqzero - self.num_bins) * mn / iqzero
        elif iqzero == 0:
            self.zero_point2, mn = 0, 0
        self.scale2 = self.num_bins / (mx - mn)

        if torch.isnan(self.scale2):
            torch.save(x, 'image_classification/ckpt/precon2x.pt')
            print("save for 2")

        second_transform = (residual - self.zero_point2) * self.scale2
        output = torch.cat([first_transform, second_transform], dim=0)
        # print("the integer is {}".format(output))
        # print("quantize shape is {}".format(output.shape))

        if debug:
            print("scale1 is {}, scale2 is {}, zero 1 {}, zero 2 {}".format(self.scale1, self.scale2, self.zero_point1,
                                                                            self.zero_point2))

        return output

    def inverse_transform(self, x):
        half_shape = int(x.shape[0] / 2)
        first, second = torch.split(x, [half_shape, half_shape], dim=0)
        # print("first shape is {}, second shape is {}".format(first.shape, second.shape))
        first = first / self.scale1 + self.zero_point1
        second = second / self.scale2 + self.zero_point2
        dequantize = torch.cat([first, second], dim=0)
        # print("scale1 is {}, scale2 is {}, zero 1 {}, zero 2 {}".format(self.scale1, self.scale2, self.zero_point1, self.zero_point2))
        # print("dequantize shape:{}".format(dequantize.shape))
        # print("input is {}, dequantize is {}, difference is ".format(x, dequantize, x-dequantize))
        if torch.isnan(dequantize[0, 0]):
            print("scale1 is {}, scale2 is {}, zero 1 {}, zero 2 {}".format(self.scale1, self.scale2, self.zero_point1,
                                                                            self.zero_point2))

        return dequantize

    def inverse_transform(self, x):
        half_shape = int(x.shape[0] / 2)
        first, second = torch.split(x, [half_shape, half_shape], dim=0)
        # print("first shape is {}, second shape is {}".format(first.shape, second.shape))
        first = first / self.scale1 + self.zero_point1
        second = second / self.scale2 + self.zero_point2
        dequantize = torch.cat([first, second], dim=0)
        # print("scale1 is {}, scale2 is {}, zero 1 {}, zero 2 {}".format(self.scale1, self.scale2, self.zero_point1, self.zero_point2))
        # print("dequantize shape:{}".format(dequantize.shape))
        # print("input is {}, dequantize is {}, difference is ".format(x, dequantize, x-dequantize))
        if torch.isnan(dequantize[0, 0]):
            print("scale1 is {}, scale2 is {}, zero 1 {}, zero 2 {}".format(self.scale1, self.scale2, self.zero_point1,
                                                                            self.zero_point2))

        return dequantize


class lsq_per_tensor(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, scale, config, bits, symm, inputtype=''):
        num_bins = 2 ** bits - 1
        bias = -num_bins / 2 if symm else 0
        num_features = input.numel()
        grad_scale = 1.0 / np.sqrt(num_features * num_bins)
        # grad_scale = 1.0 / np.sqrt(num_features)

        # Forward
        eps = 1e-7
        scale = scale + eps
        transformed = input / scale - bias
        vbar = torch.clamp(transformed, 0.0, num_bins).round()
        quantized = (vbar + bias) * scale

        # Step size gradient
        error = vbar - transformed
        mask = torch.logical_and(transformed >= 0, transformed <= num_bins)
        case1 = (transformed < 0).float() * bias
        case2 = mask.float() * error
        case3 = (transformed > num_bins).float() * (bias + num_bins)
        # TODO gradient scale might be too small, so optimizing without AdaGrad might be problematic...
        ss_gradient = (case1 + case2 + case3) * grad_scale  # * 100 * scale
        ctx.save_for_backward(mask, ss_gradient)
        ctx.others = config, inputtype
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        mask, ss_gradient = ctx.saved_tensors
        config, inputtype = ctx.others
        # if config.epoch < config.freeze_step and inputtype == "activation":
        #     return grad_output * mask.float(), (grad_output * ss_gradient).sum() * config.epoch / config.freeze_step, None, None, None, None
        return grad_output * mask.float(), (grad_output * ss_gradient).sum(), None, None, None, None


class LUQPreconditioner(Preconditioner):
    # y = (x - z) * scale
    # x = y / scale + z
    def __init__(self, x, num_bits, left=True):
        super(LUQPreconditioner, self).__init__(x, num_bits, left)

    def transform(self, x):
        # print(x.max(), x.min(), x.argmax(keepdim=False), x.argmin(keepdim=False))
        # print(x[x.argmax() // x.shape[1] - 8: x.argmax() // x.shape[1] + 2,
        #         x.argmax() % x.shape[1] - 5:x.argmax() % x.shape[1] + 5])
        # print(x.argmax() // x.shape[1], x.argmax() % x.shape[1], x.shape[0], x.shape[1])
        # print('_'*20)
        self.debug = False

        with torch.no_grad():

            mx = x.abs().max()
            self.max_bins = 2 ** (self.num_bits - 1)
            alpha = mx / 2 ** self.max_bins

            self.minivalue = 2 ** (-self.max_bins - 3)
            self.num_bins = self.max_bins + 1
            if self.debug:
                print(mx, alpha)
        if self.debug:
            print(x)
        sign = (x > 0)
        sign11 = sign.int() * 2 - torch.ones_like(x)
        if self.debug:
            print("sign", sign)
        thres = (x.abs() > alpha)
        sample_prob = (~thres) * x.abs() / alpha
        checkNAN(sample_prob, "sample prob")
        prob = torch.bernoulli(sample_prob)
        if self.debug:
            print("prob", prob)
        T = x * thres + sign11 * alpha * prob
        self.mid_ckpt = T

        self.alpha = alpha
        #
        if self.debug:
            print("T", T)

        output = T / alpha

        if self.debug:
            print("output", output)

        checkNAN(output, "luq output before log")
        logx, nearzero = self.log_with_0(output.abs())

        self.sign11 = sign11
        self.nearzero = nearzero
        output = logx * ~nearzero

        if self.debug:
            print("output log", output)
        checkNAN(output, "luq output after log")
        return output

    def log_with_0(self, x):
        small = (x < self.minivalue)
        small11 = small.int() * 2 - torch.ones_like(x)
        if self.debug:
            print("small", small)
        x = x + small * self.minivalue
        if self.debug:
            print("small x", x)
        logx = torch.log2(x) + torch.ones_like(x)
        if self.debug:
            print("logx", logx)
        return logx, small

    def exp_with_0(self, x, nearzero):
        powx = torch.pow(2, x) / 2
        if self.debug:
            print("pow x", powx)
        # print("nearzer0", nearzero)
        x = powx * ~nearzero
        if self.debug:
            print("pow nearzero x", x)
        return x

    def inverse_transform(self, x):
        if self.debug:
            print("inverse x", x)

        x = self.exp_with_0(x.abs(), self.nearzero) * self.sign11

        if self.debug:
            print("x", x)
        checkNAN(x, "luq inverse x")
        if self.debug:
            print(x)
        output = x * self.alpha

        if self.debug:
            print("final", output)

        return output


if __name__ == '__main__':
    pass

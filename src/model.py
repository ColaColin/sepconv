#
# KTH Royal Institute of Technology
#

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable, gradcheck
from src.separable_convolution import SeparableConvolutionSlow
from libs.sepconv.SeparableConvolution import SeparableConvolution
import src.config as config
import src.interpolate as interpol

from src.interpolate import _get_padding_modules

def _make_target_crop(width, height, target_w, target_h):
    tcw = (width - target_w)
    tch = (height - target_h)
    tcl = tcw // 2
    tcr = tcw - tcl
    tct = tch // 2
    tcb = tch - tct
    target_crop = nn.ReplicationPad2d([-tcl, -tcr, -tct, -tcb])
    return target_crop

class Net(nn.Module):

    def __init__(self, netMode, init_weights=True):
        super(Net, self).__init__()

        conv_kernel = (3, 3)
        conv_stride = (1, 1)
        conv_padding = 1
        sep_kernel = config.OUTPUT_1D_KERNEL_SIZE

        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.upsamp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu = nn.ReLU()

        self.netMode = netMode

        self.input_channels = 6 if "2to1" == self.netMode else 9

        self.conv32 = self._conv_module(self.input_channels, 32, conv_kernel, conv_stride, conv_padding, self.relu)
        self.conv64 = self._conv_module(32, 64, conv_kernel, conv_stride, conv_padding, self.relu)
        self.conv128 = self._conv_module(64, 128, conv_kernel, conv_stride, conv_padding, self.relu)
        self.conv256 = self._conv_module(128, 256, conv_kernel, conv_stride, conv_padding, self.relu)
        self.conv512 = self._conv_module(256, 512, conv_kernel, conv_stride, conv_padding, self.relu)
        self.conv512x512 = self._conv_module(512, 512, conv_kernel, conv_stride, conv_padding, self.relu)

        # only two options are supported: 2to1 or 3to2, which is used for any other string value than 2to1
        nHeads = 1 if "2to1" == self.netMode else 2

        if nHeads > 1:
            self.upsamp512 = nn.ModuleList()
            self.upconv256 = nn.ModuleList()
            self.upsamp256 = nn.ModuleList()
            self.upconv128 = nn.ModuleList()
            self.upsamp128 = nn.ModuleList()
            self.upconv64 = nn.ModuleList()
            self.upsamp64 = nn.ModuleList()

            self.upconv51_1 = nn.ModuleList()
            self.upconv51_2 = nn.ModuleList()
            self.upconv51_3 = nn.ModuleList()
            self.upconv51_4 = nn.ModuleList()

            for _ in range(nHeads):
                self.upsamp512.append(self._upsample_module(512, 512, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu))
                self.upconv256.append(self._conv_module(512, 256, conv_kernel, conv_stride, conv_padding, self.relu))
                self.upsamp256.append(self._upsample_module(256, 256, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu))
                self.upconv128.append(self._conv_module(256, 128, conv_kernel, conv_stride, conv_padding, self.relu))
                self.upsamp128.append(self._upsample_module(128, 128, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu))
                self.upconv64.append(self._conv_module(128, 64, conv_kernel, conv_stride, conv_padding, self.relu))
                self.upsamp64.append(self._upsample_module(64, 64, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu))

                self.upconv51_1.append(self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu))
                self.upconv51_2.append(self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu))
                self.upconv51_3.append(self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu))
                self.upconv51_4.append(self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu))

        else: # keep compatibility with old network structure that has nHeads == 1 and does not use lists
            self.upsamp512 = self._upsample_module(512, 512, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)
            self.upconv256 = self._conv_module(512, 256, conv_kernel, conv_stride, conv_padding, self.relu)
            self.upsamp256 = self._upsample_module(256, 256, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)
            self.upconv128 = self._conv_module(256, 128, conv_kernel, conv_stride, conv_padding, self.relu)
            self.upsamp128 = self._upsample_module(128, 128, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)
            self.upconv64 = self._conv_module(128, 64, conv_kernel, conv_stride, conv_padding, self.relu)
            self.upsamp64 = self._upsample_module(64, 64, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)

            self.upconv51_1 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)
            self.upconv51_2 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)
            self.upconv51_3 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)
            self.upconv51_4 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)


        self.pad = nn.ReplicationPad2d(sep_kernel // 2)

        if torch.cuda.is_available() and not config.ALWAYS_SLOW_SEP_CONV:
            self.separable_conv = SeparableConvolution.apply
        else:
            self.separable_conv = SeparableConvolutionSlow()

        if init_weights:
            print('Initializing weights...')
            self.apply(self._weight_init)

    @staticmethod
    def from_file(file_path: str, netMode: str) -> nn.Module :
        """
        Initializes a new Net object from an existing model
        :param file_path: path to the model.pth file
        :return: Net object
        """
        model = Net(netMode, init_weights=False)
        state_dict = torch.load(file_path)
        model.load_state_dict(state_dict)
        return model

    def to_file(self, file_path: str):
        """
        Writes the current model to disk as a .pth file
        :param file_path: path to the output model.pth file
        """
        torch.save(self.cpu().state_dict(), file_path)

    def interpolate(self, *args):
        return interpol.interpolate(self, *args)

    def interpolate_f(self, *args):
        return interpol.interpolate_f(self, *args)

    def interpolate_batch(self, *args):
        return interpol.interpolate_batch(self, *args)

    # use_padding = False still requires some padding in case the input size cannot be divided by 2 5 times to make the network architecture work out
    # this is the padding factory for that.
    def get_input_padding_for_32(self, width, height):
        pw = ((width // 32 + 1) * 32 - width)
        ph = ((height // 32 + 1) * 32 - height)
        pwl = pw // 2
        pht = ph // 2
        pwr = pw - pwl
        phb = ph - pht
        pads = [pwl, pwr, pht, phb]
        return torch.nn.ReplicationPad2d(pads), pads

    def _handle_paddings(self, use_padding, i1, i2):
        if use_padding:
            padded_i2 = self.pad(i2)
            padded_i1 = self.pad(i1)
        else:
            padded_i1 = i1.contiguous()
            padded_i2 = i2.contiguous()

        return padded_i1, padded_i2

    def _handle_no_pad_cut_kernels(self, i_pads, k1h, k1v, k2h, k2v):
        #cut away parts of the produced kernels that are never used anyway. Should not hurt training, since the network is fully convolutional, no training information is missed
        cut = config.OUTPUT_1D_KERNEL_SIZE // 2
        cutl = cut + i_pads[0]
        cutr = cut + i_pads[1]
        cutt = cut + i_pads[2]
        cutb = cut + i_pads[3]
        k2h = k2h[:,:,cutl:-cutr,cutt:-cutb].contiguous()
        k2v = k2v[:,:,cutl:-cutr,cutt:-cutb].contiguous()
        k1h = k1h[:,:,cutl:-cutr,cutt:-cutb].contiguous()
        k1v = k1v[:,:,cutl:-cutr,cutt:-cutb].contiguous()
        return k1h, k1v, k2h, k2v

    def _make_image_result_from_central_x(self, i1, i2, x, x64, x128, x256, x512, use_padding, i_pads, head_index):
        if  "2to1" == self.netMode:
            x = self.upsamp512(x)

            x += x512
            x = self.upconv256(x)

            x = self.upsamp256(x)
            x += x256
            x = self.upconv128(x)

            x = self.upsamp128(x)
            x += x128
            x = self.upconv64(x)

            x = self.upsamp64(x)
            x += x64

            # ------------ Final branches ------------

            k2h = self.upconv51_1(x)

            k2v = self.upconv51_2(x)

            k1h = self.upconv51_3(x)

            k1v = self.upconv51_4(x)

            padded_i1, padded_i2 = self._handle_paddings(use_padding, i1, i2)

            if not use_padding:
                k1h, k1v, k2h, k2v = self._handle_no_pad_cut_kernels(i_pads, k1h, k1v, k2h, k2v)

            # ------------ Local convolutions ------------

            result = self.separable_conv(padded_i2, k2v, k2h) + self.separable_conv(padded_i1, k1v, k1h)
        else:
            # ------------ Expansion ------------

            x = self.upsamp512[head_index](x)

            x += x512
            x = self.upconv256[head_index](x)

            x = self.upsamp256[head_index](x)
            x += x256
            x = self.upconv128[head_index](x)

            x = self.upsamp128[head_index](x)
            x += x128
            x = self.upconv64[head_index](x)

            x = self.upsamp64[head_index](x)
            x += x64

            # ------------ Final branches ------------

            k2h = self.upconv51_1[head_index](x)

            k2v = self.upconv51_2[head_index](x)

            k1h = self.upconv51_3[head_index](x)

            k1v = self.upconv51_4[head_index](x)

            padded_i1, padded_i2 = self._handle_paddings(use_padding, i1, i2)

            if not use_padding:
                k1h, k1v, k2h, k2v = self._handle_no_pad_cut_kernels(i_pads, k1h, k1v, k2h, k2v)

            # ------------ Local convolutions ------------

            result = self.separable_conv(padded_i2, k2v, k2h) + self.separable_conv(padded_i1, k1v, k1h)

        return result

    def _forward(self, inputs, use_padding):
        x = torch.cat(inputs, dim=1)

        i_pads = None
        if not use_padding:
            # make sure that the input can be divided by 2 as many times as needed by the pooling layers
            # this is not padding for the image size and is removed again in the end
            i_pad, i_pads = self.get_input_padding_for_32(x.shape[2], x.shape[3])
            x = i_pad(x)

        # ------------ Contraction ------------

        x = self.conv32(x)
        x = self.pool(x)

        x64 = self.conv64(x)
        x128 = self.pool(x64)

        x128 = self.conv128(x128)
        x256 = self.pool(x128)

        x256 = self.conv256(x256)
        x512 = self.pool(x256)

        x512 = self.conv512(x512)
        
        x = self.pool(x512)

        x = self.conv512x512(x)

        if self.netMode == "2to1":
            return [self._make_image_result_from_central_x(inputs[0], inputs[1], x, x64, x128, x256, x512, use_padding, i_pads, 0)]
        else:
            out1 = self._make_image_result_from_central_x(inputs[0], inputs[1], x, x64, x128, x256, x512, use_padding, i_pads, 0)
            out2 = self._make_image_result_from_central_x(inputs[1], inputs[2], x, x64, x128, x256, x512, use_padding, i_pads, 1)
            return [out1, out2]

    def forward(self, x, seq_length, use_padding, crop_for_training):
        if (self.netMode == "2to1"):
            # 2to1 uses the 6 input layers, 3 for each image, RGB
            workspace = [x[:, :3], x[:, 3:6]]
            takeN = 2
        else:
            # 3to2 uses 9 input layers, one extra image compared to 2to1
            workspace = [x[:, :3], x[:, 3:6], x[:, 6:9]]
            takeN = 3

        putN = takeN - 1

        # apply the network until enough output images were generated
        while len(workspace) < seq_length:
            new_workspace = []
            for wi in range(0, len(workspace) - putN, putN):
                inputs = []
                for ir in range(takeN):
                    inputs.append(workspace[wi+ir])
                
                # determine the size of the smallest image generated, which will be used as a smallest common denominator for the next forward pass through the network
                smallestw = inputs[0].shape[2]
                for inp in inputs:
                    if smallestw > inp.shape[2]:
                        smallestw = inp.shape[2]

                smallesth = inputs[1].shape[3]
                for inp in inputs:
                    if smallesth > inp.shape[3]:
                        smallesth = inp.shape[3]

                for idx in range(takeN):
                    inp = inputs[idx]
                    i_crop = _make_target_crop(inp.shape[2], inp.shape[3], smallestw, smallesth)
                    inputs[idx] = i_crop(inp)

                outputs = self._forward(inputs, use_padding)

                ix = 0
                for o in outputs:
                    new_workspace.append(inputs[ix])
                    new_workspace.append(o)
                    ix += 1

            new_workspace.append(workspace[-1])
            workspace = new_workspace

        result = []
        for wi in range(len(workspace)):
            if (wi != 0 and wi != len(workspace)-1 and (self.netMode == "2to1" or wi != len(workspace) // 2)):
                w = workspace[wi]
                if crop_for_training:
                    t_crop = _make_target_crop(w.shape[2], w.shape[3], config.CROP_SIZE, config.CROP_SIZE)
                    cropped = t_crop(w)
                    result.append(cropped)
                else:
                    result.append(w)

        return torch.cat(result, dim=1)

    @staticmethod
    def _check_gradients(func):
        print('Starting gradient check...')
        sep_kernel = config.OUTPUT_1D_KERNEL_SIZE
        inputs = (
            Variable(torch.randn(2, 3, sep_kernel, sep_kernel).cuda(), requires_grad=False),
            Variable(torch.randn(2, sep_kernel, 1, 1).cuda(), requires_grad=True),
            Variable(torch.randn(2, sep_kernel, 1, 1).cuda(), requires_grad=True),
        )
        test = gradcheck(func, inputs, eps=1e-3, atol=1e-3, rtol=1e-3)
        print('Gradient check result:', test)

    @staticmethod
    def _conv_module(in_channels, out_channels, kernel, stride, padding, relu):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel, stride, padding), relu,
            torch.nn.Conv2d(in_channels, in_channels, kernel, stride, padding), relu,
            torch.nn.Conv2d(in_channels, out_channels, kernel, stride, padding), relu,
        )

    @staticmethod
    def _kernel_module(in_channels, out_channels, kernel, stride, padding, upsample, relu):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel, stride, padding), relu,
            torch.nn.Conv2d(in_channels, in_channels, kernel, stride, padding), relu,
            torch.nn.Conv2d(in_channels, out_channels, kernel, stride, padding), relu,
            upsample,
            torch.nn.Conv2d(out_channels, out_channels, kernel, stride, padding)
        )

    @staticmethod
    def _upsample_module(in_channels, out_channels, kernel, stride, padding, upsample, relu):
        return torch.nn.Sequential(
            upsample, torch.nn.Conv2d(in_channels, out_channels, kernel, stride, padding), relu,
        )

    @staticmethod
    def _weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.orthogonal_(m.weight, init.calculate_gain('relu'))

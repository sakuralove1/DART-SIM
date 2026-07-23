import os
import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d as scipy_interp1d

from utils.option import product_of_tuple_elements
from utils.mrc import ReadMRC

from SIR_core.function_for_NN import conventional_reconstruction_in_neural_network
from DART_SIM_model.common import generate_pattern

device = torch.device('cuda')


class ModelBase(object):
    def __init__(self, opt):
        self.opt = opt  # opt
        self.save_dir = os.path.join(opt['save_path'], 'model')  # save models
        self.device = torch.device('cuda')  # single gpu | distributed training is not implemented
        self.is_train = opt['is_train']  # training or not
        self.schedulers = {}  # schedulers
        self.factor = 1.33  # 1 + num_phase / num_phase # it is a MYSTERY in the original program

        # ------------------------------------
        # data
        # ------------------------------------
        self.para_data = None
        self.json_path = None
        self.wf_input_data = None
        self.wf_target_data = None
        self.wf_infer_data = None
        self.raw_input_data = None
        self.raw_target_data = None
        self.raw_infer_data = None
        self.sim_input_data = None
        self.sim_target_data = None
        self.sim_infer_data = None
        # self.raw_input_pattern = None
        self.raw_target_pattern = None
        self.raw_infer_pattern = None
        self.predemod_pattern = None
        self.predemod_patternpsf = None

        self.OTF_TRAIN = None

        # ------------------------------------
        # log
        # ------------------------------------
        self.log_dict = None

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # network name and number of parameters
    # ----------------------------------------
    def describe_network(self, network):
        if isinstance(network, nn.DataParallel):
            network = network.module
        msg = '\n'
        msg += 'Networks name: {}'.format(network.__class__.__name__) + '\n'
        msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), network.parameters()))) + '\n'
        msg += 'Net structure:\n{}'.format(str(network)) + '\n'
        return msg

    # ----------------------------------------
    # parameters description
    # ----------------------------------------
    def describe_params(self, network):
        if isinstance(network, nn.DataParallel):
            network = network.module
        msg = '\n'
        msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'shape', 'param_name') + '\n'
        for name, param in network.state_dict().items():
            if not 'num_batches_tracked' in name:
                v = param.data.clone().float()
                msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}'.format(v.mean(), v.min(), v.max(), v.std(), v.shape, name) + '\n'
        return msg

    # ----------------------------------------
    # save the state_dict of the network
    # ----------------------------------------
    @staticmethod
    def save_network(save_dir, network, network_label, iter_label):
        save_filename = '{:0>6}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # ----------------------------------------
    # load the state_dict of the network
    # ----------------------------------------
    @staticmethod
    def load_network(load_path, network, strict=True):
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path), strict=strict)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self, net_list):
        return "\n".join(self.describe_network(net) for net in net_list)

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self, net_list):
        return "\n".join(self.describe_params(net) for net in net_list)

    # ----------------------------------------
    # learning rate during training
    # ----------------------------------------
    def current_learning_rate(self):
        dict_out = {}
        for key in self.schedulers.keys():
            dict_out[key] = self.schedulers[key].get_last_lr()[0]
        return dict_out

    def update_learning_rate(self, net):
        self.schedulers[net].step()

    # ----------------------------------------
    # cal loss for <tensor> or <list of tensor>
    # ----------------------------------------
    @staticmethod
    def cal_loss(infer_data_list, target_data, lossfn):
        if isinstance(infer_data_list, list):  # progressive learning
            if isinstance(target_data, list):  # sim-raw
                factor = 4 / target_data[1].shape[1]
                infer_data = infer_data_list[0]
                G_loss = lossfn(infer_data_list[0], target_data[0]) + factor * lossfn(infer_data_list[1], target_data[1])
                G_loss /= len(infer_data_list)
            else:  # sim-sim
                infer_data = infer_data_list[0]
                G_loss = lossfn(infer_data_list[0], target_data) + lossfn(infer_data_list[1], target_data)
                G_loss /= len(infer_data_list)
        else:
            infer_data = infer_data_list
            G_loss = lossfn(infer_data_list, target_data)
        return G_loss, infer_data

    # ----------------------------------------
    # do reconstruction
    # ----------------------------------------
    @staticmethod
    def conv_rec(input_data, para_data, num_channels_in, opt_class):
        # num_channels_in: OPC
        in_shape = input_data.shape
        if len(in_shape) == 3:  # 2D [CHW]
            T = 1
            O, P, C = num_channels_in
            D = 1
            _, H, W = in_shape
            input_data = input_data.reshape(T, O, P, C, D, H, W)  # [ChannelHW] -> [TOPCDHW]
        elif len(in_shape) == 4:  # time stack [CTHW]
            if in_shape[0] != 9:
                raise NotImplementedError
            _, T, H, W = in_shape
            O, P, C = num_channels_in
            D = 1
            input_data = input_data.permute(1, 0, 2, 3).reshape(T, O, P, C, D, H, W)  # [ChannelTHW] -> [TChannelHW] -> [TOPCDHW]
        else:
            raise NotImplementedError

        return conventional_reconstruction_in_neural_network(input_data, para_data, opt_class)

    @staticmethod
    def tensor2format(img, out_shape):
        img = img.float()
        T = 1
        (O, P, C) = out_shape
        (H, W) = (img.shape[-2], img.shape[-1])
        D = product_of_tuple_elements(img.shape) // product_of_tuple_elements((T, O, P, C, H, W))
        return img.reshape(T, O, P, C, D, H, W)

    # ----------------------------------------
    # generate pattern in raw data size - for two stream raw denoise network
    # ----------------------------------------
    @staticmethod
    def _generate_pattern(raw_input_data, para_data, json_path, modamp=0.5, center_ratio=0.5, pattern_size='raw'):  # pattern formed in raw data size
        return generate_pattern(raw_input_data, para_data, json_path, modamp, center_ratio, pattern_size)

    # ----------------------------------------
    # obtain OTF
    # ----------------------------------------
    def set_otf_while_training(self, opt, Nx, Ny, Nz=1, state='test'):  # for 2d only
        if state == 'train' and self.OTF_TRAIN is not None:
            # Assume training data use the same OTF
            return self.OTF_TRAIN
        else:
            if self.opt['data'] == '2d-sim':

                try:
                    rm_OTF = ReadMRC(opt['otf_path'], is_SIM_rawdata=False)
                except FileNotFoundError:
                    rm_OTF = ReadMRC(os.path.join(*opt['otf_path'].split('\\')[1:]), is_SIM_rawdata=False)
                rawOTF = rm_OTF.get_total_data_as_mat(convert_to_tensor=False).squeeze()
                nxotf, dkrotf = rm_OTF.opt['num_pixel_width'], rm_OTF.opt['height_space_sampling']
                diagdist = int(np.sqrt(np.square(Nx / 2) + np.square(Ny / 2)) + 2)
                OTF = np.real(rawOTF)
                x = np.arange(0, nxotf * dkrotf, dkrotf)
                dkx = 1 / (Nx * opt['width_space_sampling'])
                dky = 1 / (Ny * opt['height_space_sampling'])
                dkr = min(dkx, dky)
                xi = np.arange(0, (nxotf - 1) * dkrotf, dkr)
                interp = scipy_interp1d(x, OTF, kind='slinear')
                OTF = interp(xi)
                sizeOTF = len(OTF)
                prol_OTF = np.zeros((diagdist * 2))
                prol_OTF[0:sizeOTF] = OTF
                OTF = prol_OTF
                kxx = dkx * np.arange(-Nx / 2, Nx / 2, 1)
                kyy = dky * np.arange(-Ny / 2, Ny / 2, 1)
                [dX, dY] = np.meshgrid(kxx, kyy)
                rdist = np.sqrt(np.square(dX) + np.square(dY))
                otflen = len(OTF)
                x = np.arange(0, otflen * dkr, dkr)
                interp = scipy_interp1d(x, OTF, kind='slinear')
                OTF = interp(rdist)
                OTF = torch.from_numpy(OTF).to(device).float()
                OTF = OTF / OTF.max()
                return OTF

            else:

                raise RuntimeError

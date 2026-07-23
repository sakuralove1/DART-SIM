import functools
from torch.nn import init


def define_G(opt):
    if opt['data'] in ['2d-sim']:

        # --------------------------------------------
        # 2d reconstruction
        # --------------------------------------------
        if opt['net_G'] == 'phct':
            from DART_SIM_model.network_phct import PHCT
            netG = PHCT(in_nc=opt['net_channel_in'], out_nc=opt['net_channel_out'], scale=opt['net_scale'], para1=opt['net_G_para1'], para2=opt['net_G_para2'], para3=opt['net_G_para3'],
                        para4=opt['net_G_para4'])
        elif opt['net_G'] in ['dartsim_prppf_mambav2_ft85', 'dartsim_prppf_mambav2_ft70']:
            from DART_SIM_model.network_phct import DARTSIM_PRPPF
            default_jitter_prob = 0.15 if opt['net_G'] == 'dartsim_prppf_mambav2_ft85' else 0.30
            netG = DARTSIM_PRPPF(
                in_nc=opt['net_channel_in'],
                out_nc=opt['net_channel_out'],
                scale=opt['net_scale'],
                para1=opt['net_G_para1'] if opt['net_G_para1'] is not None else 'RLFB+ET+Conv3',
                para2=opt['net_G_para2'] if opt['net_G_para2'] is not None else 'RLFB+DPMB+Conv3',
                para3=opt['net_G_para3'],
                para4=opt['net_G_para4'],
                jitter_k0_scale=opt.get('prppf_jitter_k0_scale') if opt.get('prppf_jitter_k0_scale') is not None else 0.01,
                jitter_k0_angle_deg=opt.get('prppf_jitter_k0_angle_deg') if opt.get('prppf_jitter_k0_angle_deg') is not None else 0.5,
                jitter_phase=opt.get('prppf_jitter_phase') if opt.get('prppf_jitter_phase') is not None else 0.05,
                jitter_random=opt.get('prppf_jitter_random') if opt.get('prppf_jitter_random') is not None else True,
                sym_k0_scale=opt.get('prppf_sym_k0_scale') if opt.get('prppf_sym_k0_scale') is not None else 0.0025,
                sym_k0_angle_deg=opt.get('prppf_sym_k0_angle_deg') if opt.get('prppf_sym_k0_angle_deg') is not None else 0.125,
                sym_phase=opt.get('prppf_sym_phase') if opt.get('prppf_sym_phase') is not None else 0.025,
                jitter_prob=opt.get('prppf_jitter_prob') if opt.get('prppf_jitter_prob') is not None else default_jitter_prob
            )

        else:
            print(opt['net_G'])
            raise NotImplementedError

    else:
        raise NotImplementedError

    if opt['is_train']:
        if opt['init_type']:
            init_weights(netG, init_type=opt['init_type'], gain=opt['init_gain'])
            opt['outinfo'].info('Initialization method [{:s}], bn [{:s}], gain is [{:.2f}]\n'.format(opt['init_type'], 'uniform', opt['init_gain']))
        else:
            opt['outinfo'].info('Warning! Do nothing in init_weights  |  The initialization method shoule be involved in network init code\n')
    return netG


# --------------------------------------------
# weights initialization
# --------------------------------------------
def init_weights(net, init_type='xavier_uniform', gain=0.2):
    def init_fn(m, init_type='kaiming_normal', gain=0.2):
        classname = m.__class__.__name__
        if classname in ['Conv2d', 'Linear']:  # and classname.find('_selfinit') == -1: # if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            if init_type in ['default', 'none']:
                pass
            elif init_type == 'normal_default':
                init.normal_(m.weight.data, 0, 0.1)
            elif init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)
            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)
            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)
            elif init_type == 'kaiming_normal_leaky_relu':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)
            elif init_type == 'kaiming_uniform_leaky_relu':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            if m.affine:
                init.uniform_(m.weight.data, 0.1, 1.0)
                init.constant_(m.bias.data, 0.0)

    if init_type not in ['default', 'none']:
        fn = functools.partial(init_fn, init_type=init_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network definition!')

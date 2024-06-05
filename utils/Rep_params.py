import torch
import torch.nn as nn
import torch.nn.functional as F

def rep_params(model, model_plain, opt, device):
    state_dict_model = model.state_dict()
    state_dict_model_plain = model_plain.state_dict()
    
    for key, value in state_dict_model_plain.items():
        if key in state_dict_model:
            state_dict_model_plain[key] = state_dict_model[key]
        else:
            for m in range(opt.M_MFDB):
                for k in range(opt.K_RepConv):
                    key_weight_model_plain = 'M_block.{}.mfdb.{}.weight'.format(m, k)
                    key_bias_model_plain = 'M_block.{}.mfdb.{}.bias'.format(m, k)
                    if key == key_weight_model_plain:
                        Ka = state_dict_model['M_block.{}.mfdb.{}.repconv.0.weight'.format(m, k)]
                        Ba = state_dict_model['M_block.{}.mfdb.{}.repconv.0.bias'.format(m, k)]
                        Kb = state_dict_model['M_block.{}.mfdb.{}.repconv.1.weight'.format(m, k)]
                        Bb = state_dict_model['M_block.{}.mfdb.{}.repconv.1.bias'.format(m, k)]
                        Kc = state_dict_model['M_block.{}.mfdb.{}.repconv.2.weight'.format(m, k)]
                        Bc = state_dict_model['M_block.{}.mfdb.{}.repconv.2.bias'.format(m, k)]

                        # 1*1 + 3*3的合并
                        Kab = F.conv2d(Kb, Ka.permute(1, 0, 2, 3))
                        Bab = torch.ones(1, 32, 3, 3, device=device)*Ba.view(1, -1, 1, 1)
                        Bab = F.conv2d(Bab, Kb).view(-1,) + Bb
                        
                        # 3*3 + 1*1的合并
                        Kabc = torch.rand((16, 16, 3, 3)).to(device)
                        Babc = torch.rand([16]).to(device)
                        for i in range(Kc.shape[0]):
                            Kabc[i,...] = torch.sum(Kab * Kc[i,...].unsqueeze(1), dim=0)
                            Babc[i] = Bc[i] + torch.sum(Bab * Kc[i,...].squeeze(1).squeeze(1))

                        a = torch.eye(16).view(16, 16, 1, 1)
                        a = torch.nn.functional.pad(a, pad=(1, 1, 1, 1, 0, 0, 0, 0), mode='constant', value=0).to(device)
                        state_dict_model_plain[key_weight_model_plain] = Kabc + a
                        state_dict_model_plain[key_bias_model_plain] = Babc
    model_plain.load_state_dict(state_dict_model_plain)
    return model_plain

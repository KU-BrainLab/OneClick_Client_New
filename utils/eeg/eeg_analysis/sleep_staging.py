import mne
import copy
import torch
from pathlib import Path
from neuronet.model import NeuroNet, NeuroNetEncoderWrapper, Classifier

def get_sleep_staging(epoch_data, ch_list):
    epoch_data = copy.deepcopy(epoch_data)
    info = epoch_data.info
    epoch_data = epoch_data.get_data()

    scaler = mne.decoding.Scaler(info=info, scalings='median')
    epoch_data = scaler.fit_transform(epoch_data)

    epoch_data1 = epoch_data[:, ch_list.index('C4'), :].squeeze()
    epoch_data1 = torch.tensor(epoch_data1, dtype=torch.float32)

    epoch_data2 = epoch_data[:, ch_list.index('C3'), :].squeeze()
    epoch_data2 = torch.tensor(epoch_data2, dtype=torch.float32)

    outs = []
    for i in range(5):
        # 1. Prepared Pretrained Model
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent

        ckpt_path = project_root / 'neuronet' / 'ckpt' / str(i) / 'model' / 'best_model.pth'
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model_parameter = ckpt['model_parameter']
        pretrained_model = NeuroNet(**model_parameter)
        pretrained_model.load_state_dict(ckpt['model_state'])

        # 2. Encoder Wrapper
        backbone = NeuroNetEncoderWrapper(
            fs=model_parameter['fs'], second=model_parameter['second'],
            time_window=model_parameter['time_window'], time_step=model_parameter['time_step'],
            frame_backbone=pretrained_model.frame_backbone,
            patch_embed=pretrained_model.autoencoder.patch_embed,
            encoder_block=pretrained_model.autoencoder.encoder_block,
            encoder_norm=pretrained_model.autoencoder.encoder_norm,
            cls_token=pretrained_model.autoencoder.cls_token,
            pos_embed=pretrained_model.autoencoder.pos_embed,
            final_length=pretrained_model.autoencoder.embed_dim
        )

        # 3. Generator Classifier
        model = Classifier(backbone=backbone,
                           backbone_final_length=pretrained_model.autoencoder.embed_dim)
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        ckpt_path = project_root / 'neuronet' / 'ckpt' / str(i) / 'linear_prob' / 'best_model.pth'

        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state'])

        with torch.no_grad():
            model.eval()
            out1 = model(epoch_data1)
            out1 = torch.softmax(out1, dim=-1)

            out2 = model(epoch_data2)
            out2 = torch.softmax(out2, dim=-1)
            outs.append(out1 + out2)
    outs = torch.stack(outs)
    outs = torch.mean(outs, dim=0)
    sleep_stage = torch.argmax(outs, dim=-1)
    sleep_stage_prob = outs
    sleep_stage = sleep_stage.cpu().detach().numpy()
    sleep_stage_prob = sleep_stage_prob.cpu().detach().numpy()
    return {
        'sleep_stage': list(sleep_stage),
        'sleep_stage_prob': list([list(prob) for prob in sleep_stage_prob])
    }
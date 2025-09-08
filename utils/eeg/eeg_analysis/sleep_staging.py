import mne
import copy
import torch
from pathlib import Path
from neuronet.model import NeuroNet, NeuroNetEncoderWrapper, Classifier

def compute_sleep_metrics(stage_list, epoch_sec: int = 30):
    sleep_labels = {1, 2, 3, 4}
    wake_label = 0

    n_epochs = len(stage_list)
    epoch_min = epoch_sec / 60.0

    # Time in Bed (TIB)
    tib = n_epochs * epoch_min

    # Sleep onset (첫 수면 epoch index)
    try:
        sleep_onset_idx = next(i for i, s in enumerate(stage_list) if s in sleep_labels)
        sleep_latency = sleep_onset_idx * epoch_min
    except StopIteration:
        sleep_onset_idx = None
        sleep_latency = None

    # REM onset
    rem_latency = None
    if sleep_onset_idx is not None:
        try:
            rem_idx = next(i for i, s in enumerate(stage_list[sleep_onset_idx:], start=sleep_onset_idx) if s == 4)
            rem_latency = (rem_idx - sleep_onset_idx) * epoch_min
        except StopIteration:
            pass

    # Total Sleep Time (TST)
    tst = sum(1 for s in stage_list if s in sleep_labels) * epoch_min

    # WASO: sleep onset 이후 wake epochs
    waso = None
    if sleep_onset_idx is not None:
        waso = sum(1 for s in stage_list[sleep_onset_idx:] if s == wake_label) * epoch_min

    # Total Wake Time (TWT) = Sleep Latency + WASO
    twt = None
    if sleep_latency is not None and waso is not None:
        twt = sleep_latency + waso

    # Sleep efficiency
    sleep_eff = (tst / tib * 100.0) if tib > 0 else None

    return {
        "TIB (min)": tib,
        "TST (min)": tst,
        "TWT (min)": twt,
        "WASO (min)": waso,
        "Sleep Latency (min)": sleep_latency,
        "REM Latency (min)": rem_latency,
        "Sleep Efficiency (%)": sleep_eff,
    }

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

    sleep_stage = list(sleep_stage)

    #report
    total_epoch = len(list([list(prob) for prob in sleep_stage_prob]))
    w_tst = sleep_stage.count(0) / total_epoch * 100 
    n1_tst = sleep_stage.count(1) / total_epoch * 100 
    n2_tst = sleep_stage.count(2) / total_epoch * 100 
    n3_tst = sleep_stage.count(3) / total_epoch * 100
    nrem_tst = n1_tst + n2_tst + n3_tst
    rem_tst = sleep_stage.count(4) / total_epoch * 100 

    w_min = sleep_stage.count(0) * 30 / 60
    n1_min = sleep_stage.count(1) * 30 / 60
    n2_min = sleep_stage.count(2) * 30 / 60
    n3_min = sleep_stage.count(3) * 30 / 60
    nrem_min = n1_min + n2_min + n3_min
    rem_min = sleep_stage.count(4) * 30 / 60

    sleep_summary = compute_sleep_metrics(sleep_stage, 30)
    sleep_summary['sleep_tst'] = list([w_tst, n1_tst, n2_tst, nrem_tst, rem_tst])
    sleep_summary['sleep_min'] = list([w_min, n1_min, n2_min, nrem_min, rem_min])

    return {
        'sleep_stage': sleep_stage,
        'sleep_stage_prob': list([list(prob) for prob in sleep_stage_prob]),
        'sleep_summary' : sleep_summary
    }




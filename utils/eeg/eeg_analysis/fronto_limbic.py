import copy
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import os
import networkx as nx
import base64

def compute_plv(data):
    n_channels, n_samples = data.shape
    plv_matrix = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase_diff = np.angle(hilbert(data[i])) - np.angle(hilbert(data[j]))
            plv = np.abs(np.sum(np.exp(1j * phase_diff)) / n_samples)
            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv
    return plv_matrix

def get_fronto_limbic_analysis(epoch_data, uuid):
    wanted_ch_list = ['F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'T3', 'T4']
    wanted_ch_indices = [2, 10, 1, 11, 0, 8, 3, 12]
    freq_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 49)
    }

    epoch_data = copy.deepcopy(epoch_data)
    result = {}
    for band_name, band_freqs in freq_bands.items():
        filtered_data = copy.deepcopy(epoch_data)
        filtered_data = filtered_data.load_data().filter(l_freq=band_freqs[0], h_freq=band_freqs[1]).get_data()
        montage = mne.channels.make_standard_montage('standard_1020')
        info = mne.create_info(wanted_ch_list, sfreq=125, ch_types='eeg')
        info.set_montage(montage)

        matrix = compute_plv(filtered_data)

        fig, ax = plt.subplots(figsize=(10, 10))
        # Plot empty head layout instead of using plot_sensors and adjust the position of the channels
        pos = {ch: (loc[0], loc[1] - 0.025) for ch, loc in
               zip(info.ch_names, info.get_montage().get_positions()['ch_pos'].values())}
        pos_array = np.array(list(pos.values()))

        # Plot the head layout without markers
        mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='w')
        mne.viz.plot_topomap(np.zeros(len(wanted_ch_list)), pos_array,
                             show=False, axes=ax, mask=np.ones(len(wanted_ch_list)), mask_params=mask_params)
        G = nx.Graph()
        for idx, ch in enumerate(wanted_ch_list):
            G.add_node(ch, pos=pos[ch])

        for i in range(len(wanted_ch_list)):
            for j in range(i + 1, len(wanted_ch_list)):
                weight = matrix[i, j]
                G.add_edge(wanted_ch_list[i], wanted_ch_list[j], weight=weight)

        edges = G.edges(data=True)
        colors = [edge[2]['weight'] for edge in edges]
        nx.draw(G, pos, with_labels=False, node_size=100, node_color='gray', ax=ax, font_size=12, edge_color=colors,
                edge_cmap=plt.cm.viridis, edge_vmin=0, edge_vmax=1)

        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, orientation='horizontal', label='PLV')

        tmp_name = os.path.join('image', 'fronto_limbic', '{}_{}.jpg'.format(uuid, band_name))
        tmp_name = os.path.abspath(tmp_name)
        fig.savefig(tmp_name)

        with open(tmp_name, 'rb') as f:
            im_bytes = f.read()
        im_b64 = base64.b64encode(im_bytes).decode("utf8")
        result[band_name] = im_b64
        plt.close('all')
        plt.clf()
    return result
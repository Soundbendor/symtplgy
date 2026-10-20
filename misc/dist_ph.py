import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from util import util_main as UMN
from util import util_constants as UC
from util import util_ph as UPH
import gudhi

datasets = ['polyrhythms']

models = ['musicgen-small']
inst1 = 'Agogo'
inst2 = 'Woodblock'
offset_lvl = 0
rvb_lvl = 0
bpm = 60
suffix = 1

symb = ['onevoice', 'twovoice']
groupdir = UMN.by_projpath(UC.PH_GROUP_FOLDER)
graphdir = UMN.by_projpath(UC.PH_GRAPH_FOLDER)

dists = ['wass', 'bottle']

dist_pretty = {'wass': "Wasserstein", "bottle": "Bottleneck"}
symb_pretty = {'onevoice': "one voice", "twovoice": "two voice"}
feat_pretty = {'h0': "H0", 'h1': "H1", 'total': "H0+H1"}

CHART_FOLDER = 'ph_dists'

def plot_per_dist(dist_dict, cur_ds, cur_model, cur_dist, cur_symb, cur_feat, cur_suffix):
    dist_pprint = dist_pretty[cur_dist]
    symb_pprint = symb_pretty[cur_symb]
    feat_pprint = feat_pretty[cur_feat]
    ds_pprint = UC.DATASET_PPRINT[cur_ds]
    model_pprint = UC.MODEL_PPRINT[cur_model]
    cur_title = f'{feat_pprint} {dist_pprint} Distance for {ds_pprint} ({model_pprint}, {symb_pprint})'
    cur_xlabel = 'Layer'
    fig, ax = plt.subplots(figsize=(7, 5))
    cur_arr = dist_dict[cur_dist][cur_symb][cur_feat]
    ax.plot(np.arange(len(cur_arr)),cur_arr)
    ax.set_xlabel(cur_xlabel)
    ax.set_ylabel(f'{dist_pprint} Distance')
    ax.set_title(cur_title)
    plt.tight_layout()
    res_dir = UMN.by_projpath_multi(subpaths=[CHART_FOLDER, cur_ds, cur_model], make_dir = True)
    fname = f'{cur_dist}-{cur_feat}-{cur_symb}-{cur_suffix}.png'
    fpath = os.path.join(res_dir, fname)
    plt.savefig(fpath)
    fig.clear()
    plt.close()



for ds in datasets:
    for m in models:
        model_shape = UMN.get_acts_shape(m)
        m_dim = model_shape[1]
        m_num_layers = model_shape[0]
        cur_dists = {d: {s: {'h0': [], 'h1': [], 'total': []} for s in symb} for d in dists}
        for l in range(m_num_layers):
            h0_file = f'l{l}_inst1_{inst1}-inst2_{inst2}-offset_lvl_{offset_lvl}-rvb_lvl_{rvb_lvl}-bpm_{bpm}-h0-{suffix}.npy'
            h1_file = f'l{l}_inst1_{inst1}-inst2_{inst2}-offset_lvl_{offset_lvl}-rvb_lvl_{rvb_lvl}-bpm_{bpm}-h1-{suffix}.npy'
            h0 = np.load(os.path.join(groupdir,ds,m,h0_file))
            h1 = np.load(os.path.join(groupdir,ds,m,h1_file))
            dgm = [h0,h1]
            for s in symb:
                h0_file = f'{ds}_{s}-h0.npy'
                h1_file = f'{ds}_{s}-h1.npy'
                h0 = np.load(os.path.join(groupdir,ds,'symbolic',h0_file))
                h1 = np.load(os.path.join(groupdir,ds,'symbolic',h1_file))
                cur_s = [h0,h1]
                cur_wass_h0 = gudhi.hera.wasserstein_distance(cur_s[0], dgm[0])
                cur_wass_h1 = gudhi.hera.wasserstein_distance(cur_s[1], dgm[1])
                cur_bottle_h0 = gudhi.bottleneck_distance(cur_s[0], dgm[0])
                cur_bottle_h1 = gudhi.bottleneck_distance(cur_s[1], dgm[1])
                cur_dists['wass'][s]['h0'].append(cur_wass_h0)
                cur_dists['wass'][s]['h1'].append(cur_wass_h1)
                cur_dists['wass'][s]['total'].append(cur_wass_h0 + cur_wass_h1)
                cur_dists['bottle'][s]['h0'].append(cur_bottle_h0)
                cur_dists['bottle'][s]['h1'].append(cur_bottle_h1)
                cur_dists['bottle'][s]['total'].append(cur_bottle_h0 + cur_bottle_h1)
        for d in dists:
            for s in symb:
                for f in ['h0','h1', 'total']:
                    plot_per_dist(cur_dists, ds, m, d, s, f, suffix)



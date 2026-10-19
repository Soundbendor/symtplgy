from . import util_constants as UC
from . import util_main as UMN
import gudhi
import matplotlib.pyplot as plt
import numpy as np
from persim import plot_diagrams
from scipy.spatial.distance import pdist, squareform
from ripser import ripser


def save_ph_groups(dgm, configdict, layer_idx, tomlstr=''):
    layer_str = f'l{layer_idx}'
    for ph_idx,ph_group in enumerate(dgm):
        save_str = None
        if len(tomlstr) > 0:
            save_str = f'{layer_str}_{tomlstr}-h{ph_idx}'
        else:
            save_str = f'{layer_str}-h{ph_idx}'
        save_path = UMN.get_save_path('ph_group', configdict, other=save_str, make_dir = True)
        np.save(save_path, np.array(ph_group))



def save_ph_diagram_barcode(dgm, configdict, layer_idx, tomlstr=''):
    layer_str = f'l{layer_idx}'
    diag_str = None
    barcode_str = None
    if len(tomlstr) > 0:
        diag_str = f'{layer_str}_{tomlstr}-diagram'
        barcode_str = f'{layer_str}_{tomlstr}-barcode'
    else:
        diag_str = f'{layer_str}-diagram'
        barcode_str = f'{layer_str}-barcode'

    diag_path = UMN.get_save_path('ph_graph', configdict, other=diag_str, make_dir = True)
    barcode_path = UMN.get_save_path('ph_graph', configdict, other=barcode_str, make_dir = True)
    
    gudhi.plot_persistence_diagram(dgm)

    fig = plt.figure()
    plt.savefig(diag_path)
    fig.clear()
    plt.close()

    gudhi.plot_persistence_barcode(dgm)
    
    fig = plt.figure()
    plt.savefig(diag_path)
    fig.clear()
    plt.close()


def get_pairwise_distances(cur_data, norm=True):
    ret= pdist(cur_data, metric='euclidean')
    if norm == True:
        return ret/ret.max()
    else:
        return ret


def calc_rips(cur_data, maxdim = 1, calc_distance = True):
    cur_rips = None
    if calc_distance == False:
        cur_rips = ripser(cur_data, maxdim = maxdim, distance_matrix = False, metric = 'euclidean')
    else:
        dists = squareform(get_pairwise_distances(b_arr, norm=True)*100.)
        cur_rips = ripser(dists, maxdim = maxdim, distance_matrix = True)
    return cur_rips

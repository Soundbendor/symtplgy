import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from util import util_main as UMN
from util import util_constants as UC
from util import util_ph as UPH
import gudhi

datasets = ['polyrhythms']

models = ['musicgen-small', 'musicgen-medium', 'musicgen-large']
inst1 = 'Agogo'
inst2 = 'Woodblock'
offset_lvl = 0
rvb_lvl = 0
bpm = 60
suffix = 1
groupdir = UMN.by_projpath(UC.PH_GROUP_FOLDER)
graphdir = UMN.by_projpath(UC.PH_GRAPH_FOLDER)
for ds in datasets:
    for m in models:
        model_shape = UMN.get_acts_shape(m)
        m_dim = model_shape[1]
        m_num_layers = model_shape[0]
        for l in range(m_num_layers):
            h0_file = f'l{l}_inst1_{inst1}-inst2_{inst2}-offset_lvl_{offset_lvl}-rvb_lvl_{rvb_lvl}-bpm_{bpm}-h0-{suffix}.npy'
            h1_file = f'l{l}_inst1_{inst1}-inst2_{inst2}-offset_lvl_{offset_lvl}-rvb_lvl_{rvb_lvl}-bpm_{bpm}-h1-{suffix}.npy'
            h0 = np.load(os.path.join(groupdir,ds,m,h0_file))
            h1 = np.load(os.path.join(groupdir,ds,m,h1_file))
            dgm = [h0,h1]
            barcode_file = f'l{l}_inst1_{inst1}-inst2_{inst2}-offset_lvl_{offset_lvl}-rvb_lvl_{rvb_lvl}-bpm_{bpm}-barcode-{suffix}.png'
            diag_file = f'l{l}_inst1_{inst1}-inst2_{inst2}-offset_lvl_{offset_lvl}-rvb_lvl_{rvb_lvl}-bpm_{bpm}-diag-{suffix}.png'

            barcode_path = os.path.join(graphdir,ds,m,barcode_file)
            diag_path = os.path.join(graphdir,ds,m,diag_file)
    
            gudhi.plot_persistence_diagram(dgm)

            plt.savefig(diag_path)
            plt.close()

            gudhi.plot_persistence_barcode(dgm)
            
            plt.savefig(barcode_path)
            plt.close()


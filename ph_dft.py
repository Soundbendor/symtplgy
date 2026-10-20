import numpy as np
import gudhi
from scipy.spatial.distance import pdist, squareform
from ripser import ripser

birth_sorter = lambda x: x[np.argsort(x[:,0])][::-1]

def make_bar_rep(bar_pairs,t = 8, p=24):
    # shape should be (p,t)
    accum = np.zeros((p,t)).astype(np.complex128)
    for (k,l) in bar_pairs:
        t_arr = np.repeat(np.expand_dims(k*np.arange(t)/t,axis=0), p, axis=0)
        p_arr = np.repeat(np.expand_dims(l*np.arange(p)/p,axis=1), t, axis=1)
        cur_bar = np.exp(-2j*np.pi*(t_arr + p_arr))
        accum += cur_bar
    return accum

def get_bar_dist(bar1, bar2):
    return np.sum(np.abs(bar1-bar2))


def vectorize_bars(barlist):
    num_bars = len(barlist)
    flattened_dim = barlist[0].shape[0] * barlist[0].shape[1]
    return np.array(barlist).reshape(num_bars, flattened_dim)


def get_pairwise_distances(barlist, norm=True):
    ret= pdist(vectorize_bars(barlist), metric=get_bar_dist)
    if norm == True:
        return ret/ret.max()
    else:
        return ret

def calc_rips(_bars, t = 8, p = 24, maxdim=1):
    b_arr = [make_bar_rep(x,t=t, p=p) for x in _bars]
    dists = squareform(get_pairwise_distances(b_arr, norm=True)*100.)
    ret = ripser(dists,maxdim=maxdim,distance_matrix = True)
    return ret

# to do
def graph_rips(_bars, t = -8, p = 24, maxdim = 2):
    b_arr = [make_bar_rep(x,t=t, p=p) for x in _bars]
    dists = squareform(get_pairwise_distances(b_arr, norm=True)*100.)
    thresh = (np.arange(10) + 1)*100.
    for thr in thresh:
        cplx = gudhi.SimplexTree.create_from_array(dists, max_filtration=thr)
        cplx.expansion(maxdim)

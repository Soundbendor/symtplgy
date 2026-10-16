import torch, torch.utils.data as TUD
import os, sys, time, argparse, copy
from torch import nn
import numpy as np  

import util.util_main as UMN
import util.util_metrics as UME
import util.util_constants as UC
import util.util_data as UD
import util.util_wandb as UW
import util.util_optuna as UO
import util.util_probing as UP
import util.util_rdb as UR
import util.util_extractor as UEX

from probe_dataset import ProbeDataset

from functools import partial
from distutils.util import strtobool


def set_seed(seed):
    torch.manual_seed(seed)

def calculate_mean_stdev(generator, train_subset, train_size, emb_dim, shuffle = True, device='cpu'):
    train_dl = TUD.DataLoader(train_subset, batch_size = train_size, shuffle=shuffle, generator=generator)
    
    _mean = None
    _std = None
    _max = None
    _min = None
    with torch.no_grad():
        for batch_idx, data in enumerate(train_dl):
            _ipt, ground_truth = data

        
            if _ipt.shape[0] != train_size:
                print(f'did not load entire split of size {train_size}')
                break
            _mean = _ipt.mean(axis=0)
            _std = _ipt.std(axis=0)
            _max = _ipt.max(axis=0).values
            _min = _ipt.min(axis=0).values

            if _mean.shape[0] != emb_dim or _std.shape[0] != emb_dim or _max.shape[0] != emb_dim or _min.shape[0] != emb_dim:
                _mean = None
                _std = None
                _max = None
                _min = None
                print(f'did not match emb_dim of size {emb_dim}')
    return _mean, _std, _max, _min

if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
    parser.add_argument("-ms", "--model_size", type=str, default="musicgen-small", help="musicgen-small/musicgen-medium/musicgen-large/jukebox/MERT-v1-95M/MERT-v1-330M/wav2vec2-base/wav2vec2-large")
    parser.add_argument("-mpl", "--meanpool", type=strtobool, default=False, help="embeddings meanpooled over seq len (override for AR models)")
    parser.add_argument("-fsq", "--full_seq", type=strtobool, default=True, help="use full seq embeddings (override for both AR and Masked)")
    parser.add_argument("-dbg", "--debug", type=strtobool, default=False, help="debug")
    parser.add_argument("-cd", "--use_cuda", type=strtobool, default=True, help="use cuda")
    parser.add_argument("-sh", "--from_share", type=strtobool, default=True, help="load from share partition")
    parser.add_argument("-sj", "--slurm_job", type=int, default=0, help="slurm job")
    parser.add_argument("-sf", "--suffix", type=int, default=1, help="suffix")

    args = parser.parse_args()

    #### some initialization
    device = 'cpu'
    if args.use_cuda == True and torch.cuda.is_available() == True:
        device = 'cuda'
        torch.cuda.empty_cache()
        torch.set_default_device(device)
    from_dir = ""
    if args.from_share == True:
        from_dir = os.path.join(UC.SHARE_PATH, 'syntheory_plus_expr')
    datadict = UD.load_data_dict(args.dataset)

    meanpool, last_token =  UEX.parse_seqtype_overrides(model_size, meanpool_override = args.meanpool, full_seq_override = args.full_seq)
    cur_ds = ProbeDataset(datadict, args.model_size, layer_idx=0, from_dir = from_dir, meanpool = meanpool, last_token = last_token, to_torch = True, device = device)

    configdict = UP.build_config(args, datadict)
    subsetdict = UP.get_preq_valid_test_subsets(cur_ds, datadict, configdict)
    wandbdict = UW.build_initdict(args, configdict)

    if args.debug == True:
        pass
    elif args.stats == True:
        train_subset = subsetdict['preq_all_subset']
        train_size = subsetdict['preq_all_size']
        for layer_idx in range(configdict['model_num_layers']): 
            torch_gen = torch.Generator(device=device)
            torch_gen.manual_seed(configdict['seed'])
            train_subset.dataset.set_layer_idx(layer_idx)
            cur_mean, cur_std, cur_max, cur_min = calculate_mean_stdev(torch_gen, train_subset, train_size, configdict['model_dim'] , shuffle = True, device=device)
            print(layer_idx, cur_mean, cur_std)
            UP.save_mean(cur_mean, configdict, layer_idx)
            UP.save_std(cur_std, configdict, layer_idx)
            UP.save_max(cur_max, configdict, layer_idx)
            UP.save_min(cur_min, configdict, layer_idx)

   

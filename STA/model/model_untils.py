from torch.optim.lr_scheduler import LambdaLR
from functools import partial
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def my_collate(batch):
    r"""Custom collate function for dealing with custom types."""

    # Return both collated tensors and custom types
    return None, None, batch[0]


class FullBatchDataset(Dataset):
    def __init__(self, graph):
        self.graph = graph

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.graph


def generate_dataloader(adata, batch_size=64, train_shuffle=True, random_seed=42):
    import dgl
    from scipy.sparse import coo_matrix
    # random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    coo = coo_matrix(adata.uns['adj'])
    row = torch.from_numpy(coo.row).to(torch.long)
    col = torch.from_numpy(coo.col).to(torch.long)
    dgl_graph = dgl.graph(tuple([row, col]))
    # feature
    dgl_graph.ndata['feature'] = torch.tensor(adata.X.todense(), dtype=torch.float32)
    dgl_graph.ndata['batch'] = torch.tensor(adata.obs.loc[:, 'batch'].cat.codes.to_numpy(), dtype=torch.int64)
    dgl_graph.ndata['index'] = torch.tensor(np.arange(0, len(adata)), dtype=torch.int64)
    spot_quality = adata.obs.loc[:, 'spot_quality'].to_numpy()
    spot_quality = [True if item == 'real' else False for item in spot_quality]
    dgl_graph.ndata['spot_quality'] = torch.tensor(spot_quality, dtype=torch.bool)

    if batch_size < len(adata):
        # mini batch, dataset & dataloader
        sampler_list = [5, 5]  # TODO
        sampler = dgl.dataloading.ShaDowKHopSampler(sampler_list)
        train_nids = dgl_graph.ndata['index']
        # dataloader
        train_loader = dgl.dataloading.DataLoader(dgl_graph,
                                                  train_nids,
                                                  sampler,
                                                  batch_size=batch_size,
                                                  shuffle=train_shuffle,
                                                  drop_last=False)
        test_loader = dgl.dataloading.DataLoader(dgl_graph,
                                                 train_nids,
                                                 sampler,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 drop_last=False)
        mode = 'mini_batch'
    elif batch_size >= len(adata):
        # full batch
        st_dataset = FullBatchDataset(dgl_graph)
        train_loader = DataLoader(
            st_dataset,
            batch_size=1,
            collate_fn=my_collate,
            drop_last=False,
            shuffle=False,
            num_workers=0
        )
        test_loader = DataLoader(
            st_dataset,
            batch_size=1,
            collate_fn=my_collate,
            drop_last=False,
            shuffle=False,
            num_workers=0
        )
        mode = 'full_batch'
    else:
        raise NotImplementedError
    return train_loader, test_loader, dgl_graph, mode


def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

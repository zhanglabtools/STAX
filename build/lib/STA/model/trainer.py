#!/usr/bin/env python
import torch
import numpy as np
import os
import random

from .model import Model
from .model_untils import generate_dataloader


def STAX(
        adata=None,
        h_dim=16,
        multiple=8,
        num_heads=2,
        n_epoch=500,
        batch_size=1024,
        train_shuffle=True,
        lr=1e-3,
        alpha=1,
        beta=1,
        evaluate=False,
        impute=True,
        verbose=False,
        gpu=0,
        random_seed=42,
        output_dir='output/',
):
    """
    :param adata:
    :param h_dim:
    :param multiple:
    :param num_heads:
    :param n_epoch:
    :param batch_size:
    :param train_shuffle:
    :param lr:
    :param alpha:
    :param beta:
    :param evaluate:
    :param impute:
    :param verbose:
    :param gpu:
    :param random_seed:
    :param output_dir:
    :return: adata
    """

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)

    assert n_epoch >= 1
    assert batch_size >= 1

    if adata is None:
        print('adata is None!')
        raise TypeError
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.set_device(gpu)
    else:
        device = 'cpu'

    os.makedirs(output_dir + 'checkpoint', exist_ok=True)

    train_loader, test_loader, dgl_graph, mode = generate_dataloader(adata, batch_size, train_shuffle, random_seed)
    if 'image_embedding' in adata.obsm.keys():
        x_dim, n_domain = adata.shape[1], len(adata.obs['batch'].cat.categories)
    else:
        x_dim, n_domain = adata.shape[1], len(adata.obs['batch'].cat.categories)

    # model config
    model = Model(x_dim, h_dim, multiple=multiple, num_heads=num_heads, n_domain=n_domain)

    # train
    model.fit(
        adata,
        train_loader,
        n_epoch=n_epoch,
        lr=lr,
        mode=mode,
        dgl_graph=dgl_graph,
        alpha=alpha,
        beta=beta,
        device=device,
        verbose=verbose,
        random_seed=random_seed,
    )

    # store cell embedding, gene embedding and imputed expression
    adata.obsm['X_STAX'] = model.encode_batch(test_loader, out='latent', device=device, evaluate=evaluate,
                                              size=len(adata))
    adata.varm['gene_embedding'] = \
        (model.decoder.conv1.fc.weight.data.T.view(-1, num_heads, h_dim * multiple).mean(
            1) @ model.decoder.conv2.fc.weight.data.T).T.cpu().detach().numpy()
    if impute:
        adata.layers['impute'] = model.encode_batch(test_loader, out='impute', device=device, evaluate=evaluate,
                                                    size=len(adata))
    # save model
    model.to('cpu')
    torch.save(model.state_dict(), output_dir + '/checkpoint/model.pt')
    torch.cuda.empty_cache()
    del model
    # TODO can not store scipy sparse matrix as h5ad
    # del adata.uns['adj']

    return adata

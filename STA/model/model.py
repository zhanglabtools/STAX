#!/usr/bin/env python
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from .layer import Encoder, Decoder
from .loss import *
from .model_untils import get_linear_schedule_with_warmup


class Model(nn.Module):
    """
    model framework
    """

    def __init__(self, input_dim, h_dim, multiple=8, num_heads=2, n_domain=1):
        """
        :param input_dim:
        :param h_dim:
        :param multiple:
        :param n_domain:
        """
        super().__init__()

        self.encoder = Encoder(input_dim, h_dim, multiple=multiple, num_heads=num_heads)
        self.decoder = Decoder(h_dim, input_dim, multiple=multiple, n_domain=n_domain, num_heads=num_heads)
        self.n_domain = n_domain
        self.input_dim = input_dim
        self.h_dim = h_dim

    def load_model(self, path):
        """
        :param path:
        :return:
        """
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def encode_batch(
            self,
            dataloader,
            device='cuda',
            out='latent',
            size=None,
            evaluate=False
    ):
        """

        :param dataloader:
        :param device:
        :param out:
        :param size:
        :param evaluate:
        :return:
        """
        self.to(device)
        if evaluate:
            self.eval()
            print('evaluate mode')
        else:
            self.train()

        if out == 'latent':
            output = np.zeros((size, self.h_dim))
            for i, (input_nodes, output_nodes, block) in enumerate(dataloader):
                block = block.to(device)
                x, y, idx = block.ndata['feature'], block.ndata['batch'], block.ndata['index']
                z = self.encoder(x, block)[1]  # z, mu, var
                if len(dataloader) == 1:  # full batch
                    output[idx.detach().cpu().numpy()] = z.detach().cpu().numpy()
                else:
                    condition = torch.isin(idx, output_nodes.to(device))
                    output[output_nodes.detach().cpu().numpy()] = z[condition].detach().cpu().numpy()
        elif out == 'impute':
            output = np.zeros((size, self.input_dim))
            for i, (input_nodes, output_nodes, block) in enumerate(dataloader):
                block = block.to(device)
                x, y, idx = block.ndata['feature'], block.ndata['batch'], block.ndata['index']
                z = self.encoder(x, block)[0]  # z, mu, var
                recon_x, recon_x_scale = self.decoder(z, y, block)
                recon_x[recon_x < 0] = 0
                if len(dataloader) == 1:  # full batch
                    output[idx.detach().cpu().numpy()] = recon_x.detach().cpu().numpy()
                else:
                    condition = torch.isin(idx, output_nodes.to(device))
                    output[output_nodes.detach().cpu().numpy()] = recon_x[condition].detach().cpu().numpy()
        else:
            raise NotImplementedError

        return output

    def fit(
            self,
            adata,
            dataloader,
            lr=1e-3,
            n_epoch=500,
            device='cuda',
            verbose=False,
            mode='full',
            dgl_graph=None,
            alpha=1,
            beta=1,
            random_seed=42,
    ):
        """

        :param adata:
        :param dataloader:
        :param lr:
        :param n_epoch:
        :param device:
        :param verbose:
        :param mode:
        :param dgl_graph:
        :param alpha:
        :param beta:
        :param random_seed:
        :return:
        """
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        self.to(device)
        if mode == 'full_batch':
            dgl_graph = dgl_graph.to(device)
        optim = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': lr, 'weight_decay': 5e-4},
            {'params': self.decoder.conv1.parameters(), 'lr': lr, 'weight_decay': 5e-4},
            {'params': self.decoder.conv2.parameters(), 'lr': lr, 'weight_decay': 5e-4},
            {'params': self.decoder.norm1.parameters(), 'lr': lr * 10, 'weight_decay': 5e-4},
        ])
        scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=int(n_epoch * 0.1),
                                                    num_training_steps=n_epoch)
        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq:
            for epoch in tq:
                if mode == 'mini_batch':
                    tk0 = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Iterations',
                               disable=(not verbose))
                    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=int(len(dataloader) * 0.1),
                                                                num_training_steps=len(dataloader))
                    epoch_loss = defaultdict(float)
                    for i, (input_nodes, output_nodes, block) in tk0:
                        block = block.to(device)
                        x, y = block.ndata['feature'], block.ndata['batch']
                        z, mean, var = self.encoder(x, block)
                        recon_x, recon_x_scale = self.decoder(z, y, block)
                        # loss
                        spot_quality_mask = block.ndata['spot_quality']
                        recon_loss_mse = F.mse_loss(recon_x[spot_quality_mask], x[spot_quality_mask]) * x.size(-1) / 1
                        kl_loss = kl_div(mean, var)
                        loss = {'recon_loss_mse': alpha * recon_loss_mse, 'kl_loss': beta * kl_loss}

                        optim.zero_grad()
                        sum(loss.values()).backward()
                        torch.nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=10, norm_type=2)
                        optim.step()

                        for k, v in loss.items():
                            epoch_loss[k] += loss[k].item()
                        scheduler.step()
                        epoch_info = ', '.join(['{}={:.4f}'.format(k, v / (i + 1)) for k, v in epoch_loss.items()])
                        epoch_info += f', lr: {round(optim.param_groups[0]["lr"], 8)}'
                        tq.set_postfix_str(epoch_info)
                else:
                    epoch_loss = defaultdict(float)
                    x, y = dgl_graph.ndata['feature'], dgl_graph.ndata['batch']
                    z, mean, var = self.encoder(x, dgl_graph)
                    recon_x, recon_x_scale = self.decoder(z, y, dgl_graph)
                    spot_quality_mask = dgl_graph.ndata['spot_quality']
                    recon_loss_mse = F.mse_loss(recon_x[spot_quality_mask], x[spot_quality_mask]) * x.size(-1) / 1
                    # loss
                    kl_loss = kl_div(mean, var)
                    loss = {'recon_loss_mse': alpha * recon_loss_mse, 'kl_loss': beta * kl_loss}

                    optim.zero_grad()
                    sum(loss.values()).backward()
                    torch.nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=10, norm_type=2)
                    optim.step()

                    for k, v in loss.items():
                        epoch_loss[k] += loss[k].item()
                    scheduler.step()
                    epoch_info = ', '.join(['{}={:.4f}'.format(k, v) for k, v in epoch_loss.items()])
                    epoch_info += f', lr: {round(optim.param_groups[0]["lr"], 8)}'
                    tq.set_postfix_str(epoch_info)

        return adata

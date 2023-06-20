import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from itertools import zip_longest
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm, trange
from models.VAE import VAE
from sklearn.metrics import silhouette_score, davies_bouldin_score

# training
def training(name, max_patience, num_epochs, model, optimizer, training_loader, val_loader):
    nll_val = []
    best_nll = 1000.
    patience = 0

    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        for _, batch in enumerate(training_loader):
            if hasattr(model, 'dequantization'):
                if model.dequantization:
                    batch = batch + torch.rand(batch.shape)
            loss = model.forward(batch)


            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Validation
        loss_val = evaluation(val_loader, model_best=model, epoch=e)
        nll_val.append(loss_val)  # save for plotting

        if e == 0:
            print('saved!')
            torch.save(model, name + '.model')
            best_nll = loss_val
        else:
            if loss_val < best_nll:
                print('saved!')
                torch.save(model, name + '.model')
                best_nll = loss_val
                patience = 0

                samples_generated(name, val_loader, extra_name="_epoch_" + str(e))
            else:
                patience = patience + 1

        if patience > max_patience:
            break

    nll_val = np.asarray(nll_val)

    return nll_val





def evaluation(test_loader, name=None, model_best=None, epoch=None):
    # EVALUATION
    if model_best is None:
        # load best performing model
        model_best = torch.load(name + '.model')

    model_best.eval()
    loss = 0.
    N = 0.
    for _, test_batch in enumerate(test_loader):
        loss_t = model_best.forward(test_batch, reduction='sum')
        loss = loss + loss_t.item()
        N = N + test_batch.shape[0]
    loss = loss / N

    if epoch is None:
        print(f'FINAL LOSS: nll={loss}')
    else:
        print(f'Epoch: {epoch}, val nll={loss}')
    return loss



def samples_generated(name, data_loader, extra_name=''):
    x = next(iter(data_loader)).detach().numpy()

    # GENERATIONS-------
    model_best = torch.load(name + '.model')
    model_best.eval()

    num_x = 2
    num_y = 2
    x = model_best.sample(num_x * num_y)
    x = x.detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (32, 32, 3))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name + '_generated_images' + extra_name + '.pdf', bbox_inches='tight')
    plt.close()

def samples_real(name, test_loader):
    # REAL-------
    num_x = 2
    num_y = 2
    x = next(iter(test_loader)).detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (32, 32, 3))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name+'_real_images.pdf', bbox_inches='tight')
    plt.close()


def plot_curve(name, nll_val):
    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('NLL')
    plt.savefig(name + '_nll_val_curve.pdf', bbox_inches='tight')
    plt.close()



class VAEAnalyzer:
    """Class for analysing an autoencoder model."""

    def __init__(
        self,
        model: VAE,
        dataset: Dataset,
        n_samplings: int = 1,
    ):
        """
        :param model: trained autoencoder model
        :param dataset: test dataset
        :param n_samplings: number of samplings performed for analysis, defaults to 1
        """
        self.model = model
        self.dataset = dataset

        self.n_samplings = n_samplings

        self._inputs: Optional[torch.Tensor] = None
        self._latents: Optional[torch.Tensor] = None
        self._reconstructions: Optional[torch.Tensor] = None
        self._labels: Optional[torch.Tensor] = None

        self._plot_indices: Optional[np.array] = None

        self._class_indices: Dict[int, np.array] = {}
        self._averages: Optional[torch.Tensor] = None

        self._retrieve_reconstructions()

    def _retrieve_reconstructions(self):
        """Get data for analysis."""
        loader = DataLoader(self.dataset, batch_size=64, shuffle=False, drop_last=True)

        inps = []
        lats = []
        recs = []
        lbls = []

        for inputs, labels in loader:
            reconstructions = []
            latents = []
            for _ in range(self.n_samplings):
                latents.append(self.model.sample(inputs, is_decoder=False).detach())
                reconstructions.append(self.model.sample(inputs).detach())
            inps.append(inputs)
            lats.append(torch.stack(latents, dim=1))
            recs.append(torch.stack(reconstructions, dim=1))
            lbls.append(labels)

        self._inputs = torch.cat(inps, dim=0).view(-1, 32, 32, 3)
        self._latents = torch.cat(lats, dim=0).view(
            -1, self.n_samplings, self.model.L
        )
        self._reconstructions = torch.cat(recs, dim=0).view(
            -1, self.n_samplings, 32, 32, 3
        )
        self._labels = torch.cat(lbls, dim=0).view(-1)
        self._plot_indices = np.random.permutation(len(self._inputs))

    def get_metrics(self):
        x = self._reconstructions.reshape(self._reconstructions.shape[0], -1)
        silh = silhouette_score(X=x, labels=self._labels)
        dav = davies_bouldin_score(X=x, labels=self._labels)
        print(f"Silhouette score: {silh}")
        print(f"Davies Bouldin Index: {dav}")
        return silh, dav
        


def visualize_samples(
    images: torch.Tensor,
    title: str,
    labels: Optional[Union[np.ndarray, List]] = None,
    other_images: Optional[torch.Tensor] = None,
    n_cols: int = 5,
):
    """Visualize images with their labels."""
    n_rows = len(images) // n_cols

    figsize = (n_cols, n_rows)
    if other_images is not None:
        figsize = (1 + other_images.shape[1]) * figsize[0], figsize[1]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    axes = np.array(axes)

    if labels is None:
        labels = []
    if other_images is None:
        other_images = []

    for idx, (image, other, label) in enumerate(
        zip_longest(images, other_images, labels)
    ):
        x = idx % n_cols
        y = idx // n_cols
        ax = axes[y, x]
        if other is not None:
            for o in other:
                image = torch.cat((image, o), 1)
        ax.imshow(image.numpy(), cmap="gray")
        ax.text(0, 5, label, fontsize=12, weight="bold", c="w")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, y=1)
    return fig
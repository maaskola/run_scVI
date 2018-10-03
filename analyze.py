#!/usr/bin/env python

# %matplotlib inline
import os
import os.path as osp
import argparse

# do this before importing pylab or pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
import pandas as pd


# from scvi.dataset import CortexDataset, RetinaDataset
# from scvi.dataset import LoomDataset, CsvDataset, Dataset10X, AnnDataset
# from scvi.dataset import CsvDataset #, MouseOBDataset
from mycsv import CsvDataset, MouseOBDataset
import scvi.dataset as ds

from scvi.models import *
from scvi.inference import UnsupervisedTrainer

pp = None

# from sciv/dataset/csv.py
# class VAE(nn.Module):
#     r"""Variational auto-encoder model.
# 
#     :param n_input: Number of input genes
#     :param n_batch: Number of batches
#     :param n_labels: Number of labels
#     :param n_hidden: Number of nodes per hidden layer
#     :param n_latent: Dimensionality of the latent space
#     :param n_layers: Number of hidden layers used for encoder and decoder NNs
#     :param dropout_rate: Dropout rate for neural networks
#     :param dispersion: One of the following
# 
#         * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
#         * ``'gene-batch'`` - dispersion can differ between different batches
#         * ``'gene-label'`` - dispersion can differ between different labels
#         * ``'gene-cell'`` - dispersion can differ for every gene in every cell
# 
#     :param log_variational: Log variational distribution
#     :param reconstruction_loss:  One of
# 
#         * ``'nb'`` - Negative binomial distribution
#         * ``'zinb'`` - Zero-inflated negative binomial distribution
# 
#     Examples:
#         >>> gene_dataset = CortexDataset()
#         >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
#         ... n_labels=gene_dataset.n_labels, use_cuda=True )
# 
#     """


def analyze(dataset,
        save_prefix,
        use_cuda=False,
        use_batches=False,
        n_epochs=100,
        lr=1e-3,
        n_samples_tsne=200,
        n_latent=10,
        dispersion="gene"):
    print(f"dataset.n_labels = {dataset.n_labels}")
    print(f"dataset.n_batches = {dataset.n_batches}")
    vae = VAE(dataset.nb_genes,
            dispersion=dispersion,
            n_batch=dataset.n_batches * use_batches,
            n_latent=n_latent,
            n_labels=dataset.n_labels)
    model = UnsupervisedTrainer(vae,
            dataset,
            train_size=0.75,
            use_cuda=use_cuda,
            frequency=5)
    model.train(n_epochs=n_epochs, lr=lr)

    ll_train_set = model.history["ll_train_set"]
    ll_test_set = model.history["ll_test_set"]
    x = np.linspace(0, n_epochs, (len(ll_train_set)))

    # plt.figure(figsize=(args.plotsize * args.num_clusters, args.plotsize * num_samples))
    plt.plot(x, ll_train_set)
    plt.plot(x, ll_test_set)
    # plt.ylim(1150,1600)
    # plt.show()

    # model.train_set.show_t_sne(n_samples=n_samples_tsne, color_by='labels', save_name=save_prefix + "/tsne.pdf")

    print("done")
    
    return model

def run_scvi(paths,
        save_prefix,
        use_cuda=False,
        use_batches=False,
        n_epochs=100,
        lr=1e-3,
        n_samples_tsne=200,
        new_n_genes=600,
        n_latent=10,
        do_individual=False,
        dispersion="gene"):
    if new_n_genes == 0:
        new_n_genes = 1000000

    idx = 0
    if do_individual:
        pp = PdfPages(save_prefix + '/scvi-individual.pdf')
    datasets = []
    models = {}
    # for idx in range(1, 13):
    for idx in range(1, 4):
    # for idx in range(8, 13):
    # for path in paths:
        save_path = save_prefix + f"/count_{idx:03d}.tsv.gz"
        print(save_path)
        # dataset = CsvDataset(path, save_path=save_path, compression='gzip')
        dataset = MouseOBDataset(new_n_genes=new_n_genes, index=idx)
        idx = idx + 1
        print(dataset)

        if do_individual:
            models[idx] = analyze(dataset,
                    save_prefix,
                    use_cuda=use_cuda,
                    use_batches=use_batches,
                    n_epochs=n_epochs,
                    lr=lr,
                    n_samples_tsne=n_samples_tsne,
                    n_latent=n_latent,
                    dispersion=dispersion)
            plt.savefig(pp, format="pdf")

        datasets = datasets + [dataset]

    if do_individual:
        pp.close()

    if len(datasets) > 1:
        pp = PdfPages(save_prefix + '/scvi-joint.pdf')
        # NOTE this uses the intersection of genes
        # TODO use union
        dataset = ds.GeneExpressionDataset.concat_datasets(*datasets, shared_labels=False)
        print(dataset.n_labels)
        # TODO use top
        # TODO use bottom
        # TODO use random
        models["joint"] = analyze(dataset,
                save_prefix,
                use_cuda=use_cuda,
                use_batches=use_batches,
                n_epochs=n_epochs,
                lr=lr,
                n_samples_tsne=n_samples_tsne,
                n_latent=n_latent,
                dispersion=dispersion)
        plt.savefig(pp, format="pdf")
        pp.close()

    # TODO save results
    for key in models.keys():
        model = models[key]

        latent, batch_indices, labels = model.train_set.get_latent(sample=True)
        # print(latent)
        # print(batch_indices)
        # print(labels)
        indices = model.train_set.indices
        print(indices)
        # print(dataset.labels)
        # print(dataset.labels[indices])
        # print(dataset.X.index)
        # print(dataset.X.columns)
        # print(dataset.spot_names)
        df = pd.DataFrame(latent, index=labels) #, index=
        df.to_csv(osp.join(save_prefix, f"scVI-{key}-hidden.tsv.gz"), sep="\t", compression="gzip")

        model.train_set.show_t_sne(n_samples=n_samples_tsne,
                # color_by='batches',
                color_by='labels',
                save_name=save_prefix + f"/scVI-{key}-tSNE.pdf")



def main():
    """ Parses the command line and runs scVI
    """
    parser = argparse.ArgumentParser(description='Miniature std')
    parser.add_argument('paths', metavar='count-matrix', type=str, nargs='*',
        help='the path to a count matrix with genes in the rows and spots in the columns')
    parser.add_argument('-t', '--types', dest='n_latent', type=int, default=10)
    parser.add_argument('-n', '--n_epochs', dest='n_epochs', type=int, default=400)
    # TODO
    # parser.add_argument('--transpose', action='store_true', help="tranpose input count matrices")
    parser.add_argument('--individual', action='store_true', dest='do_individual', help="also perform individual analyses")
    parser.add_argument('--cuda', action='store_true', dest="use_cuda", help="tranpose input count matrices")
    # TODO maybe invert switch logic
    parser.add_argument('--batches', action='store_true', dest="use_batches", help="allow batches as covariates")
    parser.add_argument('--learning-rate', dest='lr', type=float, default=1e-3)
    # parser.add_argument('--restore', type=str)
    parser.add_argument('-s', '--save_prefix', type=str, metavar="PATH", default="./data")
    parser.add_argument('-d', '--dispersion', type=str, metavar="STRING", help="one of gene, gene-batch, gene-label, gene-cell", default="gene")
    # TODO
    # parser.add_argument('-d', '--design', type=str, default="")
    # TODO fix default values
    parser.add_argument('--top', dest ='new_n_genes', metavar='N', type=int, default=0,
        help='use only the top N expressed genes, 0 for all genes (default: 0)')
    # TODO fix default values
    parser.add_argument('--tsne', dest ='n_samples_tsne', metavar='N', type=int, default=200,
        help='how many spots to visualize using tSNE (default: 200)')
    args = vars(parser.parse_args())
    print(args)
    run_scvi(**args)

if __name__ == '__main__':
    main()

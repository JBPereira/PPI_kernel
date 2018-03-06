import pandas as pd
from sklearn.manifold import TSNE, MDS, SpectralEmbedding
import matplotlib.pyplot as plt

from Pacific_analysis.PPI import ppi_kernel

'''
Represent the data using Manifold Representations
'''


def manifold_rep(X, y, ppi):

    plaque_class_c = pd.cut(y, 10, labels = False)

    ppi_kernel_ = ppi_kernel(ppi=ppi, gamma_n=4, gamma_alpha=10)

    kernel, _ = ppi_kernel_.compute_interaction_kernel(data=X, finishing_time=False, norm_kernel=True)

    TSNE_coords = TSNE().fit_transform(X)
    MDS_coords = MDS().fit_transform(X)
    SE_coords = SpectralEmbedding().fit_transform(X)

    TSNE_ppi_coords = TSNE(metric='precomputed').fit_transform(kernel)
    MDS_ppi_coords = MDS(dissimilarity='precomputed').fit_transform(kernel)


    ax = plt.figure(figsize=(20, 20))

    plt.subplot(221)

    plt.scatter(TSNE_coords[:, 0], TSNE_coords[:, 1], c=plaque_class_c, alpha=0.6)
    plt.title('TSNE')
    plt.colorbar()

    plt.subplot(222)

    plt.scatter(MDS_coords[:, 0], MDS_coords[:, 1], c=plaque_class_c, alpha=0.6)
    plt.title('MDS')
    plt.colorbar()

    plt.subplot(223)

    plt.scatter(SE_coords[:,0], SE_coords[:, 1], c= plaque_class_c, alpha = 0.6)
    plt.title('SpectralEmbedding')
    plt.colorbar()

    plt.subplot(224)

    plt.scatter(TSNE_ppi_coords[:,0], TSNE_ppi_coords[:, 1], c= plaque_class_c, alpha=0.6)
    plt.title('TSNE_ppi_kernel')
    plt.colorbar()

    plt.show()
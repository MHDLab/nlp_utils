from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE


def calc_tsne(texts):

    texts_bigram_sklearn = [' '.join(doc) for doc in texts]

    # maxx_features = 2**12
    vectorizer = TfidfVectorizer(max_features=None)
    X = vectorizer.fit_transform(texts_bigram_sklearn)

    tsne = TSNE(verbose = 0, random_state = 42)
    X_tsne = tsne.fit_transform(X)

    tsne_x = X_tsne[:,0].astype(float)
    tsne_y = X_tsne[:,1].astype(float)

    return tsne_x, tsne_y




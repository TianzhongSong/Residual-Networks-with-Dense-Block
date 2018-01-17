import gzip
import idx2numpy


def load_data():
    """Loads the Fashion MNIST dataset.
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    with gzip.open('data/fashion_mnist/train-images-idx3-ubyte.gz', 'rb') as f:
        x_train = idx2numpy.convert_from_string(f.read())
    with gzip.open('data/fashion_mnist/train-labels-idx1-ubyte.gz', 'rb') as f:
        y_train = idx2numpy.convert_from_string(f.read())
    with gzip.open('data/fashion_mnist/t10k-images-idx3-ubyte.gz', 'rb') as f:
        x_test = idx2numpy.convert_from_string(f.read())
    with gzip.open('data/fashion_mnist/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        y_test  = idx2numpy.convert_from_string(f.read())

    return (x_train, y_train), (x_test, y_test)
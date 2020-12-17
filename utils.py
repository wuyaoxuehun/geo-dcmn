import numpy as np


def get_metrics(out, labels):
    out = np.array(out)
    labels = np.array(labels)
    return {
         'accuracy': (out == labels).mean()
    }

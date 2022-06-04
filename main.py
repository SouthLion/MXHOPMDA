import warnings

from train import Train
from utils import plot_auc_curves, plot_prc_curves


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    fprs, tprs, auc, precisions, recalls, prc = Train(directory='data/HMDD V2.0',
                                                      num_layers=6,
                                                      hid_dim=64,
                                                      p=[0, 1, 2],
                                                      epochs=1000,
                                                      input_dropout=0.4,
                                                      layer_dropout=0.3,
                                                      out_dim=64,
                                                      lr=0.001,
                                                      wd=5e-3,
                                                      random_seed=1234,
                                                      cuda=False,
                                                      model_type='MXHOPMDA')

    plot_auc_curves(fprs, tprs, auc, directory='roc_result', name='MXHOPMDAROC')
    plot_prc_curves(precisions, recalls, prc, directory='roc_result', name='MXHOPMDAPRC')
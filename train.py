import time
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn import metrics
from utils import build_graph, weight_reset
from model import MixHop

import copy
import torch.optim as optim


def Train(directory, num_layers, hid_dim, p, epochs, input_dropout, layer_dropout, out_dim,lr, wd, random_seed, cuda, model_type):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        if not cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed(random_seed)

    if cuda:
        context = torch.device('cuda:0')
    else:
        context = torch.device('cpu')

    g, disease_vertices, mirna_vertices, ID, IM, samples = build_graph(directory, random_seed)   #有修改

    samples_df = pd.DataFrame(samples, columns=['miRNA', 'disease', 'label'])

    print('## vertices:', g.number_of_nodes())
    print('## edges:', g.number_of_edges())
    print('## disease nodes:', torch.sum(g.ndata['type'] == 1).numpy())
    print('## mirna nodes: ', torch.sum(g.ndata['type'] == 0).numpy())

    g.to(context)

    auc_result = []
    acc_result = []
    pre_result = []
    recall_result = []
    f1_result = []
    prc_result = []

    fprs = []
    tprs = []
    precisions = []
    recalls = []

    i = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    for train_idx, test_idx in kf.split(samples[:, 2]):
        i += 1
        print('Training for Fold', i)

        samples_df['train'] = 0
        samples_df['train'].iloc[train_idx] = 1

        train_tensor = torch.from_numpy(samples_df['train'].values.astype('int64'))    #将numpy转换为torch

        edge_data = {'train': train_tensor}

        g.edges[disease_vertices, mirna_vertices].data.update(edge_data)    #更新边的数据
        g.edges[mirna_vertices, disease_vertices].data.update(edge_data)    #更新边的数据

        train_eid = g.filter_edges(lambda edges: edges.data['train'])   #选出训练边的边id
        g_train = g.edge_subgraph(train_eid, preserve_nodes=True)   #将挑选出的训练边重新构成子图
        g_train.copy_from_parent()  #集成节点数据和边数据

        label_train = g_train.edata['label'].unsqueeze(1)   #第二维增加维度，变成Tensor（17373,1）
        src_train, dst_train = g_train.all_edges()  #得到子图中所有边的源节点和目标节点的id

        test_eid = g.filter_edges(lambda edges: edges.data['train'] == 0)   #挑选出测试边的id
        src_test, dst_test = g.find_edges(test_eid)    #找出测试边对应的源节点和目标节点的id
        label_test = g.edges[test_eid].data['label'].unsqueeze(1)   #挑选出所有测试边对应的边数据label，label_test {Tensor:(4344,1)}

        print('## Training edges:', len(train_eid))
        print('## Testing edges:', len(test_eid))

        if model_type == 'MXHOPMDA':
            model = MixHop(g=g_train,
                           d_sim_dim=ID.shape[1],
                           m_sim_dim=IM.shape[1],
                           disease_number=ID.shape[0],
                           mirna_number=IM.shape[0],
                           hid_dim=hid_dim,
                           out_dim=out_dim,
                           num_layers=num_layers,
                           p=p,
                           input_dropout=input_dropout,
                           layer_dropout=layer_dropout,
                           activation=torch.tanh,
                           batchnorm=True)

        model.apply(weight_reset)
        model.to(context)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        loss = nn.BCELoss()

        for epoch in range(epochs):
            start = time.time()
            model.train()
            with torch.autograd.set_detect_anomaly(True):
                score_train = model(g_train, src_train, dst_train)
                loss_train = loss(score_train, label_train)

                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                score_val = model(g, src_test, dst_test)
                loss_val = loss(score_val, label_test)

            score_train_cpu = np.squeeze(score_train.cpu().detach().numpy())
            score_val_cpu = np.squeeze(score_val.cpu().detach().numpy())
            label_train_cpu = np.squeeze(label_train.cpu().detach().numpy())
            label_val_cpu = np.squeeze(label_test.cpu().detach().numpy())

            train_auc = metrics.roc_auc_score(label_train_cpu, score_train_cpu)
            val_auc = metrics.roc_auc_score(label_val_cpu, score_val_cpu)

            pred_val = [0 if j < 0.5 else 1 for j in score_val_cpu]
            acc_val = metrics.accuracy_score(label_val_cpu, pred_val)
            pre_val = metrics.precision_score(label_val_cpu, pred_val)
            recall_val = metrics.recall_score(label_val_cpu, pred_val)
            f1_val = metrics.f1_score(label_val_cpu, pred_val)

            end = time.time()

            if (epoch + 1) % 10 == 0:
                print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.item(),
                      'Val Loss: %.4f' % loss_val.cpu().detach().numpy(),
                      'Acc: %.4f' % acc_val, 'Pre: %.4f' % pre_val, 'Recall: %.4f' % recall_val, 'F1: %.4f' % f1_val,
                      'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc, 'Time: %.2f' % (end - start))

        model.eval()
        with torch.no_grad():
            score_test = model(g, src_test, dst_test)

        score_test_cpu = np.squeeze(score_test.cpu().detach().numpy())
        label_test_cpu = np.squeeze(label_test.cpu().detach().numpy())

        fpr, tpr, thresholds = metrics.roc_curve(label_test_cpu, score_test_cpu)
        precision, recall, _ = metrics.precision_recall_curve(label_test_cpu, score_test_cpu)
        test_auc = metrics.auc(fpr, tpr)
        test_prc = metrics.auc(recall, precision)

        pred_test = [0 if j < 0.5 else 1 for j in score_test_cpu]
        acc_test = metrics.accuracy_score(label_test_cpu, pred_test)
        pre_test = metrics.precision_score(label_test_cpu, pred_test)
        recall_test = metrics.recall_score(label_test_cpu, pred_test)
        f1_test = metrics.f1_score(label_test_cpu, pred_test)

        print('Fold: ', i, 'Test acc: %.4f' % acc_test, 'Test Pre: %.4f' % pre_test,
              'Test Recall: %.4f' % recall_test, 'Test F1: %.4f' % f1_test, 'Test PRC: %.4f' % test_prc,
              'Test AUC: %.4f' % test_auc)

        auc_result.append(test_auc)
        acc_result.append(acc_test)
        pre_result.append(pre_test)
        recall_result.append(recall_test)
        f1_result.append(f1_test)
        prc_result.append(test_prc)

        fprs.append(fpr)
        tprs.append(tpr)
        precisions.append(precision)
        recalls.append(recall)

    print('## Training Finished !')
    print('-----------------------------------------------------------------------------------------------')
    print('-AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc_result), np.std(auc_result)),
          'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc_result), np.std(acc_result)),
          'Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre_result), np.std(pre_result)),
          'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall_result), np.std(recall_result)),
          'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1_result), np.std(f1_result)),
          'PRC mean: %.4f, variance: %.4f \n' % (np.mean(prc_result), np.std(prc_result)))

    return fprs, tprs, auc_result, precisions, recalls, prc_result
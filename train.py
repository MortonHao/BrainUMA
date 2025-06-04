import numpy as np
from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn as nn
from loader import BrainTemplateDataset
from model import Model
import random
from sklearn.model_selection import KFold
import torch.optim as optim
from sklearn import metrics


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def average_fcn(data, label):
    data_0 = data[label == 0]
    data_1 = data[label == 1]
    mean_0 = torch.mean(data_0, dim=0)
    mean_1 = torch.mean(data_1, dim=0)
    return mean_0, mean_1

##按超边数划分
# def get_hypergraph(average_data, k):
#     h_list = []
#     for data in average_data:
#         # ind = torch.argsort(data.abs(), dim=0, descending=True)
#         ind = torch.argsort(data, dim=0, descending=True)
#         for i in range(data.shape[1]):
#             data[:, i] = data[:, i].clone().scatter(0, ind[k:, i], 0)
#             data[:, i] = data[:, i].scatter(0, ind[:k, i], 1)
#         h_list.append(data)
#     return h_list

##按阈值划分
def get_hypergraph(average_data, threshold):
    h_list = []
    for data in average_data:
        h = torch.zeros_like(data)
        h[data > threshold] = 1.0
        h_list.append(h)
    return h_list


def train_and_test(args, path_data, path_label, temperature, k, threshold):
    setup_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # print(device)
    template_data = []
    num_nodes = []
    acc_fold = []
    auc_fold = []
    sen_fold = []
    spec_fold = []
    for path in path_data:
        data = torch.from_numpy(np.load(path)).float().to(device)
        num_nodes.append(data.size(-1))
        template_data.append(data)
    label = torch.from_numpy(np.loadtxt(path_label)).long().to(device)
    dataset = BrainTemplateDataset(template_data, label)
    kf = KFold(n_splits=args.kFold, shuffle=True, random_state=args.seed)
    for fold, (train_indices, test_indices) in enumerate(kf.split(dataset)):
        # print(f'kFold_index: {fold + 1}')
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        train_data = train_dataset.dataset.data
        train_label = train_dataset.dataset.label
        loss_fn = nn.CrossEntropyLoss()
        average_health = []
        average_disease = []
        for i in range(args.num_atlas):
            data_health, data_disease = average_fcn(train_data[i], train_label)
            average_health.append(data_health)
            average_disease.append(data_disease)
        health = get_hypergraph(average_health, threshold)
        disease = get_hypergraph(average_disease, threshold)
        H = []
        for he, di in zip(health, disease):
            h = torch.cat((he, di), dim=1)
            H.append(h)
        model = Model(args, k, num_nodes, H).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        current_fold = []
        current_fold_auc = []
        current_fold_sen = []
        current_fold_spec = []
        for epoch in range(args.epoch_cf):
            model.train()
            total_train = 0
            correct_train = 0
            loss_train = 0
            for data, label in train_loader:
                optimizer.zero_grad()
                outputs, _, loss_cont, loss_cc = model(data, label, 'cont', 'cc', temperature, None)
                _, ind = torch.max(outputs, dim=1)
                total_train += label.size(0)
                correct_train += (ind == label).sum().item()
                loss_cf = loss_fn(outputs, label)
                loss_all = loss_cf + args.lambda_cont * loss_cont + args.lambda_cc * loss_cc
                loss_train += loss_all
                loss_all.backward()
                optimizer.step()
            loss_train /= len(train_loader)
            acc = correct_train / total_train
            print('------------------------epoch:', epoch + 1, '----------------------------')
            print(f"Train acc: {acc}", 'Train loss:', loss_train.item())
            correct_acc_test = 0
            correct_auc_test = 0
            correct_sen_test = 0
            correct_spe_test = 0
            correct_pre_test = 0
            correct_f1_test = 0
            model.eval()
            with torch.no_grad():
                for data1, label1 in test_loader:
                    output_test, _, _, _ = model(data1, None, None, None, None, None)
                    _, indices = torch.max(output_test, dim=1)
                    pre = indices.cpu()
                    label_test = np.array(label1.cpu())
                    correct_acc_test += metrics.accuracy_score(label_test, pre)
                    fpr, tpr, _ = metrics.roc_curve(label_test, pre)
                    correct_auc_test += metrics.auc(fpr, tpr)
                    tn, fp, fn, tp = metrics.confusion_matrix(label_test, pre).ravel()
                    correct_sen_test += tp / (tp + fn)
                    correct_spe_test += tn / (tn + fp)
                    correct_pre_test += metrics.precision_score(label_test, pre, zero_division=1)
                    correct_f1_test += metrics.f1_score(label_test, pre)
            acc_test = correct_acc_test / len(test_loader)
            auc_test = correct_auc_test / len(test_loader)
            sen_test = correct_sen_test / len(test_loader)
            spe_test = correct_spe_test / len(test_loader)
            pre_test = correct_pre_test / len(test_loader)
            f1_test = correct_f1_test / len(test_loader)
            current_fold.append(acc_test)
            current_fold_auc.append(auc_test)
            current_fold_sen.append(sen_test)
            current_fold_spec.append(spe_test)
            print("Test acc: %.2f auc: %.2f sen: %.2f spe: %.2f pre: %.2f f1: %.2f" % (
                acc_test * 100, auc_test * 100, sen_test * 100, spe_test * 100, pre_test * 100, f1_test * 100))
            print('--------------------------------------------------------------')
        # torch.save(model.state_dict(), './result/' + str(fold + 1) + '.pt')
        acc_fold.append(current_fold)
        auc_fold.append(current_fold_auc)
        sen_fold.append(current_fold_sen)
        spec_fold.append(current_fold_spec)
    acc_fold = np.array(acc_fold)
    auc_fold = np.array(auc_fold)
    sen_fold = np.array(sen_fold)
    spec_fold = np.array(spec_fold)
    acc_fold = np.sum(acc_fold, axis=0)
    auc_fold = np.sum(auc_fold, axis=0)
    sen_fold = np.sum(sen_fold, axis=0)
    spec_fold = np.sum(spec_fold, axis=0)

    # for i in range(acc_fold.shape[0]):
    #     print(f'epoch {i + 1} acc: {acc_fold[i] / args.kFold} auc: {auc_fold[i] / args.kFold}')
    idx = np.argmax(acc_fold)
    print(
        f'epoch {idx + 1} acc: {acc_fold[idx] / args.kFold * 100} auc: {auc_fold[idx] / args.kFold * 100} sen: {sen_fold[idx] / args.kFold * 100} spec: {spec_fold[idx] / args.kFold * 100}')
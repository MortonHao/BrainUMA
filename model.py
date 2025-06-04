import torch.nn as nn
import torch.nn.functional as F
import torch


class E2E_dyncmic(nn.Module):

    def __init__(self, in_channel, out_channel, input_shape, k):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.d = input_shape[0]
        self.k = k
        self.conv1xd = Dynamic_conv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=(self.d, 1), stride=1,
                                      padding=0, dilation=1, groups=1, bias=True, K=self.k)
        self.convdx1 = Dynamic_conv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=(1, self.d), stride=1,
                                      padding=0, dilation=1, groups=1, bias=True, K=self.k)

        self.nodes = self.d

    def forward(self, A):
        A = A.view(-1, self.in_channel, self.nodes, self.nodes)

        a = self.conv1xd(A)
        b = self.convdx1(A)

        concat1 = torch.cat([a] * self.d, 2)
        concat2 = torch.cat([b] * self.d, 3)

        return concat1 + concat2


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, K=8, temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size1, self.kernel_size2 = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(
            torch.randn(K, out_planes, in_planes // groups, self.kernel_size1, self.kernel_size2), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):

        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)
        weight = self.weight.view(self.K, -1)
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size * self.out_planes,
                                                                    self.in_planes // self.groups, self.kernel_size1,
                                                                    self.kernel_size2)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output


class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature % 3 == 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios) + 1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        # print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, 1)


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(HGNN_conv, self).__init__()

        self.in_ft = in_ft
        self.out_ft = out_ft

        self.weight = nn.Parameter(torch.Tensor(self.in_ft, self.out_ft))
        nn.init.kaiming_uniform_(self.weight)

        self.V = nn.Parameter(torch.Tensor(self.in_ft, 1))
        nn.init.kaiming_uniform_(self.V)

        self.bias = nn.Parameter(torch.Tensor(self.out_ft))
        nn.init.zeros_(self.bias)

    def forward(self, x, hypergraph):
        hypergraph_t = hypergraph.t()
        edge_emb = torch.matmul(hypergraph_t, x)
        hypergraph_weight = F.sigmoid(torch.matmul(edge_emb, self.V))
        degree_v = torch.sum(torch.matmul(hypergraph, hypergraph_weight), dim=-1)
        degree_e = torch.sum(hypergraph, dim=0)
        # inv_degree_e = torch.diag(torch.pow(degree_e, -1))
        inv_degree_e = torch.diag(degree_e)
        # degree_v_2 = torch.diag_embed(torch.pow(degree_v, -0.5))
        degree_v_2 = torch.diag_embed(degree_v)  # 当出现nan时，不求指数
        weight = torch.diag_embed(hypergraph_weight.squeeze())
        adj = degree_v_2 @ hypergraph @ weight @ inv_degree_e @ hypergraph_t @ degree_v_2
        out = torch.matmul(x, self.weight)
        out = torch.matmul(adj, out)
        if self.bias is not None:
            out = out + self.bias
        return out, hypergraph_weight.squeeze()


class ClassConsistencyModule(torch.nn.Module):
    def __init__(self, num_classes):
        super(ClassConsistencyModule, self).__init__()
        self.num_classes = num_classes

    def forward(self, features, labels):
        """
        features: (batch_size, feature_dim), 输入特征
        labels: (batch_size,), 样本对应的类别标签
        """
        # 计算类别的中心（聚类中心）
        class_centers = self.compute_class_centers(features, labels)

        # 计算特征和类别中心之间的距离
        consistency_loss = self.compute_consistency_loss(features, labels, class_centers)

        return consistency_loss

    def compute_class_centers(self, features, labels):
        """
        计算每个类别的特征中心
        """
        centers = torch.zeros(self.num_classes, features.size(1)).to(features.device)
        for i in range(self.num_classes):
            class_mask = (labels == i)
            class_features = features[class_mask]
            if class_features.size(0) > 0:
                centers[i] = class_features.mean(dim=0)
        return centers

    def compute_consistency_loss(self, features, labels, class_centers):
        """
        计算类别一致性损失
        """
        loss = 0.0
        for i in range(self.num_classes):
            class_mask = (labels == i)
            class_features = features[class_mask]
            if class_features.size(0) > 0:
                # 计算每个样本到类别中心的距离
                dist = F.pairwise_distance(class_features, class_centers[i].unsqueeze(0).expand_as(class_features))
                loss += dist.mean()
        return loss

class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelfAttention, self).__init__()

        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)

        self.out = nn.Linear(output_dim, input_dim)

    def forward(self, x):

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention_scores = torch.bmm(query, key.transpose(1, 2))  # Query和Key进行点乘，得到的注意力分数
        attention_weights = F.softmax(attention_scores, dim=-1)
        weighted_values = torch.bmm(attention_weights, value)

        output = self.out(weighted_values)
        return output


class AttentionFusionModel(nn.Module):
    def __init__(self, input_dim, num_templates, output_dim):
        super(AttentionFusionModel, self).__init__()
        self.num_templates = num_templates
        self.attention = SelfAttention(input_dim, output_dim)

    def forward(self, x):

        attention_output = self.attention(x)
        fused_features = attention_output.mean(dim=1) #平均池化

        return fused_features


class Model(nn.Module):
    def __init__(self, args, k, num_nodes, H):
        super().__init__()

        self.dropout = args.dropout
        self.H = H
        self.num_atlas = args.num_atlas

        '''
            节点特征学习
        '''
        self.e2e2n = nn.ModuleList([nn.Sequential(
            E2E_dyncmic(1, args.e2e_feature1, (num_nodes[i], num_nodes[i]), k),
            nn.LeakyReLU(0.33),
            nn.Conv2d(args.e2e_feature1, args.e2e_feature2, (1, num_nodes[i])),
            nn.LeakyReLU(0.33)
        ) for i in range(self.num_atlas)])
        '''
            共享超图卷积
        '''
        self.gcn_shared = HGNN_conv(args.e2e_feature2, args.gcn_hid)
        '''
            common和specific超图卷积
        '''
        self.gcn_com = HGNN_conv(args.gcn_hid, args.gcn_out)
        self.gcn_spe = HGNN_conv(args.gcn_hid, args.gcn_out)
        '''
            类内损失
        '''
        self.Cc = ClassConsistencyModule(num_classes=2)
        '''
            readout
        '''
        self.mlp_readout_common = nn.ModuleList([nn.Linear(num_nodes[i], 1) for i in range(self.num_atlas)])
        self.mlp_readout_specific = nn.ModuleList([nn.Linear(num_nodes[i], 1) for i in range(self.num_atlas)])
        '''
            attention fusion
        '''
        self.attention = AttentionFusionModel(input_dim=args.gcn_out, num_templates=args.num_atlas, output_dim=args.gcn_out)

        self.linear_disease_classification = nn.Linear(args.gcn_out, 2)

    def forward(self, data, label, flag1, flag2, temperature, flag3):

        for index, e2e2n in enumerate(self.e2e2n):
            node = e2e2n(data[index])
            node = node.squeeze(3).transpose(1, 2)
            data[index] = F.normalize(node, p=2, dim=-1)
        x_common = []
        x_specific = []
        # x_weight = []
        for i in range(len(data)):
            x_shared, _ = self.gcn_shared(data[i], self.H[i])
            x_shared = F.relu(x_shared)
            x_shared = F.dropout(x_shared, self.dropout)
            common, _ = self.gcn_com(x_shared, self.H[i])
            # x_weight.append(weight)
            x_common.append(torch.transpose(common, 1, 2))
            specific, _ = self.gcn_spe(x_shared, self.H[i])
            x_specific.append(torch.transpose(specific, 1, 2))

        for index_common, mlp_common in enumerate(self.mlp_readout_common):
            x_common[index_common] = F.relu(mlp_common(x_common[index_common]).squeeze())
            x_common[index_common] = F.normalize(x_common[index_common], p=2, dim=-1)

        for index_specific, mlp_specific in enumerate(self.mlp_readout_specific):
            x_specific[index_specific] = F.relu(mlp_specific(x_specific[index_specific]).squeeze())
            x_specific[index_specific] = F.normalize(x_specific[index_specific], p=2, dim=-1)

        cont_loss = 0
        cc_loss = 0
        if flag1 == 'cont':
            pos_loss = 0
            neg_loss = []
            for i in range(0, self.num_atlas - 1):
                pos_loss += torch.div((x_common[i] * x_common[i + 1]).sum(dim=1), temperature)

                neg_loss.append(torch.div((x_common[i] * x_specific[i]).sum(dim=1), temperature).unsqueeze(-1))
            neg_loss.append(torch.div((x_common[self.num_atlas - 1] * x_specific[self.num_atlas - 1]).sum(dim=1),
                                      temperature).unsqueeze(-1))
            neg_loss = torch.cat(neg_loss, dim=1)
            cont_loss = (-pos_loss + torch.logsumexp(neg_loss, dim=-1)).mean()

        x = torch.stack(x_common, dim=1)
        x = self.attention(x)
        if flag2 == 'cc':
            cc_loss = self.Cc(x, label)
        x = self.linear_disease_classification(x)

        return x, None, cont_loss, cc_loss

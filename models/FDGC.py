import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# source from https://github.com/Yejin0111/ADD-GCN
class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_nodes=20):
        super(DynamicGraphConvolution, self).__init__()
        self.num_nodes = num_nodes     
        self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features*2, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

        # self-attention graph
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward_static_gcn(self, x):
        x = self.static_adj(x.transpose(1, 2))                  
        x = self.static_weight(x.transpose(1, 2))
        return x

    def forward_construct_dynamic_graph(self, x):
        m_batchsize, C, class_num = x.size()
        proj_query = x
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, 
                            keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value) 
        x_glb = self.gamma*out + x
        x = torch.cat((x_glb, x), dim=1)
        dynamic_adj = self.conv_create_co_mat(x)                  
        dynamic_adj = torch.sigmoid(dynamic_adj)                  
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x, sds=[0, 0, 1]):
        static, dynamic, static_dynamic = sds
        if static:
            out_static = self.forward_static_gcn(x)
        
        if dynamic:
            dynamic_adj = self.forward_construct_dynamic_graph(x)
        
        if static_dynamic:
            out_static = self.forward_static_gcn(x) 
            x = x + out_static  # residual
            dynamic_adj = self.forward_construct_dynamic_graph(x)
            x = self.forward_dynamic_gcn(x, dynamic_adj)          

        return x


class DropBlock2D(nn.Module):
    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            gamma = self._compute_gamma(x)
            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            mask = mask.to(x.device)
            block_mask = self._compute_block_mask(mask)
            out = x * block_mask[:, None, :, :]
            out = out * block_mask.numel() / block_mask.sum()
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)


class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, 
                            stop=stop_value, num=int(nr_steps))

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]

        self.i += 1


class FDGC(nn.Module):
    def __init__(self, input_channels, num_nodes, num_classes, patch_size, drop_prob=0.1, block_size=3):
        super(FDGC, self).__init__()
        self.input_channels = input_channels
        self.num_node = num_nodes
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.dropblock = LinearScheduler(DropBlock2D(drop_prob=drop_prob, block_size=block_size),
            start_value=0.,
            stop_value=drop_prob,
            nr_steps=5e3)

        # bone 
        self.conv1 = nn.Conv2d(self.input_channels, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2, 2))
        
        # second branch
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.bn4 = nn.BatchNorm2d(128)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # first branch
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3)
        self.bn6 = nn.BatchNorm2d(64)

        # statistic
        self.features_size = self._get_final_flattened_size()

        # third branch
        self.fc_sam = nn.Conv2d(64, self.num_node, (1,1), bias=False)
        self.conv_transform = nn.Conv2d(64, 64, (1,1))
        self.gcn = DynamicGraphConvolution(64, 64, num_nodes=self.num_node)
        
        # last
        self.fc1 = nn.Linear(self.features_size, 1024)
        self.drop1 = nn.Dropout(0.5)
        self.bn_f1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn_f2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, self.num_classes)     

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, self.input_channels, self.patch_size, self.patch_size))
            x = self.conv1(x)
            x = self.conv2(x)
            x_pool = self.pool1(x)

            x = self.conv3(x_pool)
            x = self.conv4(x)
            x =self.avgpool(x)
            _, c, w, h = x.size()

            x = self.conv5(x_pool)
            x = self.conv6(x)
            _, c2, w2, h2 = x.size()
        return c * w * h + c2 * w2 * h2 + 64*self.num_node
    
    def forward_sam(self, x):
        mask = self.fc_sam(x)
        mask = mask.view(mask.size(0), mask.size(1), -1) 
        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)
        x = self.conv_transform(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask)
        return x

    def forward(self, x):
        x = x.squeeze()

        self.dropblock.step() 
        x1 = F.leaky_relu(self.conv1(x))
        x1 = self.bn1(x1)

        x2 = F.leaky_relu(self.conv2(x1))
        x2 = self.bn2(x2)
        x_pool = self.pool1(x2)
        
        x3 = F.leaky_relu(self.conv3(x_pool))
        x3 = self.bn3(x3)
        x3 = self.dropblock(x3)
        
        x4 = F.leaky_relu(self.conv4(x3))
        x4 = self.bn4(x4)
        x4 = self.dropblock(x4)
        x_4 = self.avgpool(x4)

        x5 = F.leaky_relu(self.conv5(x_pool))
        x5 = self.bn5(x5)
        x5 = self.dropblock(x5)

        x6 = F.leaky_relu(self.conv6(x5))
        x6 = self.bn6(x6)
        x_6 = self.dropblock(x6)

        x7 = self.forward_sam(x2)
        x7 = self.gcn(x7) + x7
        x_7 = x7.view(-1, x7.size(1)*x7.size(2))

        x_6 = x_6.view(-1, x_6.size(1)*x_6.size(2)*x_6.size(3))
        x_4 = x_4.view(-1, x_4.size(1)*x_4.size(2)*x_4.size(3))
        x = torch.cat((x_4, x_6, x_7), dim=-1)

        x = F.leaky_relu(self.fc1(x))
        x = self.bn_f1(x)
        x = self.drop1(x)
 
        x = F.leaky_relu(self.fc2(x))
        x = self.bn_f2(x)
        x = self.fc3(x)
        return x


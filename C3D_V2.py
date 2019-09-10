# -*- coding: utf-8 -*-
#modified done
#2019.06.03
#haidong
#total 16 layer 

import torch
import torch.nn as nn
#import mypath as Path # pre-trained的模型的地址；



class C3D(nn.Module):
    def __init__(self,num_classes,pretrained=False):
        super(C3D,self).__init__()

        self.conv1 = nn.Conv3d(3,64,kernel_size=(3,3,3),padding=(1,1,1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2),stride=(1,2,2))

        self.conv2 = nn.Conv3d(64,128,kernel_size=(3,3,3),padding=(1,1,1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))

        self.conv3a = nn.Conv3d(128,256,kernel_size=(3,3,3),padding=(1,1,1))
        self.conv3b = nn.Conv3d(256,256,kernel_size=(3,3,3),padding=(1,1,1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))
        
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 16, 16), padding=(0, 1, 1))

        self.fc6 = nn.Linear(512,num_classes)

        self.relu = nn.ReLU()
        #self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

        
    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        

        x = x.view(x.shape[0], -1)
        print(x.shape)
        x = self.relu(self.fc6(x))
        return x



    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }
"""
        p_dict = torch.load(Path.model_dir())
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
"""
def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

    

if __name__ == "__main__": # 对自己的网络进行测试的入口；当该文件被调用的时候，不会执行该模块；当不被调用，它可以自己原地执行；进行测试；
    data = torch.autograd.Variable(torch.randn(2,3,16,224,224)) #torch.rand(N, C{in}, D{in}, H{in}, W{in}).N is batch.
    net = C3D(num_classes=101,pretrained=False)
    output = net.forward(data)



# -*- coding: utf-8 -*-

__author__ = 'kohou.wang'
__time__ = '19-9-23'
__email__ = 'oukohou@outlook.com'

# If this runs wrong, don't ask me, I don't know why;
# If this runs right, thank god, and I don't know why.
# Maybe the answer, my friend, is blowing in the wind.
# Well, I'm kidding... Always, Welcome to contact me.

"""Description for the script:
SSR-Net in Pytorch.
"""

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import time

time_list_ = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]


class SSRNet(nn.Module):
    def __init__(self, stage_num=[3, 3, 3], image_size=64,
                 class_range=101, lambda_index=1., lambda_delta=1.):
        super(SSRNet, self).__init__()
        self.image_size = image_size
        self.stage_num = stage_num
        self.lambda_index = lambda_index
        self.lambda_delta = lambda_delta
        self.class_range = class_range
        
        self.stream1_stage3 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.stream1_stage2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.stream1_stage1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.AvgPool2d(2, 2) # paper has this layer, but official codes don't.
        )
        
        self.stream2_stage3 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(2, 2)
        )
        self.stream2_stage2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(2, 2)
        )
        self.stream2_stage1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            # nn.MaxPool2d(2, 2) # paper has this layer, but official codes don't.
        )
        
        # fusion block
        self.funsion_block_stream1_stage_3_before_PB = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(8, 8)
        )
        self.funsion_block_stream1_stage_3_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10 * 4 * 4, self.stage_num[2]),
            nn.ReLU()
        )
        
        self.funsion_block_stream1_stage_2_before_PB = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(4, 4)
        )
        self.funsion_block_stream1_stage_2_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10 * 4 * 4, self.stage_num[1]),
            nn.ReLU()
        )
        
        self.funsion_block_stream1_stage_1_before_PB = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0),
            nn.ReLU(),
            # nn.AvgPool2d(2, 2) # paper has this layer, but official codes don't.
        )
        self.funsion_block_stream1_stage_1_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10 * 8 * 8, self.stage_num[0]),
            nn.ReLU()
        )
        
        # stream2
        self.funsion_block_stream2_stage_3_before_PB = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(8, 8)
        )
        self.funsion_block_stream2_stage_3_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10 * 4 * 4, self.stage_num[2]),
            nn.ReLU()
        )
        
        self.funsion_block_stream2_stage_2_before_PB = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(4, 4)
        )
        self.funsion_block_stream2_stage_2_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10 * 4 * 4, self.stage_num[1]),
            nn.ReLU()
        )
        
        self.funsion_block_stream2_stage_1_before_PB = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2) # paper has this layer, but official codes don't.
        )
        self.funsion_block_stream2_stage_1_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10 * 8 * 8, self.stage_num[0]),
            nn.ReLU()
        )
        
        self.stage3_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[0], 2 * self.stage_num[0]),
            nn.ReLU()
        )
        self.stage3_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.ReLU()
        )
        self.stage3_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh()
        )
        self.stage3_delta_k = nn.Sequential(
            nn.Linear(10 * 4 * 4, 1),
            nn.Tanh()
        )
        
        self.stage2_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[0], 2 * self.stage_num[0]),
            nn.ReLU()
        )
        self.stage2_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.ReLU()
        )
        self.stage2_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh()
        )
        self.stage2_delta_k = nn.Sequential(
            nn.Linear(10 * 4 * 4, 1),
            nn.Tanh()
        )
        
        self.stage1_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[0], 2 * self.stage_num[0]),
            nn.ReLU()
        )
        self.stage1_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.ReLU()
        )
        self.stage1_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh()
        )
        self.stage1_delta_k = nn.Sequential(
            nn.Linear(10 * 8 * 8, 1),
            nn.Tanh()
        )
        self.init_params()
    
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, image_):
        # current_time_ = time.time()
        feature_stream1_stage3 = self.stream1_stage3(image_)
        # time_list_[0] = time_list_[0] + (time.time() - current_time_)
        
        # current_time_ = time.time()
        feature_stream1_stage2 = self.stream1_stage2(feature_stream1_stage3)
        # time_list_[1] = time_list_[1] + (time.time() - current_time_)
        
        # current_time_ = time.time()
        feature_stream1_stage1 = self.stream1_stage1(feature_stream1_stage2)
        # time_list_[2] = time_list_[2] + (time.time() - current_time_)
        
        # current_time_ = time.time()
        feature_stream2_stage3 = self.stream2_stage3(image_)
        # time_list_[3] = time_list_[3] + (time.time() - current_time_)
        
        # current_time_ = time.time()
        feature_stream2_stage2 = self.stream2_stage2(feature_stream2_stage3)
        # time_list_[4] = time_list_[4] + (time.time() - current_time_)
        
        # current_time_ = time.time()
        feature_stream2_stage1 = self.stream2_stage1(feature_stream2_stage2)
        # time_list_[5] = time_list_[5] + (time.time() - current_time_)
        
        # current_time_ = time.time()
        feature_stream1_stage3_before_PB = self.funsion_block_stream1_stage_3_before_PB(feature_stream1_stage3)
        feature_stream1_stage2_before_PB = self.funsion_block_stream1_stage_2_before_PB(feature_stream1_stage2)
        feature_stream1_stage1_before_PB = self.funsion_block_stream1_stage_1_before_PB(feature_stream1_stage1)
        
        feature_stream2_stage3_before_PB = self.funsion_block_stream2_stage_3_before_PB(feature_stream2_stage3)
        feature_stream2_stage2_before_PB = self.funsion_block_stream2_stage_2_before_PB(feature_stream2_stage2)
        feature_stream2_stage1_before_PB = self.funsion_block_stream2_stage_1_before_PB(feature_stream2_stage1)
        # time_list_[6] = time_list_[6] + (time.time() - current_time_)
        
        # current_time_ = time.time()
        embedding_stream1_stage3_before_PB = feature_stream1_stage3_before_PB.view(feature_stream1_stage3_before_PB.size(0), -1)
        embedding_stream1_stage2_before_PB = feature_stream1_stage2_before_PB.view(feature_stream1_stage2_before_PB.size(0), -1)
        embedding_stream1_stage1_before_PB = feature_stream1_stage1_before_PB.view(feature_stream1_stage1_before_PB.size(0), -1)
        
        embedding_stream2_stage3_before_PB = feature_stream2_stage3_before_PB.view(feature_stream2_stage3_before_PB.size(0), -1)
        embedding_stream2_stage2_before_PB = feature_stream2_stage2_before_PB.view(feature_stream2_stage2_before_PB.size(0), -1)
        embedding_stream2_stage1_before_PB = feature_stream2_stage1_before_PB.view(feature_stream2_stage1_before_PB.size(0), -1)
        # time_list_[7] = time_list_[7] + (time.time() - current_time_)
        # import ipdb
        # ipdb.set_trace()
        # current_time_ = time.time()
        stage1_delta_k = self.stage1_delta_k(torch.mul(embedding_stream1_stage1_before_PB, embedding_stream2_stage1_before_PB))
        stage2_delta_k = self.stage2_delta_k(torch.mul(embedding_stream1_stage2_before_PB, embedding_stream2_stage2_before_PB))
        stage3_delta_k = self.stage3_delta_k(torch.mul(embedding_stream1_stage3_before_PB, embedding_stream2_stage3_before_PB))
        
        embedding_stage1_after_PB = torch.mul(self.funsion_block_stream1_stage_1_prediction_block(embedding_stream1_stage1_before_PB),
                                              self.funsion_block_stream2_stage_1_prediction_block(embedding_stream2_stage1_before_PB))
        embedding_stage2_after_PB = torch.mul(self.funsion_block_stream1_stage_2_prediction_block(embedding_stream1_stage2_before_PB),
                                              self.funsion_block_stream2_stage_2_prediction_block(embedding_stream2_stage2_before_PB))
        embedding_stage3_after_PB = torch.mul(self.funsion_block_stream1_stage_3_prediction_block(embedding_stream1_stage3_before_PB),
                                              self.funsion_block_stream2_stage_3_prediction_block(embedding_stream2_stage3_before_PB))
        
        embedding_stage1_after_PB = self.stage1_FC_after_PB(embedding_stage1_after_PB)
        embedding_stage2_after_PB = self.stage2_FC_after_PB(embedding_stage2_after_PB)
        embedding_stage3_after_PB = self.stage3_FC_after_PB(embedding_stage3_after_PB)
        # time_list_[8] = time_list_[8] + (time.time() - current_time_)
        
        # current_time_ = time.time()
        prob_stage_1 = self.stage1_prob(embedding_stage1_after_PB)
        index_offset_stage1 = self.stage1_index_offsets(embedding_stage1_after_PB)
        
        prob_stage_2 = self.stage2_prob(embedding_stage2_after_PB)
        index_offset_stage2 = self.stage2_index_offsets(embedding_stage2_after_PB)
        
        prob_stage_3 = self.stage3_prob(embedding_stage3_after_PB)
        index_offset_stage3 = self.stage3_index_offsets(embedding_stage3_after_PB)
        # time_list_[9] = time_list_[9] + (time.time() - current_time_)
        
        # current_time_ = time.time()
        stage1_regress = prob_stage_1[:, 0] * 0
        stage2_regress = prob_stage_2[:, 0] * 0
        stage3_regress = prob_stage_3[:, 0] * 0
        # import ipdb
        for index in range(self.stage_num[0]):
            stage1_regress = stage1_regress + (index + self.lambda_index * index_offset_stage1[:, index]) * prob_stage_1[:, index]
        # ipdb.set_trace()
        stage1_regress = torch.unsqueeze(stage1_regress, 1)
        stage1_regress = stage1_regress / (self.stage_num[0] * (1 + self.lambda_delta * stage1_delta_k))
        # stage1_regress = stage1_regress / (self.stage_num[0] * (1 + self.lambda_delta * torch.squeeze(stage1_delta_k, 1)))
        
        for index in range(self.stage_num[1]):
            stage2_regress = stage2_regress + (index + self.lambda_index * index_offset_stage2[:, index]) * prob_stage_2[:, index]
        stage2_regress = torch.unsqueeze(stage2_regress, 1)
        stage2_regress = stage2_regress / (self.stage_num[0] * (1 + self.lambda_delta * stage1_delta_k) *
                                           (self.stage_num[1] * (1 + self.lambda_delta * stage2_delta_k)))
        # stage2_regress = stage2_regress / ((self.stage_num[0] * (1 + self.lambda_delta * torch.squeeze(stage1_delta_k, 1))) *
        #                                    (self.stage_num[1] * (1 + self.lambda_delta * torch.squeeze(stage2_delta_k, 1))))
        
        for index in range(self.stage_num[2]):
            stage3_regress = stage3_regress + (index + self.lambda_index * index_offset_stage3[:, index]) * prob_stage_3[:, index]
        stage3_regress = torch.unsqueeze(stage3_regress, 1)
        stage3_regress = stage3_regress / (self.stage_num[0] * (1 + self.lambda_delta * stage1_delta_k) *
                                           (self.stage_num[1] * (1 + self.lambda_delta * stage2_delta_k)) *
                                           (self.stage_num[2] * (1 + self.lambda_delta * stage3_delta_k))
                                           # stage3_regress = stage3_regress / ((self.stage_num[0] * (1 + self.lambda_delta * torch.squeeze(stage1_delta_k, 1))) *
                                           #                                    (self.stage_num[1] * (1 + self.lambda_delta * torch.squeeze(stage2_delta_k, 1))) *
                                           #                                    (self.stage_num[2] * (1 + self.lambda_delta * torch.squeeze(stage3_delta_k, 1)))
                                           )
        # import ipdb
        # ipdb.set_trace()
        regress_class = (stage1_regress + stage2_regress + stage3_regress) * self.class_range
        regress_class = torch.squeeze(regress_class, 1)
        # time_list_[10] = time_list_[10] + (time.time() - current_time_)
        return regress_class


class TrialSSRNet(nn.Module):
    def __init__(self, stage_num=[3, 3, 3], image_size=64,
                 class_range=101, lambda_index=1., lambda_delta=1.):
        super(TrialSSRNet, self).__init__()
        self.image_size = image_size
        self.stage_num = stage_num
        self.lambda_index = lambda_index
        self.lambda_delta = lambda_delta
        self.class_range = class_range
        
        self.stream1_stage3 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.stream1_stage3_former = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.AvgPool2d(2, 2)
        )
        self.stream1_stage3_latter = nn.Sequential(
            # nn.Conv2d(3, 32, 3, 1, 1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        
        self.stream2_stage3 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(2, 2)
        )
        
        self.stream2_stage3_former = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            # nn.MaxPool2d(2, 2)
        )
        self.stream2_stage3_latter = nn.Sequential(
            # nn.Tanh(),
            nn.MaxPool2d(2, 2)
        )
    
    def forward(self, image_):
        current_time_ = time.time()
        feature_stream1_stage3 = self.stream1_stage3(image_)
        time_list_[0] = time_list_[0] + (time.time() - current_time_)
        
        current_time_ = time.time()
        feature_stream1_stage3 = self.stream1_stage3_former(image_)
        time_list_[1] = time_list_[1] + (time.time() - current_time_)
        
        current_time_ = time.time()
        feature_stream1_stage3 = self.stream1_stage3_latter(feature_stream1_stage3)
        time_list_[2] = time_list_[2] + (time.time() - current_time_)
        
        current_time_ = time.time()
        feature_stream1_stage3 = self.stream2_stage3(image_)
        time_list_[3] = time_list_[3] + (time.time() - current_time_)
        
        current_time_ = time.time()
        feature_stream1_stage3 = self.stream2_stage3_former(image_)
        time_list_[4] = time_list_[4] + (time.time() - current_time_)
        
        current_time_ = time.time()
        feature_stream1_stage3 = self.stream2_stage3_latter(feature_stream1_stage3)
        time_list_[5] = time_list_[5] + (time.time() - current_time_)


def demo_test():
    import time
    net = SSRNet()
    # net = TrialSSRNet()
    net = net.cuda('cuda')
    net.eval()
    x = torch.randn(1, 3, 64, 64).cuda('cuda')
    test_numbers_ = 1000
    a_time = time.time()
    for i in range(test_numbers_):
        y = net(x)
    cost_time = time.time() - a_time
    print("time costs:{} s, average_time:{} s\n".format(cost_time, cost_time / test_numbers_))
    # print(y.size())
    # print(y)
    for time_sum_ in time_list_:
        print(1000 * time_sum_ / test_numbers_, end=' ms\n')
    print("\ntotol: {} ms".format(sum(time_list_)))


if __name__ == "__main__":
    demo_test()

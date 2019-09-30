# -*- coding: utf-8 -*-

__author__ = 'kohou.wang'
__time__ = '19-9-24'
__email__ = 'oukohou@outlook.com'

# If this runs wrong, don't ask me, I don't know why;
# If this runs right, thank god, and I don't know why.
# Maybe the answer, my friend, is blowing in the wind.
# Well, I'm kidding... Always, Welcome to contact me.

"""Description for the script:
train SSR-Net.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import time
import copy
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
from datasets.read_imdb_data import IMDBDatasets
from datasets.read_megaasian_data import MegaAgeAsianDatasets
from datasets.read_face_age_data import FaceAgeDatasets
from SSR_models.SSR_Net_model import SSRNet
from SSR_models.ssrnet_hans import ssrnet_hans
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model_, dataloaders_, criterion_, optimizer_, num_epochs_=25, tensorboard_writer=None):
    global lr_scheduler
    
    since = time.time()
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model_.state_dict())
    best_acc = 0.0
    # tensorboard_writer.add_graph(model_, dataloaders_['train'])
    for epoch in range(num_epochs_):
        print('\nEpoch {}/{}'.format(epoch, num_epochs_ - 1))
        print('-' * 10)
        
        # for phase in ['train', 'val']:
        for phase in sorted(dataloaders_.keys()):
            if phase == 'train':
                model_.train()  # Set model to training mode
                print('in train mode...')
            else:
                print('in {} mode...'.format(phase))
                model_.eval()  # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects_3 = 0
            running_corrects_5 = 0
            
            for i, (inputs, labels) in enumerate(dataloaders_[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device).float()
                
                # zero the parameter gradients
                optimizer_.zero_grad()
                
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_(inputs)
                    loss = criterion_(outputs, labels)
                    # import ipdb
                    # ipdb.set_trace()
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer_.step()
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects_3 += torch.sum(torch.abs(outputs - labels) < 3)  # CA 3
                running_corrects_5 += torch.sum(torch.abs(outputs - labels) < 5)  # CA 5
            
            epoch_loss = running_loss / len(dataloaders_[phase].dataset)
            CA_3 = running_corrects_3.double() / len(dataloaders_[phase].dataset)
            CA_5 = running_corrects_5.double() / len(dataloaders_[phase].dataset)
            
            # add running loss to tensorboard.
            if tensorboard_writer:
                tensorboard_writer.add_scalars('losses',
                                               {'{} loss'.format(phase): epoch_loss},
                                               epoch + 1)
            
            # print("inputs:{}".format(inputs))
            # print("outputs:{}".format(outputs))
            # print("labels:{}".format(labels))
            
            print('{} Loss: {:.4f} CA_3: {:.4f}, CA_5: {:.4f}'.format(phase, epoch_loss, CA_3, CA_5))
            time_elapsed = time.time() - since
            print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            
            # deep copy the model
            if phase == 'val' and CA_3 > best_acc:
                best_acc = CA_3
                best_model_wts = copy.deepcopy(model_.state_dict())
            if phase == 'val':
                val_acc_history.append(CA_3)
        
        lr_scheduler.step(epoch)
        if tensorboard_writer:
            tensorboard_writer.add_graph(model_, inputs)
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val CA_3: {:4f}'.format(best_acc))
    
    # load best model weights
    model_.load_state_dict(best_model_wts)
    return model_, val_acc_history


if __name__ == "__main__":
    # train_data_base_path = '/home/data/CVAR-B/study/projects/face_properties/age_estimation/datasets/IMDB/filtered_imdb_crop/resized_64'
    train_data_base_path = '/home/CVAR-B/study/projects/face_properties/age_estimation/datasets/megaage_asion/megaage_asian/megaage_asian/train'
    # batch_size = 1248
    batch_size = 50
    input_size = 50
    num_epochs = 90
    learning_rate = 0.0015  # originally 0.001
    weight_decay = 1e-4  # originally 1e-4
    augment = False
    load_pretrained = True
    
    tensorboard_writer = SummaryWriter(
        '/home/CVAR-B/study/projects/face_properties/age_estimation/trained_models/SSR_Net_MegaAge_Asian/logdir/L1Loss_epoch{}_lr{}_batch{}'.format(
            num_epochs, learning_rate, batch_size
        ))
    
    model_to_train = SSRNet(image_size=input_size)
    # model_to_train = ssrnet(stage_num=[3, 3, 3], lambda_local=1., lambda_d=1., age=101)
    if load_pretrained:
        loaded_model = torch.load(
            '/home/data/CVAR-B/study/projects/face_properties/age_estimation/trained_models/SSR_Net_MegaAge_Asian/model_Adam_MSELoss_LRDecay_weightDecay0.0001_batch50_lr0.0005_epoch90_64x64.pth'
        )
        model_to_train.load_state_dict(loaded_model['state_dict'])
    
    # # for IMDB:
    # all_files = pd.read_csv("datasets/train.csv")
    # all_files = all_files[:16000]  # get a small part for fast convergence.
    # train_data_list, val_data_list = train_test_split(all_files, test_size=0.2, random_state=2019)
    #
    # # load dataset
    # train_gen = IMDBDatasets(train_data_list, train_data_base_path, mode="train",
    #                          augment=augment,
    #                          )
    # train_loader = DataLoader(train_gen, batch_size=batch_size, shuffle=True, pin_memory=True,
    #                           num_workers=0)
    #
    # val_gen = IMDBDatasets(val_data_list, train_data_base_path,
    #                        augment=augment,
    #                        mode="train",
    #                        )
    # val_loader = DataLoader(val_gen, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    
    import random
    
    # for MegaAgeAsian datasets:
    total_image_path = open(
        '/home/CVAR-B/study/projects/face_properties/age_estimation/datasets/megaage_asion/megaage_asian/megaage_asian/list/train_name.txt').readlines()
    total_age_label = open(
        '/home/CVAR-B/study/projects/face_properties/age_estimation/datasets/megaage_asion/megaage_asian/megaage_asian/list/train_age.txt').readlines()
    random.seed(2019)
    random.shuffle(total_image_path)
    random.seed(2019)
    random.shuffle(total_age_label)
    train_image_path = total_image_path[:int(len(total_image_path) * 0.9)]
    val_image_path = total_image_path[int(len(total_image_path) * 0.9):]
    
    train_age_label = total_age_label[:int(len(total_age_label) * 0.9)]
    val_age_label = total_age_label[int(len(total_age_label) * 0.9):]
    train_gen = MegaAgeAsianDatasets(train_image_path, train_age_label, train_data_base_path, mode="train",
                                     augment=augment,
                                     )
    val_gen = MegaAgeAsianDatasets(val_image_path, val_age_label, train_data_base_path,
                                   augment=augment,
                                   mode="train",
                                   )
    
    # # for face age Datasets
    # all_files = pd.read_csv("/home/data/CVAR-B/study/projects/face_properties/age_estimation/datasets/face_age_train.csv")
    # all_files = all_files.sample(frac=1.)
    # all_files = all_files[:4000]  # get a small part for fast convergence.
    # train_data_list, val_data_list = train_test_split(all_files, test_size=0.2, random_state=2019)
    # train_gen = FaceAgeDatasets(train_data_list,)
    # val_gen = FaceAgeDatasets(val_data_list)
    
    train_loader = DataLoader(train_gen, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=0)
    val_loader = DataLoader(val_gen, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    
    test_image_path = open(
        '/home/data/CVAR-B/study/projects/face_properties/age_estimation/datasets/megaage_asion/megaage_asian/megaage_asian/list/test_name.txt').readlines()
    test_age_label = open(
        '/home/data/CVAR-B/study/projects/face_properties/age_estimation/datasets/megaage_asion/megaage_asian/megaage_asian/list/test_age.txt').readlines()
    test_data_base_path = '/home/data/CVAR-B/study/projects/face_properties/age_estimation/datasets/megaage_asion/megaage_asian/megaage_asian/test'
    test_gen = MegaAgeAsianDatasets(test_image_path, test_age_label, test_data_base_path, mode="train",
                                    augment=augment,
                                    )
    test_loader = DataLoader(test_gen, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    
    total_dataloader = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }
    
    model_to_train = model_to_train.to(device)
    
    params_to_update = model_to_train.parameters()
    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    optimizer_ft = optim.Adam(params_to_update, lr=learning_rate, weight_decay=weight_decay)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
    
    # Train and evaluate
    model_to_train, hist = train_model(model_to_train, total_dataloader, criterion, optimizer_ft,
                                       num_epochs_=num_epochs,
                                       )
    
    torch.save({
        'epoch': num_epochs,
        'state_dict': model_to_train.state_dict(),
        'optimizer_state_dict': optimizer_ft.state_dict(),
    },
        '/home/CVAR-B/study/projects/face_properties/age_estimation/trained_models/SSR_Net_MegaAge_Asian/model_Adam_L1Loss_LRDecay_weightDecay{}_batch{}_lr{}_epoch{}+90_64x64.pth'.format(
            weight_decay, batch_size, learning_rate, num_epochs))

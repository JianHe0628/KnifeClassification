## import libraries for training
import sys
import warnings
from datetime import datetime
from timeit import default_timer as timer
import pandas as pd
import torch.optim
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data import knifeDataset
from datetime import datetime
import timm
from torchvision import models
from sklearn.metrics import confusion_matrix
import seaborn as sns
import csv
from argparse import ArgumentParser
from utils import *
warnings.filterwarnings('ignore')

def _parser():
    parser = ArgumentParser()
    parser.add_argument('Run_Location',help='Select either "Colab" or "PC"',type=str,default='PC')
    return parser.parse_args()

if not _parser().Run_Location.lower()=='colab':
    print('Running in PC Mode')
else:
    print('Running in Colab Mode')

## Writing the loss and results
if not os.path.exists("./logs/"):
    os.mkdir("./logs/")

log = Logger()
log.open("logs/%s_log_train.txt")
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                           |----- Train -----|----- Valid----|---------|\n')
log.write('mode     iter     epoch    |       loss      |        mAP    | time    |\n')
log.write('-------------------------------------------------------------------------------------------\n')

## Training the model
def train(train_loader,model,criterion,optimizer,epoch,valid_accuracy,start):
    losses = AverageMeter()
    model.train()
    model.training=True
    for i,(images,target,fnames) in enumerate(train_loader):
        img = images.cuda(non_blocking=True)
        label = target.cuda(non_blocking=True)
        
        with torch.cuda.amp.autocast():
            logits = model(img)
        # loss = criterion(logits, nn.functional.one_hot(label, config.n_classes))
        loss = criterion(logits, label)
        losses.update(loss.item(),images.size(0))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()        
        optimizer.zero_grad()
        scheduler.step()

        print('\r',end='',flush=True)
        message = '%s %5.1f %6.1f        |      %0.3f     |      %0.3f     | %s' % (\
                "train", i, epoch,losses.avg,valid_accuracy[0],time_to_str((timer() - start),'min'))
        print(message , end='',flush=True)
        ##Store Result Data
        global lr
        lr = get_learning_rate(optimizer)
        mode_data.append('train');epoch_data.append(epoch);loss_data.append(float(losses.avg));lr_data.append(lr)
        accuracy_data.append(float(valid_accuracy[0]));time_data.append(time_to_str((timer() - start),'min'))

    log.write("\n")
    log.write(message)
    return [losses.avg]

# Validating the model
def evaluate(val_loader,model,criterion,epoch,train_loss,start):
    model.cuda()
    model.eval()
    model.training=False
    map = AverageMeter()
    with torch.no_grad():
        for i, (images,target,fnames) in enumerate(val_loader):
            img = images.cuda(non_blocking=True)
            label = target.cuda(non_blocking=True)
            
            with torch.cuda.amp.autocast():
                logits = model(img)
                preds = logits.softmax(1)

            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
            map.update(valid_map5,img.size(0))
            print('\r',end='',flush=True)
            message = '%s   %5.1f %6.1f       |      %0.3f     |      %0.3f    | %s' % (\
                    "val", i, epoch, train_loss[0], map.avg,time_to_str((timer() - start),'min'))
            print(message, end='',flush=True)
            ##Store Result Data
            mode_data.append('val');epoch_data.append(epoch);loss_data.append(float(train_loss[0]));lr_data.append(lr)
            accuracy_data.append(float(map.avg));time_data.append(time_to_str((timer() - start),'min'))
        log.write("\n")  
        log.write(message)
    return [map.avg]

## Computing the mean average precision, accuracy 
def map_accuracy(probs, truth, k=5):
    with torch.no_grad():
        value, top = probs.topk(k, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))

        # top accuracy
        correct = correct.float().sum(0, keepdim=False)
        correct = correct / len(truth)

        accs = [correct[0], correct[0] + correct[1] + correct[2] + correct[3] + correct[4]]
        map5 = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5
        acc1 = accs[0]
        acc5 = accs[1]
        return map5, acc1, acc5



#train
def main_training(_lr,_optimizer,_batchsize,weight_decay,moment):
    d = datetime.now()
    ## Loading the model to run
    global model; global device
    # model = models.efficientnet_v2_m(weights='IMAGENET1K_V1')
    # ftrs = model.classifier[1].in_features
    # model.classifier[1] = torch.nn.Linear(in_features=ftrs,out_features=config.n_classes)
    model = timm.create_model(model_name, pretrained=True,num_classes=config.n_classes)
    model.load_state_dict(torch.load('New_Result_Files\\EffcientnetV2\\Final_extraaug\\highest\\tf_efficientnetv2_m_doubleddataset\\E20-0.9426.pt'))
    model.classifier = torch.nn.Sequential(
    nn.BatchNorm1d(1280),
    # torch.nn.Linear(in_features=1280, 
    #                 out_features=1046),
    # nn.LeakyReLU(),
    nn.Dropout(p=0.5, inplace=False),
    torch.nn.Linear(in_features=1280, out_features=config.n_classes, bias=True))
    print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # feats_list = list(model.features)
    # new_feats = []
    # for feats in feats_list:
    #     new_feats.append(feats)
    #     if isinstance(feats, torch.nn.Conv2d):
    #         new_feats.append(torch.nn.Dropout(p=0.1,inplace=True))
    # model.features = torch.nn.Sequential(*new_feats)

    model.to(device)
    ######################## load file and get splits #############################
    train_imlist = pd.read_csv("train.csv")
    train_gen = knifeDataset(train_imlist,mode="train")
    train_loader = DataLoader(train_gen,batch_size=_batchsize,shuffle=True,pin_memory=True,num_workers=0)
    val_imlist = pd.read_csv("test.csv")
    val_gen = knifeDataset(val_imlist,mode="val")
    val_loader = DataLoader(val_gen,batch_size=_batchsize,shuffle=False,pin_memory=True,num_workers=0)
    ############################# Parameters #################################
    optimizer = optim.Adam(model.parameters(), _lr) if _optimizer.lower() == 'adam' else optim.SGD(model.parameters(), lr=_lr)
    global scheduler; global criterion
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.epochs * len(train_loader), eta_min=0,last_epoch=-1)
    criterion = nn.CrossEntropyLoss().cuda()
    ############################# Training #################################
    global scaler
    # start_epoch = 0
    val_metrics = [0]
    scaler = torch.cuda.amp.GradScaler()
    start = timer()
    accuracy_compare = 0
    #################Init Save Path ####################################
    Folder_Name = f'{model_name}_{_lr}_{_optimizer}_{_batchsize}_{d.strftime("%Y%m%d_%H%M%S")}'
    Model_File_Location = "drive/MyDrive/New_Result_Files" if _parser().Run_Location.lower() == 'colab' else "New_Result_Files"
    Unique_Folder = f'{Model_File_Location}/{Folder_Name}' if _parser().Run_Location.lower() == 'colab' else f'{Model_File_Location}\\{Folder_Name}'
    
    if not os.path.exists(Model_File_Location):
        os.mkdir(Model_File_Location)
    if not os.path.exists(Unique_Folder):
        os.mkdir(Unique_Folder)

    for epoch in range(0,config.epochs):
        train_metrics = train(train_loader,model,criterion,optimizer,epoch,val_metrics,start)
        val_metrics = evaluate(val_loader,model,criterion,epoch,train_metrics,start)

        ## Storing Data into CSV
        df = pd.DataFrame({'Mode':mode_data,'Epoch':epoch_data,'Loss':loss_data,'Accuracy':accuracy_data,'Time':time_data, 'Learning Rate':lr_data})
        df['Optimizer'] = _optimizer; df['Batchsize'] = _batchsize; df['Data Aug'] = str(hyper_parameter_sweep['data augmented']); df['Additional Param'] = str(hyper_parameter_sweep['Additional'])
        if _parser().Run_Location.lower() == 'colab':
            df.to_csv(f'{Unique_Folder}/Result.csv',index=False)
        else:
            df.to_csv(f'{Unique_Folder}\\Result.csv',index=False)

        #Save Model
        if accuracy_data[-1] > accuracy_compare:  
            accuracy_compare = round(accuracy_data[-1],4)
            print(f'\nStoring Model Knife-{model_name}-E{epoch}')
            if _parser().Run_Location.lower() == 'colab':
                filename = f"{Unique_Folder}/E{epoch}-{accuracy_compare}" + ".pt" 
            else:
                filename = f"{Unique_Folder}\\E{epoch}-{accuracy_compare}" + ".pt" 
            torch.save(model.state_dict(), filename)

if __name__ == '__main__':
    global mode_data; global epoch_data; global loss_data; global accuracy_data; global time_data; global lr_data; global model_name; global hyper_parameter_sweep
    hyper_parameter_sweep = {'lr':[0.01],
                             'optimizer':['SGD'],
                             'batchsize':[8],
                             'Model Name':'tf_efficientnetv2_m.in21k_ft_in1k',
                             #Not Affected Change
                             'data augmented':['Special Layers'],
                             'Total Epoch':config.epochs,
                             'Additional':'',
                             'weight decay':[0],
                             'Momentum':[0]
                             }

    model_name = hyper_parameter_sweep['Model Name']
    for __lr in hyper_parameter_sweep['lr']:
        for _optim in hyper_parameter_sweep['optimizer']:
            for moment in hyper_parameter_sweep['Momentum']:
                for weight_decay in hyper_parameter_sweep['weight decay']:
                    torch.cuda.empty_cache()
                    mode_data,epoch_data,loss_data,accuracy_data,time_data,lr_data = [],[],[],[],[],[]
                    time = datetime.now().strftime("%Y%m%d_%H%M%S"); track_file = 'Run_Tracker.csv'
                    main_training(__lr,_optim,8,weight_decay,moment)
                    if not os.path.isfile(track_file):
                        df = pd.DataFrame({'Model Name':hyper_parameter_sweep['Model Name'],
                                        'Total Epoch':hyper_parameter_sweep['Total Epoch'],
                                        'Learning Rate':__lr,
                                        'Batch Size':8,
                                        'Optimzer':_optim, 
                                        'Data Augmented':hyper_parameter_sweep['data augmented'],
                                        'Time Created':time,
                                        'Weight Decay':weight_decay,
                                        'Additional':hyper_parameter_sweep['Additional']})
                        df.to_csv(f'{track_file}',index=False)
                    else:
                        with open(track_file,'a') as f:
                            writer = csv.writer(f)
                            row = [hyper_parameter_sweep['Model Name'],hyper_parameter_sweep['Total Epoch'],__lr,8,_optim,hyper_parameter_sweep['data augmented'],time,weight_decay,f"{hyper_parameter_sweep['Additional']}"]
                            writer.writerow(row)
    

   

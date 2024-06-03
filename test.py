## import libraries for training
import warnings
from datetime import datetime
from timeit import default_timer as timer
import pandas as pd
import torch.optim
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from data import knifeDataset
import timm
from utils import *
from argparse import ArgumentParser
warnings.filterwarnings('ignore')
def _parser():
    parser = ArgumentParser()
    parser.add_argument('Model_Location',help='Pass in the Path of Model',type=str)
    return parser.parse_args()
# Validating the model
def evaluate(val_loader,model):
    model.cuda()
    model.eval()
    model.training=False
    map = AverageMeter(); acc1 = AverageMeter(); acc5 = AverageMeter()
    with torch.no_grad():
        for i, (images,target,fnames) in enumerate(val_loader):
            img = images.cuda(non_blocking=True)
            label = target.cuda(non_blocking=True)
            
            with torch.cuda.amp.autocast():
                logits = model(img)
                preds = logits.softmax(1)
            
            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
            map.update(valid_map5,img.size(0))
            acc1.update(valid_acc1,img.size(0))
            acc5.update(valid_acc5,img.size(0))
            print(acc1.avg,acc5.avg)
    return map.avg

## Computing the mean average precision, accuracy 
def map_accuracy(probs, truth, k=5):
    with torch.no_grad():
        value, top = probs.topk(k, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))
        # print(torch.argmax(probs, dim=1),torch.argmax(probs, dim=1).shape)
        # print(top.shape)
        # print(truth,truth.shape)
        
        All_probs.extend(torch.argmax(probs, dim=1).cpu().numpy())
        All_truth.extend(truth.cpu().numpy())
        # top accuracy
        correct = correct.float().sum(0, keepdim=False)
        correct = correct / len(truth)

        accs = [correct[0], correct[0] + correct[1] + correct[2] + correct[3] + correct[4]]
        map5 = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5
        acc1 = accs[0]
        acc5 = accs[1]
        return map5, acc1, acc5

######################## load file and get splits #############################
print('reading test file')
test_files = pd.read_csv("test.csv")
print('Creating test dataloader')
test_gen = knifeDataset(test_files,mode="val")
test_loader = DataLoader(test_gen,batch_size=64,shuffle=False,pin_memory=True,num_workers=0)

print('loading trained model')
model = timm.create_model('tf_efficientnetv2_m.in21k_ft_in1k', pretrained=True,num_classes=config.n_classes)
Model_Location = _parser().Model_Location
model.classifier = torch.nn.Sequential(
    nn.BatchNorm1d(1280),
    torch.nn.Linear(in_features=1280, out_features=config.n_classes, bias=True))
model.load_state_dict(torch.load(Model_Location),strict=False)
# model.load_state_dict(torch.load(Model_Location))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

############################# Training #################################
print('Evaluating trained model')
All_probs = []
All_truth = []
map = evaluate(test_loader,model)
# cm = confusion_matrix(All_truth, All_probs)
# df_cm = pd.DataFrame(cm)
# plt.figure(figsize=(20,10)) 

# ax = sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')
# print(type(ax))

# plt.ylabel('True label')
# plt.xlabel('Predicted label')

# plt.show()

print("mAP =",map)
    
   

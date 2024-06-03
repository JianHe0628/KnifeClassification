import matplotlib.pyplot as plt
import argparse
import pandas as pd
import os

def _parser():
    parser = argparse.ArgumentParser(description='Plotting script')
    parser.add_argument('path', help='Enter path of data',default=None,type=str)
    parser.add_argument('--m', help='Multiple Plots',default=0,type=int)
    return parser.parse_args()

def create_df(file_list,path):
    old_count = 0; Epoch = [0]; Loss = [0]; Map = [0]
    for file in file_list:
        if file.endswith('.csv'):
            with open(os.path.join(path,file)) as f:
                lines = f.read().splitlines()
                for x in lines:
                    items = x.split(
                        ',')
                    count = items[1]
                    if items[0] == 'val':
                        if str(count) == old_count:
                            Epoch[int(count)] = float(items[1]); Loss[int(count)] = float(items[2]); Map[int(count)] = float(items[3])
                        else:
                            old_count = count
                            Epoch.append(float(items[1])); Loss.append(float(items[2])); Map.append(float(items[3]))
    return pd.DataFrame({'Epoch':Epoch,'Loss':Loss,'Map':Map})

if _parser().path:
    datapath = _parser().path
    #plot line graph from csv
    file_list = os.listdir(datapath)
    if _parser().m == 0:
        df = create_df(file_list,datapath)
        df.drop(df.tail(1).index,inplace=True)
        print(df)
        df.plot(x='Epoch',y=['Loss'],title='Loss vs Epoch',xlabel='Epoch',ylabel='Loss',grid=True)
        df.plot(x='Epoch',y=['Map'],title='Map vs Epoch',xlabel='Epoch',ylabel='Map',grid=True)
        df['Loss'] = df['Loss'].div(df['Loss'][0])
        df.plot(x='Epoch',y=['Loss','Map'],title='Loss vs Epoch',xlabel='Epoch',ylabel='Loss and Map',grid=True)
        plt.show()
    else:
        print('Multiple Plots Enabled')
        for folder in file_list:
            folder_list = os.listdir(os.path.join(datapath,folder))
            df = create_df(folder_list,os.path.join(datapath,folder))
            plt.plot(df['Epoch'],df['Loss'],label=f'{folder}')
            plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.show()
        for folder in file_list:
            folder_list = os.listdir(os.path.join(datapath,folder))
            df = create_df(folder_list,os.path.join(datapath,folder))
            df.drop(df.tail(1).index,inplace=True)
            plt.plot(df['Epoch'],df['Map'],label=f'{folder}')
            plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Map')
        plt.title('Map vs Epoch')
        plt.show()
    
    


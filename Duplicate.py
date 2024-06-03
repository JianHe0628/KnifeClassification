from shutil import copyfile
import argparse
import os

def _parser():
    parse = argparse.ArgumentParser(description='Copy files from one directory to another')
    parse.add_argument('datapath', help='Source directory',type=str)
    parse.add_argument('--csv', help='test csv',type=str,default='')
    return parse.parse_args()



folderlist = os.listdir(_parser().datapath)
for folder in folderlist:
    filelist = os.listdir(os.path.join(_parser().datapath,folder))
    for file in filelist:
        copyfile(os.path.join(_parser().datapath,folder,file),os.path.join(_parser().datapath,folder,f'{file.rstrip(".jpg").rstrip(".JPEG")}_copy.jpg'))

if not _parser().csv == '':
    content = []
    with open(_parser().csv,'r') as f:
        lines = f.readlines()
        for line in lines:
            if '.j' in line.lower():
                name = line.split(',')[0].rstrip('.jpg').rstrip('.JPEG')
                label = line.split(',')[1]
                content.append(f'{name}_copy.jpg,{label}')
        f.close()
    print(content[:10])
    with open(_parser().csv,'a') as f:
        for line in content:
            f.write(line)
        f.close()

    
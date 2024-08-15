import glob
import os
import SimpleITK as sitk
import numpy as np
import argparse
from medpy import metric

def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())
def hd(pred,gt):
        if pred.sum() > 0 and gt.sum()>0:
            hd95 = metric.binary.hd95(pred, gt)
            return  hd95
        else:
            return 0
            
def process_label(label):
    mcl = label == 1
   
    return mcl

def test(fold):
    label_path = "/"
    infer_path = "/"
    
    label_files = glob.glob(os.path.join(label_path, 'labels*.nii.gz'))
    infer_files = glob.glob(os.path.join(infer_path, 'images*.nii.gz'))

    label_files.sort(key=lambda x: int(os.path.basename(x)[6:-7]))
    infer_files.sort(key=lambda x: int(os.path.basename(x)[6:-7]))

    label_list = label_files
    infer_list = infer_files

    # label_list = [os.path.basename(file) for file in label_files]
    # infer_list = [os.path.basename(file) for file in infer_files]
    print("loading success...")
    print(label_list)
    print(infer_list)
    Dice_mcl=[]
    hd_mcl=[]

    
    file=infer_path + '_zj123best/'

    if not os.path.exists(file):
        os.makedirs(file)
    fw = open(file+'/dice_pre.txt', 'a')
    
    for label_path,infer_path in zip(label_list,infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label,infer = read_nii(label_path),read_nii(infer_path)
        label_mcl=process_label(label)
        infer_mcl=process_label(infer)
        
        Dice_mcl.append(dice(infer_mcl,label_mcl))

        
        hd_mcl.append(hd(infer_mcl,label_mcl))
        
        fw.write('*'*20+'\n',)
        fw.write(infer_path.split('/')[-1]+'\n')
        fw.write('Dice_mcl: {:.4f}\n'.format(Dice_mcl[-1]))

        
        fw.write('hd_mcl: {:.4f}\n'.format(hd_mcl[-1]))
        
        dsc=[]
        HD=[]

        dsc.append(np.mean(Dice_mcl[-1]))

        fw.write('DSC:'+str(np.mean(dsc))+'\n')
        
        HD.append(hd_mcl[-1])

        fw.write('hd:'+str(np.mean(HD))+'\n')
        
    
    fw.write('*'*20+'\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_mcl:'+str(np.mean(Dice_mcl))+'\n')
    
    fw.write('Mean_hd\n')
    fw.write('hd_mcl:'+str(np.mean(hd_mcl))+'\n')
   
    fw.write('*'*20+'\n')
    
    dsc=[]
    dsc.append(np.mean(Dice_mcl))
    fw.write('dsc:'+str(np.mean(dsc))+'\n')
    
    HD=[]
    HD.append(np.mean(hd_mcl))
    fw.write('hd:'+str(np.mean(HD))+'\n')
    
    print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("fold", help="fold name")
    args = parser.parse_args()
    fold=args.fold
    test(fold)

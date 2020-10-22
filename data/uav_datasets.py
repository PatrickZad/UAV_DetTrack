from detectron2.data import DatasetCatalog,MetadataCatalog
from detectron2.structures import BoxMode
from os.path import join
import os
import cv2 as cv
import numpy as np

__all__=['visdrone_d_train','visdrone_d_val','visdrone_dt_train',
        'visdrone_t_val','uavdt_d_train','uavdt_d_val','uavdt_t_val']

datasets_dir=join('..','Datasets')
visdrone_dir=join(datasets_dir,'VisDrone')
uavdt_dir=join(datasets_dir,'UAVDT')
'''
ignored regions(0)->ignored regions(-1),ignored when training and evaluating
pedestrian(1)->people(0)
people(2)->people(0)
bicycle(3)->bicycle(1)
motor(10)->motor(2)
bus(9)->bus(3)
car(4)->car(4)
van(5)->van(5)
truck(6)->truck(6)
tricycle(7)->tricycle(7)
awning-tricycle(8), 
others(11)->others(9)
'''
visdrone_category_trans=[-1,0,0,1,4,5,6,7,8,3,2,9]
def visdrone_d_train():
    train_dir=join(visdrone_dir,'VisDrone2019-DET-train')
    anno_dir=join(train_dir,'annotations')
    img_dir=join(train_dir,'images')
    result_list=[]
    for img_file in os.listdir(img_dir):
        file_dict={}
        img_id=img_file[:-3]
        file_dict['file_name']=img_file
        img_arr=cv.imread(join(img_dir,img_file))
        file_dict['height']=img_arr.shape[0]
        file_dict['width']=img_arr.shape[1]
        file_dict['image_id']=img_id
        anno_list=[]
        with open(join(anno_dir,img_id+'.txt'),'r') as anno_file:
            obj_line=anno_file.readline()
            while obj_line is not None:
                num_arr=np.float32(obj_line.strip().split(','))
                cate=visdrone_category_trans[int(num_arr[5])]
                if num_arr[4] == 0 or cate<0 or num_arr[6]>0 or num_arr[7]>1:
                    continue
                obj_dict={}
                obj_dict['bbox']=num_arr[:4].tolist()
                obj_dict['bbox_mode']=BoxMode.XYWH_ABS
                obj_dict['category_id']=cate
                anno_list.append(obj_dict)
        file_dict['annotations']=anno_list
        result_list.append()
    return result_list
                

    
def visdrone_d_val():
    pass
def visdrone_dt_train():
    pass
def visdrone_t_val():
    pass
def uavdt_d_train():
    videos_dir=join(uavdt_dir,'UAV-benchmark-M')
    gts_dir=join(uavdt_dir,'UAV-benchmark-MOTD_v1.0','GT')
    video_id=os.listdir(videos_dir)
def uavdt_d_val():
    pass
def uavdt_t_val():
    pass


DatasetCatalog.register('visdrone_d_train',visdrone_d_train)
DatasetCatalog.register('visdrone_d_val',visdrone_d_val)
DatasetCatalog.register('visdrone_dt_train',visdrone_dt_train)
DatasetCatalog.register('visdrone_t_val',visdrone_t_val)

MetadataCatalog.get('visdrone_d_train').thin_classes=['people','bicycle','motor','bus','car','van',
                                                      'truck','tricycle','awning-tricycle', 'others']
MetadataCatalog.get('visdrone_d_val').thin_classes=['people','bicycle','motor','bus','car','van',
                                                      'truck','tricycle','awning-tricycle', 'others']
MetadataCatalog.get('visdrone_dt_train').thin_classes=['people','bicycle','motor','bus','car','van',
                                                      'truck','tricycle','awning-tricycle', 'others']
MetadataCatalog.get('visdrone_t_val').thin_classes=['people','bicycle','motor','bus','car','van',
                                                      'truck','tricycle','awning-tricycle', 'others']

DatasetCatalog.register('uavdt_d_train',uavdt_d_train)
DatasetCatalog.register('uavdt_d_val',uavdt_d_val)
DatasetCatalog.register('uavdt_t_val',uavdt_t_val)
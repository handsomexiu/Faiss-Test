import h5py
import numpy as np
import faiss
import os
import time

# 字典用来存储数据集的相关信息
data_info={
    'glove':{25:'glove-25-angular',
           50: 'glove-50-angular',
           100: 'glove-100-angular',
           200: 'glove-200-angular'},
    'sift':{128:'sift-128-euclidean'}
}

# 用于存储可视化函数的相关信息
# 如果还需要进行其他实验可以修改相关的字典
task_data_plot ={
    'search_time_npiece_number':{
        'x_label_bar3d':'Number of pieces',
        'y_label_bar3d':'Number of querys',
        'z_label_bar3d':'Time (s)',
        'dx_bar3d':1.5,
        'dy_bar3d':300,
        'title':'Search time with different number of pieces and number of querys',
        'set_xlabel':'number of querys ',
        'set_ylabel':'Times (s)',
        'set_title':'number of pieces',
    },
    'search_recall_npiece_number':{
        'x_label_bar3d':'Number of pieces',
        'y_label_bar3d':'Number of querys',
        'z_label_bar3d':'Recall',
        'dx_bar3d':1.5,
        'dy_bar3d':300,
        'title':'Search Recall with different number of pieces and number of querys',
        'set_xlabel':'number of querys ',
        'set_ylabel':'Recall',
        'set_title':'number of pieces',
    },
    'search_time_number_k':{
        'x_label_bar3d':'Number of querys',
        'y_label_bar3d':'k',
        'z_label_bar3d':'Time (s)',
        'dx_bar3d':300,
        'dy_bar3d':50,
        'title':'Time to search for different number of query and k',
        'set_xlabel':'k ',
        'set_ylabel':'Time (s)',
        'set_title':'number of querys',
    },
    'search_recall_number_k':{
        'x_label_bar3d':'Number of querys',
        'y_label_bar3d':'k',
        'z_label_bar3d':'Recall',
        'dx_bar3d':300,
        'dy_bar3d':50,
        'title':'Recall with different number of query and k',
        'set_xlabel':'k ',
        'set_ylabel':'Recall',
        'set_title':'number of querys',
    },
    'search_time_npiece_k':{
        'x_label_bar3d':'Number of pieces',
        'y_label_bar3d':'k',
        'z_label_bar3d':'Time (s)',
        'dx_bar3d':1.5,
        'dy_bar3d':1.5,# dx_bard,dy_bard是用来调整bar3d的宽度和高度
        'title':'Time to search for different number of pieces and k',
        'set_xlabel':'k ',
        'set_ylabel':'Time (s)',
        'set_title':'number of pieces',
    },
    'search_recall_npiece_k':{
        'x_label_bar3d':'Number of pieces',
        'y_label_bar3d':'k',
        'z_label_bar3d':'Recall',
        'dx_bar3d':1.5,
        'dy_bar3d':1.5,
        'title':'Recall with different number of pieces and k',
        'set_xlabel':'k ',
        'set_ylabel':'Recall',
        'set_title':'number of pieces',
    }
}
import h5py
import numpy as np
import faiss
import os
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from config import *
from faiss_database_create import *
from faiss_test import *

'''
报告内容：可以写道一个docx文件中去
1. 创建向量数据库所需要的时间
2、固定k和number下查询所需要的时间
3、固定k和number下查询的召回率
4、利用for循环增加实验如n_piece和k
4.1用字典进行存储

文件统一命名方式：
切片向量数据库文件名：glove-25-angular_n5_1.index
存放文件夹：index/glove-25-angular_n5
其中index表示存放整体向量数据库；glove-25-angular表示采用的数据集；n5表示切片数量：切5片；1表示第一片
data_name= data_info[data_choice][dim]
folder_path = f"index/{data_name}_n{n_piece}"
data_key=data_name+'_n'+str(n_piece)+'_'+str(i+1)
file_path = f"{folder_path}/{data_key}.index"
'''

# -------------------任务----------------------
# 创建不同分片大小的向量数据库所需要的时间对比
def get_report_create_index_time(data_choice:str='glove',dim:int=25,n_piece:list=[1,5,10,15,20,25],param:str='HNSW64'):
    # 关于创建向量数据库的实验
    create_index_time={}# 创建不同切片向量数据库所需要的时间
    # data_name= data_info[data_choice][dim]
    for n in n_piece:
        start_time = time.time()
        create_index(data_choice,n,dim,param)
        end_time = time.time()
        elapsed_time = end_time - start_time  # 计算创建n_piece向量数据库所需的时间
        # database_key=data_name+'_n'+str(n)
        # create_index_time[database_key]=elapsed_time
        # 不用database_key了，直接用n，这里是为了方便画图，下面的几组实验也是一样的
        create_index_time[n]=elapsed_time
    return create_index_time

# 不同切片数据库在不同k值下的查询时间和召回率对比,query=100(也可以自己定义)
def get_report_n_piece_k(data_choice:str='glove',dim:int=25,n_piece:list=[1,5,10,15,20,25],
               k:list=[10,100,1000,5000,10000,50000,100000],number:int=100,param:str='HNSW64'):
    # 定义一个两层循环，外层循环是n_piece，内层循环是k
    # 首先是创建数据库，这里记录创建不同分片数据库的时间
    # 注意：这里有一个两层字典，字典的第一层是n_piece，第二层是k
    create_index_time=get_report_create_index_time(data_choice,dim,n_piece,param)
    print('数据库创建完毕')
    # 然后是查询数据库，这里记录不同分片数据库在不同k值下的查询时间和召回率对比
    search_time={}# 查询时间
    seach_recall={}# 查询召回率
    for n in n_piece:
        search_time[n]={}
        seach_recall[n]={}
        for k_ in k:
            print(f'现在处理的是{n}分片数据库，k值是{k_}')
            mean_recall,elapsed_time=get_recall(data_choice,n,dim,k_,number)#get_recall返回的是平均召回率和查询时间(np.mean(recall_list),elapsed_time)
            search_time[n][k_]=elapsed_time
            seach_recall[n][k_]=mean_recall
    return create_index_time,search_time,seach_recall

# 此时npiece是固定的为5，对比不同query数量下的查询时间和召回率对比
def get_report_number_k(data_choice:str='glove',dim:int=25,n_piece:int=5,
               k:list=[10,100,1000,5000,10000,50000,100000],number:list=[10,500,1000,5000,10000],param:str='HNSW64'):
    # 首先是创建相关向量数据库
    create_index(data_choice=data_choice,n_piece=n_piece,dim=dim,param=param)
    print('数据库创建完毕')
    # 接下来是查询数据库
    # 注意：这里有一个两层字典，字典的第一层是number，第二层是k
    search_time={}# 查询时间
    seach_recall={}# 查询召回率
    for n in number:
        search_time[n]={}
        seach_recall[n]={}
        for k_ in k:
            print(f'现在处理的是{n}个查询数据，k值是{k_}')
            mean_recall,elapsed_time=get_recall(data_choice,n_piece,dim,k_,n)
            search_time[n][k_]=elapsed_time
            seach_recall[n][k_]=mean_recall
    return search_time,seach_recall

# 此时k固定为1000，对比不同分片数据库下，不同query数量下的查询时间和召回率对比
def get_report_n_piece_number(data_choice:str='glove',dim:int=25,n_piece:list=[1,5,10,15,20,25],
               k:int=1000,number:list=[10,500,1000,5000,10000],param:str='HNSW64'):
    # 首先是创建相关向量数据库
    create_index_time=get_report_create_index_time(data_choice,dim,n_piece,param)
    print('数据库创建完毕')
    # 接下来是查询数据库
    # 注意：这里有一个两层字典，字典的第一层是n_piece，第二层是number
    search_time={}# 查询时间
    seach_recall={}# 查询召回率
    for n in n_piece:
        search_time[n]={}
        seach_recall[n]={}
        for n_ in number:
            print(f'现在处理的是{n}分片数据库，查询数据是{n_}')
            mean_recall,elapsed_time=get_recall(data_choice,n,dim,k,n_)
            search_time[n][n_]=elapsed_time
            seach_recall[n][n_]=mean_recall
    return create_index_time,search_time,seach_recall

# -----------------------可视化-------------------------

# 创建存储图片所需要的文件夹
def create_photo_store(data_choice:str='glove',dim:int=25,faiss_style:str='HNSW64'):
    data_name= data_info[data_choice][dim]
    folder_path = f"figure/{data_name}_{faiss_style}"
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        # 不存在时创建文件夹
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 不存在，已创建。")
    else:
        print(f"文件夹 '{folder_path}' 已经存在。")
    return folder_path

# create_index_time结果可视化
def plot_create_index_time(create_index_time:dict,data_choice:str='glove',dim:int=25,faiss_style:str='HNSW64'):
    # 首先时创建存储图片的文件夹
    folder_path=create_photo_store(data_choice,dim,faiss_style)
    file_path = f"{folder_path}/create_index_time.png"
    # plt.figure(figsize=(10, 6))
    x_values=list(create_index_time.keys())
    y_values=list(create_index_time.values())
    plt.plot(x_values, y_values,label='create_index_time',marker='o')
    # 在每个点上标注具体数值
    # 在每个点上标注y值
    for x, y in zip(x_values, y_values):
        plt.text(x, y, f'{y:.3f}', ha='left', va='bottom')

    # # 添加每个点到x轴的虚线连接
    # for x, y in zip(x_values, y_values):
    #     plt.plot([x, x], [0, y], 'k--', lw=1)
    plt.xlabel("Number of pieces")
    plt.ylabel("Time (s)")
    plt.title("Time to create index for different number of pieces")
    # 添加图列
    plt.legend()
    # 保存图片
    plt.savefig(file_path)
    # # 展示图片
    # plt.show()

# 画柱状图。这里的task_choice是用来选择我们要执行的任务类型，这个在config.py中有定义。task_data_plot这个字典中
# data_dict是该任务对应的输出的数据
def plot_bar3d(task_choice:str,data_dict:dict, data_choice:str='glove', dim:int=25,faiss_style:str='HNSW64'):
    # 首先时创建存储图片的文件夹
    folder_path=create_photo_store(data_choice,dim,faiss_style)
    file_path = f"{folder_path}/{task_choice}_bar3d.png"
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_values=[key1 for key1, inner_dict in data_dict.items() for key2, value in inner_dict.items()]
    y_values=[key2 for key1, inner_dict in data_dict.items() for key2, value in inner_dict.items()]
    z_values=[value for key1, inner_dict in data_dict.items() for key2, value in inner_dict.items()]
    print(x_values,y_values,z_values)
    # Normalize the data for color mapping
    norm = plt.Normalize(min(z_values), max(z_values))
    colors = plt.cm.coolwarm(norm(z_values))
    # Create 3D bar chart with transparency
    dx = task_data_plot[task_choice]['dx_bar3d']
    dy = task_data_plot[task_choice]['dy_bar3d']
    ax.bar3d(x_values, y_values, np.zeros_like(z_values), dx, dy, z_values, shade=False, color=colors, alpha=0.5)
    xlabel,ylabel,zlabel=task_data_plot[task_choice]['x_label_bar3d'],task_data_plot[task_choice]['y_label_bar3d'],task_data_plot[task_choice]['z_label_bar3d']
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    font_size = 12
    plt_title=task_data_plot[task_choice]['title']
    plt.title(plt_title, fontsize=font_size)
    # 设置视角（仰角，方位角）
    ax.view_init(elev=15, azim=250)
    # 保存图片
    plt.savefig(file_path)
    # # 展示图片
    # plt.show()

# 用于画子图，这的操作和plot_bar3d类似，只是这里是画子图
def plot_subplot(task_choice:str,data_dict:dict, data_choice:str='glove', dim:int=25,faiss_style:str='HNSW64'):
    folder_path=create_photo_store(data_choice,dim,faiss_style)
    file_path = f"{folder_path}/{task_choice}_subplot.png"
    key1= list(data_dict.keys())# 这里的key1是number of npiece
    num_subplots = len(data_dict)
    # 每行显示的子图数量
    subplots_per_row = 3
    # 计算总共需要多少行
    num_rows = math.ceil(num_subplots / subplots_per_row)
    # 生成对应数量的子图
    fig, axs = plt.subplots(num_rows, subplots_per_row, figsize=(15, 4 * num_rows), sharex=True)
    # 遍历第一层 key，每个 key 对应一个子图
    for i, (key, inner_dict) in enumerate(data_dict.items()):
        # 计算当前子图在网格中的位置
        row_idx = i // subplots_per_row
        col_idx = i % subplots_per_row
        # 获取子图对象
        ax = axs[row_idx, col_idx] if num_rows > 1 else axs[col_idx]
        # 提取数据
        x_values = list(inner_dict.keys())
        y_values = list(inner_dict.values())
        # 在子图中绘制折线图
        ax.plot(x_values, y_values, marker='o')
        # 添加标签和标题
        set_xlabel, set_ylabel, set_title=task_data_plot[task_choice]['set_xlabel'],task_data_plot[task_choice]['set_ylabel'],task_data_plot[task_choice]['set_title']
        ax.set_xlabel(set_xlabel)
        ax.set_ylabel(set_ylabel)
        ax.set_title(f'{set_title} = {key1[i]}')
        # # 设置y轴的刻度值
        # ax.set_yticks(y_values)
        # ax.set_yticklabels([f'{val:.2f}' for val in y_values])
        # 添加图例
        ax.legend()
    # 隐藏未使用的子图
    for i in range(num_subplots, num_rows * subplots_per_row):
        row_idx = i // subplots_per_row
        col_idx = i % subplots_per_row
        axs[row_idx, col_idx].axis('off')
    # 调整布局
    plt.tight_layout()
    # 增加总的标题和子图标题之间的间隔
    plt.subplots_adjust(top=0.85)
    # 添加总的标题
    plt_title=task_data_plot[task_choice]['title']
    plt.suptitle(plt_title)
    # 保存图片
    plt.savefig(file_path, bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    '''
    相关参数注意：
    data_choice:数据集的选择，glove或者sift
    task_choice:任务的选择，这个在config.py中有定义(其实就是对应上述几个任务的输出，要求命名规范)
    param:faiss中的参数，这个与创建什么类型的数据库有关.
    faiss_style:str='HNSW64'和param是一样的且需要保持一致
    param:主要来创建相关类型数据库，faiss_style是用来命名文件夹的
    '''
    #---------------执行任务----------------
    # 首先定义采用的向量数据库的形式,在这里选择你要构造的数据库类型，参考index_factory的要求
    faiss_style=param='HNSW64'

    # 创建不同分片大小的向量数据库所需要的时间对比
    create_index_time=get_report_create_index_time(data_choice='glove',dim=25,n_piece=[1,5,10,15,20,25],param=param)
    # 不同切片数据库在不同k值下的查询时间和召回率对比，此时query是100
    _,search_time_npiece_k,search_recall_npiece_k=get_report_n_piece_k('glove',25,[1,5,10],
                k=[10,100,1000,5000],number=100,param=param)
    # 此时npiece是固定的为5，对比不同数量query和k的查询时间和召回率对比
    search_time_number_k,search_recall_number_k=get_report_number_k('glove',25,5,k=[10,100,1000,5000],
                number=[10,500,1000,5000],param=param)
    # 此时k固定为1000，对比不同分片数据库下，不同query数量下的查询时间和召回率对比
    _,search_time_npiece_number,search_recall_npiece_number=get_report_n_piece_number('glove',25,[1,5,10],k=1000,
                number=[10,500,1000,5000],param=param)
    
    #---------------可视化----------------
    # create_index_time可视化
    plot_create_index_time(create_index_time,data_choice='glove',dim=25,faiss_style=faiss_style)
    # report_n_piece_k可视化
    plot_bar3d('search_time_npiece_k',search_time_npiece_k, data_choice='glove', dim=25,faiss_style=faiss_style)
    plot_bar3d('search_recall_npiece_k',search_recall_npiece_k, data_choice='glove', dim=25,faiss_style=faiss_style)
    plot_subplot('search_time_npiece_k',search_time_npiece_k, data_choice='glove', dim=25,faiss_style=faiss_style)
    plot_subplot('search_recall_npiece_k',search_recall_npiece_k, data_choice='glove', dim=25,faiss_style=faiss_style)
    # report_number_k可视化
    plot_bar3d('search_time_number_k',search_time_number_k, data_choice='glove', dim=25,faiss_style=faiss_style)
    plot_bar3d('search_recall_number_k',search_recall_number_k, data_choice='glove', dim=25,faiss_style=faiss_style)
    plot_subplot('search_time_number_k',search_time_number_k, data_choice='glove', dim=25,faiss_style=faiss_style)
    plot_subplot('search_recall_number_k',search_recall_number_k, data_choice='glove', dim=25, faiss_style=faiss_style)
    # report_n_piece_number可视化
    plot_bar3d('search_time_npiece_number',search_time_npiece_number, data_choice='glove', dim=25,faiss_style=faiss_style)
    plot_bar3d('search_recall_npiece_number',search_recall_npiece_number, data_choice='glove', dim=25,faiss_style=faiss_style)
    plot_subplot('search_time_npiece_number',search_time_npiece_number, data_choice='glove', dim=25,faiss_style=faiss_style)
    plot_subplot('search_recall_npiece_number',search_recall_npiece_number, data_choice='glove', dim=25,faiss_style=faiss_style)





import h5py
import numpy as np
import faiss
import os
import time
from config import *

# 训练数据分片
# 我们对训练数据进行分片，分成五组即可
# 到时设计的数据切片函数 参数：data_name:str,n_piece:int=5,返回一个字典和data_name对应的数据
# 对于data_name 除了指定具体的名字外如glove-25-angular，还可以指定glove和数据维度25，到时我们用一个字典进行匹配就可以了
def data_piece(data_choice:str='glove',n_piece:int=5,dim:int=25)->dict:# 如果data_choice是glove则dim必须是25,50,100,200，如果是sift则dim必须是128，否则会报错
    n_piece= n_piece#这里是用来定义切片的数量
    data_name= data_info[data_choice][dim] # 这里是为了方便创建文件夹和数据库文件名，方便识别
    glove_file_path = f"data/{data_name}.hdf5"# 数据地址，hdf5格式
    glove_hdf = h5py.File(glove_file_path, "r")# 读取数据
    length_all = len(glove_hdf['train'])# 获取数据的总长度  
    cut_point=int(length_all/n_piece)# 获取切分点,int是向下取整
    data_dict={}# 这里用字典来存取每一段切分点
    id_dict={}# 这里用字典来存取每一段对应的索引
    for i in range(n_piece):
        data_key=data_name+'_n'+str(n_piece)+'_'+str(i+1)
        if i+1 == n_piece:
            data_dict[data_key]=glove_hdf['train'][i*cut_point:]# 把剩下的余数全部纳入不在单独考虑了（如果能整除则恰好每一块长度相同，如果不能整除则最后一段的长度=cut_point+余数），这里是最后一段，
            id=list(range(i*cut_point,length_all))
            id_dict[data_key]=id
        else:
            data_dict[data_key]=glove_hdf['train'][i*cut_point:(i+1)*cut_point]
            id=list(range(i*cut_point,(i+1)*cut_point))
            id_dict[data_key]=id
    return data_dict,id_dict,data_name

# 创建向量数据库所需要的文件夹
def create_folder_if_not_exists(data_name:str,n_piece:int=5)->str:#这里的n_piece需要与data_piece函数中的n_piece一致
    folder_path = f"index/{data_name}_n{n_piece}"
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        # 不存在时创建文件夹
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 不存在，已创建。")
    else:
        print(f"文件夹 '{folder_path}' 已经存在。")
    return folder_path

# 创建向量数据库
# 关于向量数据库所需要的一些个参数也可以设为该函数需要传的参数如measure, param
# 这里的关于分片后的数据的id还得自定义一下
def create_index(data_choice:str='glove',n_piece:int=5,dim:int=25,param:str='HNSW64'):# 如果data_choice是glove则dim必须是25,50,100,200，如果是sift则dim必须是128，否则会报错
    data_dict,id_dict,data_name=data_piece(data_choice,n_piece,dim)
    folder_path = create_folder_if_not_exists(data_name,n_piece)
    for i in range(n_piece):
        data_key=data_name+'_n'+str(n_piece)+'_'+str(i+1) # 我们统一这样命名
        file_path = f"{folder_path}/{data_key}.index"
        dim = dim
        param = param
        measure=faiss.METRIC_L2# 这里固定因为glove和sift都用的是L2距离
        index = faiss.index_factory(dim, param, measure)  
        # 导入数据
        df=data_dict[data_key]
        if data_choice=='glove':
            print(f'现在处理的glove数据集{data_key},需要进行归一化处理')
            faiss.normalize_L2(df)# glove数据采用的angular距离，首先需要进行归一化,然后再进行faiss.METRIC_L2
        # index.add(df)
        # faiss.write_index(index, file_path)
        # 做id的映射和自定义
        xids = np.array(id_dict[data_key])
        IDMap_index = faiss.IndexIDMap(index)
        IDMap_index.add_with_ids(df, xids)
        faiss.write_index(IDMap_index, file_path)
        print(f"{data_key}.index 已创建。")

if __name__ == "__main__":
    '''
    需要传参数：
    data_choice:str,仅可以选择'glove'或'sift'
    n_piece:int,默认为5,表示数据分片的数量
    dim:int,默认为25,表示数据的维度。如果选择了glove则dim必须是25,50,100,200 ,如果选择了sift则dim必须是128
    param:str,默认为'HNSW64',表示faiss的索引参数，具体可以参考faiss的官方文档
    '''
    create_index(data_choice='glove',n_piece=5,dim=25,param='HNSW64')#自定义参数传入
    # create_index('glove',5,25,'HNSW64')

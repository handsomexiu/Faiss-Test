import h5py
import numpy as np
import faiss
import os
import time
from config import *
from faiss_database_create import *

# 测试数据集获取
def get_test_data(data_choice:str='glove',dim:int=25,number:int=100):#这里定义一个number是为了选取测试集数量，默认是100个
    data_name= data_info[data_choice][dim] # 这里是为了方便创建文件夹和数据库文件名，方便识别
    glove_file_path = f"data/{data_name}.hdf5"
    glove_hdf = h5py.File(glove_file_path, "r")# 读取数据
    glove_test = glove_hdf['test'][:number]# 获取测试查询数据
    if data_choice=='glove':
        faiss.normalize_L2(glove_test)# glove数据采用的angular距离，首先需要进行归一化,然后再进行faiss.METRIC_L2
        # faiss.normalize_L2直接作用再原数组上，不需要返回值
    glove_neighbors = glove_hdf['neighbors'][:number]# 获取测
    glove_distances = glove_hdf['distances'][:number]# 获取测试数据集
    return glove_test,glove_neighbors,glove_distances

# 获取id
def get_id(data_choice:str='glove',n_piece:int=5,dim:int=25):
    data_name= data_info[data_choice][dim]
    glove_file_path = f"data/{data_name}.hdf5"# 数据地址，hdf5格式
    glove_hdf = h5py.File(glove_file_path, "r")# 读取数据
    length_all = len(glove_hdf['train'])# 获取数据的总长度  
    cut_point=int(length_all/n_piece)# 获取切分点,int是向下取整
    id_dict={}# 这里用字典来存取每一段对应的索引
    for i in range(n_piece):
        keys='npiece_'+str(i+1)
        if i+1 == n_piece:
            id=list(range(i*cut_point,length_all))
            id=np.array(id)
            id_dict[keys]=id
        else:
            id=list(range(i*cut_point,(i+1)*cut_point))
            id=np.array(id)
            id_dict[keys]=id
    return id_dict

# 测试数据集的拼接
def combine_list(data: list, ids: list, k: int = 100):
    # 将每个元素和对应ID放在一个元组中
    combined_tuples = [(element, id_val) for element, id_val in zip(data, ids)]
    # 对拼接后的元组列表按照元素排序，sorted是从小到大排序的，正好取前k个即可
    sorted_combined_tuples = sorted(combined_tuples, key=lambda x: x[0])
    # 输出排序后的元素和对应的ID列表
    sorted_elements = [element[0] for element in sorted_combined_tuples]
    sorted_ids = [element[1] for element in sorted_combined_tuples]
    return sorted_elements[:k], sorted_ids[:k]

# 获取测试结果
# 这里不需要传参measure，因为两个数据集采用的measure=faiss.METRIC_L2是一样的，除了glove需要加一个归一化
def get_search_result(data_choice:str='glove',n_piece:int=5,dim:int=25,k:int=100,number:int=100,faiss_style:str='HNSW64'):# number是指定测试集的数量,要和get_test_data函数中的number一致
    # data_dict,data_name=data_piece(data_choice,n_piece,dim)
    data_name= data_info[data_choice][dim] 
    folder_path = create_folder_if_not_exists(data_name,n_piece,faiss_style)
    golve_test,_,_=get_test_data(data_choice,dim,number)# 这是一个双层数组，因为有这么多测试数据集
    # 测试数据提取
    id_dict=get_id(data_choice,n_piece,dim)
    # print(id_dict)
    search_id=[]
    search_distance=[]
    for i in range(n_piece):
        data_key=data_name+'_n'+str(n_piece)+'_'+str(i+1)
        keys='npiece_'+str(i+1)
        print(f'正在处理数据集{data_key}')
        file_path = f"{folder_path}/{data_key}.index"
        index = faiss.read_index(file_path)
        sd,sid=index.search(golve_test, k)
        id_real=id_dict[keys][sid]
        search_id.append(id_real)
        search_distance.append(sd)
    search_id=np.array(search_id)
    search_distance=np.array(search_distance)
    # 现在的数据维度是(n_piece,number,k),我们需要转换成(number,k*n_piece),然后实现对后面两个维度的拼接
    # np.concatenate(arr_3d, axis=1) 会沿着第二个维度（axis=1）将每个子数组拼接在一起，得到你想要的结果。
    # 这里不能用reshape
    # 用这种方法也可以np.transpose(arr_3d, (1, 0, 2)).reshape(arr_3d.shape[1],-1)，
        # 这样的话可以for循环遍历number，然后剩下的维度可以交给二维combine_list函数处理
    search_id=np.concatenate(search_id, axis=1)
    search_distance=np.concatenate(search_distance, axis=1)
    # return search_id,search_distance
    # 然后进行排序操作
    test_id=[]
    test_distance=[]
    for i in range(number):
        # print(f'正在处理第{i}个测试数据')
        distance_list,id_list = combine_list(search_distance[i], search_id[i],k)
        test_id.append(id_list) 
        test_distance.append(distance_list)
    test_id=np.array(test_id)
    test_distance=np.array(test_distance)
    return test_id,test_distance

# 计算召回率
def calculate_recall_np(test_id, ground_truth_id):
    # 计算 True Positives，计算交集
    tp = len(np.intersect1d(test_id, ground_truth_id))
    # 这里不用计算False Positives，只需要求出TP后，除以ground_truth_id的长度即可
    # 计算召回率
    recall = tp / len(ground_truth_id) if len(ground_truth_id) != 0 else 0.0
    return recall

def get_recall(data_choice:str='glove',n_piece:int=5,dim:int=25,k:int=100,number:int=100,faiss_style:str='HNSW64'):
    # glove_test,distance_true和neighbors_true暂时没用到
    _,neighbors_true,_=get_test_data(data_choice,dim,number)
    start_time = time.time()
    test_id,_=get_search_result(data_choice,n_piece,dim,k,number,faiss_style)
    end_time = time.time()
    elapsed_time = end_time - start_time # 计算查询所需的时间
    # print(test_id.shape,neighbors_true.shape)
    recall_list=[]
    for i in range(number):
        recall=calculate_recall_np(test_id[i],neighbors_true[i])
        recall_list.append(recall)
    return np.mean(recall_list),elapsed_time

if __name__ == "__main__":
    '''
    查询需要传入的参数：
    data_choice:数据集的选择，glove或者sift
    n_piece:数据集的分片数量，这里需要与create_faiss.py中的n_piece一致
    dim:数据集的维度，glove只能是25,50,100,200，sift只能是128

    前三个参数都是为了构造正确的地址，来读取index数据
    ------------------------------------------------------------------
    faiss_style:str='HNSW64'
    k:int,  这里是查询的k个最近邻即topk,默认值为100
    number:int, 这里是选取的测试数据的数量，默认值为100。number不能大于测试数据的总量，glove和sift都是10,000    
    '''
    rm,et=get_recall(data_choice='glove',n_piece=5,dim=25,k=100,number=100,faiss_style='HNSW64')
    print(f'召回率为{rm},查询时间为{et}秒')
    # get_recall('glove',5,25,100,100) 
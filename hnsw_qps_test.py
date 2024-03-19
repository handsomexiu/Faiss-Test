# 专门用来测试hnsw的qps的
import h5py
import numpy as np
import faiss
import os
import time
from config import *
import matplotlib.pyplot as plt
import seaborn as sns
# 如果要用data_expand_info中的数据集，需要修改这里的data_info为data_expand_info,
# 即data_name= data_info[data_choice][dim]修改成：data_name= data_expand_info[data_choice][dim]
# 然后将获取数据的地址（f"data/{data_name}.hdf5"）修改成：f"data_expand/{data_name}.hdf5"

# 单个数据处理
def get_test_data_QPS(data_choice:str='glove',dim:int=25,number:int=100,k:int=10):#这里定义一个number是为了选取测试集数量，默认是100个
    data_name= data_expand_info[data_choice][dim] # 这里是为了方便创建文件夹和数据库文件名，方便识别
    glove_file_path = f"data_expand/{data_name}.hdf5"
    glove_hdf = h5py.File(glove_file_path, "r")# 读取数据
    glove_test = glove_hdf['test'][:number]# 获取测试查询数据
    if data_choice=='glove':
        faiss.normalize_L2(glove_test)# glove数据采用的angular距离，首先需要进行归一化,然后再进行faiss.METRIC_L2
    glove_neighbors = glove_hdf['neighbors'][:number,:k]# 获取测试数据集
    glove_distances = glove_hdf['distances'][:number,:k]# 获取测试数据集
    return glove_test,glove_neighbors,glove_distances

# 数据分片
def data_piece(data_choice:str='glove',n_piece:int=5,dim:int=25,M:int=16,efConstruction:int=500)->dict:# 如果data_choice是glove则dim必须是25,50,100,200，如果是sift则dim必须是128，否则会报错
    n_piece= n_piece#这里是用来定义切片的数量
    data_name= data_expand_info[data_choice][dim] # 这里是为了方便创建文件夹和数据库文件名，方便识别
    glove_file_path = f"data_expand/{data_name}.hdf5"# 数据地址，hdf5格式
    glove_hdf = h5py.File(glove_file_path, "r")# 读取数据
    length_all = len(glove_hdf['train'])# 获取数据的总长度  
    cut_point=int(length_all/n_piece)# 获取切分点,int是向下取整
    data_dict={}# 这里用字典来存取每一段切分点
    for i in range(n_piece):
        data_key=data_name+'_M'+str(M)+'_efcon'+str(efConstruction)+'_n'+str(n_piece)+'_'+str(i+1)
        if i+1 == n_piece:
            data_dict[data_key]=glove_hdf['train'][i*cut_point:]# 把剩下的余数全部纳入不在单独考虑了（如果能整除则恰好每一块长度相同，如果不能整除则最后一段的长度=cut_point+余数），这里是最后一段，
        else:
            data_dict[data_key]=glove_hdf['train'][i*cut_point:(i+1)*cut_point]
    return data_dict,data_name

# 获取对应的索引
def get_id(data_choice:str='glove',n_piece:int=5,dim:int=25):
    data_name= data_expand_info[data_choice][dim]
    glove_file_path = f"data_expand/{data_name}.hdf5"# 数据地址，hdf5格式
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

# 创建存储数据的文件
def create_index_folder_choice(data_name:str,n_piece:int=5,index_style:str='HNSW',M:int=16,efConstruction:int=500)->str:#这里的n_piece需要与data_piece函数中的n_piece一致
    folder_path = f"index_{index_style}_QPS/{data_name}_n{n_piece}_M{M}_efcon{efConstruction}"
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        # 不存在时创建文件夹
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 不存在，已创建。")
    else:
        print(f"文件夹 '{folder_path}' 已经存在。")
    return folder_path

# 那我们就定义几个构造的函数，不同类型的向量数据库有不同选择方式，这个通过if条件去判断就行了
def create_index_HNSW_QPS(data_choice:str='glove',n_piece:int=5,dim:int=25,M:int=16,efConstruction:int=500):
    data_dict,data_name=data_piece(data_choice,n_piece,dim,M,efConstruction)
    folder_path = create_index_folder_choice(data_name,n_piece,'HNSW',M,efConstruction)
    for i in range(n_piece):
        data_key=data_name+'_M'+str(M)+'_efcon'+str(efConstruction)+'_n'+str(n_piece)+'_'+str(i+1)
        file_path = f"{folder_path}/{data_key}.index"
        index_factory_str = f"HNSW{M}"  # 设置 HNSW 参数,HNSW64就是M=64
        dim = dim
        measure=faiss.METRIC_L2# 这里固定因为glove和sift都用的是L2距离
        index = faiss.index_factory(dim, index_factory_str, measure)  
        index.hnsw.efConstruction = efConstruction
        # 导入数据
        df=data_dict[data_key]
        if data_choice=='glove':
            print(f'现在处理的glove数据集{data_key},需要进行归一化处理')
            faiss.normalize_L2(df)
        index.add(df)
        faiss.write_index(index, file_path)
        print(f"{data_key}.index 已创建。")

# 多个数据之间的合并
def combine_list(data: list, ids: list, k: int = 10):
    # 将每个元素和对应ID放在一个元组中
    combined_tuples = [(element, id_val) for element, id_val in zip(data, ids)]
    # 对拼接后的元组列表按照元素排序，sorted是从小到大排序的，正好取前k个即可
    sorted_combined_tuples = sorted(combined_tuples, key=lambda x: x[0])
    # 输出排序后的元素和对应的ID列表
    sorted_elements = [element[0] for element in sorted_combined_tuples]
    sorted_ids = [element[1] for element in sorted_combined_tuples]
    return sorted_elements[:k], sorted_ids[:k]

# 获取搜索结果
def get_search_result_HNSW_QPS(data_choice:str='glove',n_piece:int=5,dim:int=25,M:int=16,efConstruction:int=500,efsearch:int=10,k:int=10,number:int=100):# number是指定测试集的数量,要和get_test_data函数中的number一致
    # data_dict,data_name=data_piece(data_choice,n_piece,dim)
    data_name= data_expand_info[data_choice][dim] 
    folder_path = create_index_folder_choice(data_name,n_piece,'HNSW',M,efConstruction)
    golve_test,_,_= get_test_data_QPS(data_choice,dim,number,k)# 这是一个双层数组，因为有这么多测试数据集
    # 测试数据提取
    search_id=[]
    search_distance=[]
    id_dict=get_id(data_choice,n_piece,dim)
    for i in range(n_piece):
        data_key=data_name+'_M'+str(M)+'_efcon'+str(efConstruction)+'_n'+str(n_piece)+'_'+str(i+1)
        keys='npiece_'+str(i+1)
        print(f'正在处理数据集{data_key}')
        file_path = f"{folder_path}/{data_key}.index"
        index = faiss.read_index(file_path)
        # 设置搜索参数
        index.hnsw.efSearch = efsearch
        sd,sid=index.search(golve_test, k)
        id_real=id_dict[keys][sid]
        search_id.append(id_real)
        search_distance.append(sd)
    search_id=np.array(search_id)
    search_distance=np.array(search_distance)
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
    tp = len(np.intersect1d(test_id, ground_truth_id))
    recall = tp / len(ground_truth_id) if len(ground_truth_id) != 0 else 0.0
    return recall

# 计算QPS和召回率
def get_QPS(data_choice:str='glove',n_piece:int=5,dim:int=25,M:int=16,efConstruction:int=500,efsearch:int=10,k:int=10,number:int=100):
    # glove_test,distance_true和neighbors_true暂时没用到
    _,neighbors_true,_=get_test_data_QPS(data_choice,dim,number,k)# 获取真实id
    start_time = time.time()
    test_id,_=get_search_result_HNSW_QPS(data_choice,n_piece,dim,M,efConstruction,efsearch,k,number)
    end_time = time.time()
    elapsed_time = end_time - start_time  # 计算查询所需的时间
    # print(test_id.shape,neighbors_true.shape)
    recall_list=[]
    for i in range(number):
        recall=calculate_recall_np(test_id[i],neighbors_true[i])
        recall_list.append(recall)
    mean_reacall=np.mean(recall_list)
    qps=number/elapsed_time
    return mean_reacall,qps

# 多组实验----------------------------------------
# 我们首先固定其他所有的，就调节efsearch，然后计算QPS
# 数据库起名字:data_key=data_name+'_M'+str(M)+'_efcon'+str(efConstruction)+'_n'+str(n_piece)+'_'+str(i+1)
# glove-25-angular_n5_M16_efcon500
# glove-25-angular_M16_efcon500_n5_1.index
# efsearch可以取一组列表：[1,3,5,7,10,15,20,30,40,60,80,100,500]

# 创建多组实验所采用的数据集
def get_multiple_data(data_choice:str='glove',dim:int=25,n_piece:list=[5],M:list=[16,64],efConstruction:list=[100,500]):
    npiece=n_piece
    M = M
    efConstruction=efConstruction
    for i in npiece:
        for j in M:
            for e in efConstruction:
                print(i,j,e)
                create_index_HNSW_QPS(data_choice,n_piece=i,dim=dim,M=j,efConstruction=e)

# 进行多组实验：输出一个字典，key是data_key，value是对应的recall和qps
def get_multiple_QPS(data_choice:str='glove',dim:int=25,n_piece:list=[5],M:list=[16,64],efConstruction:list=[100,500],efsearch:list=[1,3,5,7,10,15,20,30,40,60,80,100,500],k:int=10,number:int=1000):
    npiece=n_piece
    M = M
    efConstruction=efConstruction
    efsearch=efsearch
    result_dict={}
    for i in npiece:
        for j in M:
            for e in efConstruction:
                data_name= data_expand_info[data_choice][dim] 
                data_key=data_name+'_M'+str(j)+'_efcon'+str(e)+'_n'+str(i)
                result_dict[data_key]={}
                qps=[]
                recall=[]
                for l in efsearch:
                    print(i,j,e,l)
                    mean_reacall,qps_=get_QPS(data_choice,n_piece=i,dim=dim,M=j,efConstruction=e,efsearch=l,k=k,number=number)
                    qps.append(qps_)
                    recall.append(mean_reacall)
                result_dict[data_key]['recall']=np.array(recall)
                result_dict[data_key]['qps']=np.array(qps)
    return result_dict

# 可视化-------------------------------------------
# 创建存储图片所需要的文件夹
def create_photo_store_qps(data_choice:str='glove',dim:int=25,faiss_style:str='HNSW')->str:
    data_name= data_expand_info[data_choice][dim]
    folder_path = f"figure/{data_name}_{faiss_style}_QPS"
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        # 不存在时创建文件夹
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 不存在，已创建。")
    else:
        print(f"文件夹 '{folder_path}' 已经存在。")
    return folder_path

def plot_multiple_lines_qps( data_dict: dict, data_choice: str = 'glove', dim: int = 25,faiss_style:str='HNSW',k:int=10):
    # 创建文件夹并确定文件路径
    data_name= data_expand_info[data_choice][dim] 
    folder_path = create_photo_store_qps(data_choice, dim,faiss_style)
    file_path = f"{folder_path}/{faiss_style}_k{k}_multiple_lines.png"
    num_lines = len(data_dict)
    # 创建图表对象
    plt.figure(figsize=(8, 5))
    # 使用seaborn颜色板
    colors = sns.color_palette("husl", num_lines)
    # 遍历每条折线图
    for i, (key, inner_dict) in enumerate(data_dict.items()):
        # 提取数据
        x_values = inner_dict['recall']
        y_values = inner_dict['qps']
        # 选择不同颜色，可以根据需要修改颜色
        # line_color = plt.cm.viridis(i / num_lines)
        line_color = colors[i]
        # 在同一个图中绘制多条折线图，并设置标签位置在图外
        sub_label=key
        plt.plot(x_values, y_values, marker='o', label=sub_label, color=line_color)
    # 添加标签和标题
    set_xlabel, set_ylabel = 'Recall', 'query per second(1/s)'
    plt.xlabel(set_xlabel)
    plt.ylabel(set_ylabel)
    plt_title=data_name+'-'+faiss_style+'-k'+str(k)+'-QPS'
    plt.title(plt_title)

    # 添加图例，并设置位置在图外
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # 保存图片
    plt.savefig(file_path, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    '''
    这里分三步进行：创建多组实验的数据集：
    1. 通过npiece，M，efConstruction来进行控制，每个参数类型都是一个列表
    2. 创建完数据集后，通过get_multiple_QPS函数来控制efsearch得到每组实验的数据
    3. 最后通过plot_multiple_lines_qps函数来进行可视化
    
    命名规范：
    总文件夹：index_HNSW_QPS
    每个数据的文件夹：glove-25-angular_n5_M16_efcon500
    每个数据的文件：glove-25-angular_M16_efcon500_n5_1.index
    存储图片的文件夹：figure/glove-25-angular_HNSW_QPS
    存储图片的文件：HNSW_QPS_multiple_lines.png

    如果要用data_expand_info中的数据集，需要修改这里的data_info为data_expand_info,
    即data_name= data_info[data_choice][dim]修改成：data_name= data_expand_info[data_choice][dim]
    然后将获取数据的地址（f"data/{data_name}.hdf5"）修改成：f"data_expand/{data_name}.hdf5"

    如果命名不规范可能会导致查询不到对应的数据而报错
    '''
    # 创建多组实验所采用的数据集
    # get_multiple_data(data_choice='glove',dim=25,n_piece=[5,10],M=[4,16,32],efConstruction=[100,300,500])

    # k取10，100，1000，10000，100000进行测试
    k=100000
    # 进行多组实验：输出一个双层字典，key是data_key，value是对应的recall和qps字典
    result_dict=get_multiple_QPS(data_choice='glove',dim=25,n_piece=[5,10],M=[4,16,32],efConstruction=[100,300,500],k=k)
    # result_dict=get_multiple_QPS(data_choice='glove',dim=25,n_piece=[5],M=[4],efConstruction=[300],efsearch=[1,3,5,7,10,15,20,30,40,60,80,100,500],k=k,number=1000)

    # 可视化
    plot_multiple_lines_qps(result_dict, data_choice='glove', dim=25,faiss_style='HNSW',k=k)
# 专门用来测试IVFFLat的qps的
# 这里关键是找到对应的参数
# ivfPQR的创建参数：nlist，M，nbits,M_refine,nbits_refine(参数有点多，后面for循环的时候考虑用笛卡尔积)；
# 搜索参数nprobe
# 其中M需要被数据维数d整除，考虑到glove25是25维度的，因此M取5
# mrefine需要被d整除，因此mrefine取5
# nbits取值为[8,12,16]
import h5py
import numpy as np
import faiss
import os
import time
from itertools import product
from config import *
import matplotlib.pyplot as plt

# 单个数据处理
def get_test_data_QPS(data_choice:str='glove',dim:int=25,number:int=100):#这里定义一个number是为了选取测试集数量，默认是100个
    data_name= data_info[data_choice][dim] # 这里是为了方便创建文件夹和数据库文件名，方便识别
    glove_file_path = f"data/{data_name}.hdf5"
    glove_hdf = h5py.File(glove_file_path, "r")# 读取数据
    glove_test = glove_hdf['test'][:number]# 获取测试查询数据
    if data_choice=='glove':
        faiss.normalize_L2(glove_test)# glove数据采用的angular距离，首先需要进行归一化,然后再进行faiss.METRIC_L2
    glove_neighbors = glove_hdf['neighbors'][:number,:10]# 获取测试数据集
    glove_distances = glove_hdf['distances'][:number,:10]# 获取测试数据集
    return glove_test,glove_neighbors,glove_distances

# 数据分片
def data_piece(data_choice:str='glove',n_piece:int=5,dim:int=25,nlist:int=100,M:int=5,nbits:int=8,
               M_refine:int=5,nbits_refine:int=8)->dict:
    n_piece= n_piece#这里是用来定义切片的数量
    data_name= data_info[data_choice][dim] # 这里是为了方便创建文件夹和数据库文件名，方便识别
    glove_file_path = f"data/{data_name}.hdf5"# 数据地址，hdf5格式
    glove_hdf = h5py.File(glove_file_path, "r")# 读取数据
    length_all = len(glove_hdf['train'])# 获取数据的总长度  
    cut_point=int(length_all/n_piece)# 获取切分点,int是向下取整
    data_dict={}# 这里用字典来存取每一段切分点
    for i in range(n_piece):
        data_key=data_name+'_nlist'+str(nlist)+'_M'+str(M)+'_nbits'+str(nbits)+'_MR'+str(M_refine)+'_nbitsR'+str(nbits_refine)+'_n'+str(n_piece)+'_'+str(i+1)
        if i+1 == n_piece:
            data_dict[data_key]=glove_hdf['train'][i*cut_point:]# 把剩下的余数全部纳入不在单独考虑了（如果能整除则恰好每一块长度相同，如果不能整除则最后一段的长度=cut_point+余数），这里是最后一段，
        else:
            data_dict[data_key]=glove_hdf['train'][i*cut_point:(i+1)*cut_point]
    return data_dict,data_name

# 获取对应的索引
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

# 创建存储数据的文件
def create_index_folder_choice(data_name:str,n_piece:int=5,index_style:str='IVFPQR',nlist:int=100,M:int=5,
                               nbits:int=8,M_refine:int=5,nbits_refine:int=8)->str:
    folder_path = f"index_{index_style}_QPS/{data_name}_n{n_piece}_nlist{nlist}_M{M}_nbits{nbits}_MR{M_refine}_nbitsR{nbits_refine}"
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        # 不存在时创建文件夹
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 不存在，已创建。")
    else:
        print(f"文件夹 '{folder_path}' 已经存在。")
    return folder_path

# 那我们就定义几个构造的函数，不同类型的向量数据库有不同选择方式，这个通过if条件去判断就行了
def create_index_IVFPQR_QPS(data_choice:str='glove',n_piece:int=5,dim:int=25,nlist:int=100,M:int=5
                           ,nbits:int=8,M_refine:int=5,nbits_refine:int=8):
    data_dict,data_name=data_piece(data_choice,n_piece,dim,nlist,M,nbits,M_refine,nbits_refine)
    folder_path = create_index_folder_choice(data_name,n_piece,'IVFPQR',nlist,M,nbits)
    for i in range(n_piece):
        data_key=data_name+'_nlist'+str(nlist)+'_M'+str(M)+'_nbits'+str(nbits)+'_MR'+str(M_refine)+'_nbitsR'+str(nbits_refine)+'_n'+str(n_piece)+'_'+str(i+1)
        file_path = f"{folder_path}/{data_key}.index"
        # index_factory_str = f"IVF{nlist},PQ{M}x{nbits}"  
        # IVFPQR 不用index_factory，直接用IndexIVFPQR来创建
        # 因为index_factory无法改变nbits和nbits_refine，只能用默认值：8
        dim = dim
        # measure=faiss.METRIC_L2# 这里固定因为glove和sift都用的是L2距离
        # index = faiss.index_factory(dim, index_factory_str, measure)  
        quantizer = faiss.IndexFlatL2(dim)
        index=faiss.IndexIVFPQR(quantizer, dim, nlist, M, nbits, M_refine, nbits_refine)
        df=data_dict[data_key]
        if data_choice=='glove':
            print(f'现在处理的glove数据集{data_key},需要进行归一化处理')
            faiss.normalize_L2(df)
        if not index.is_trained: # 输出为True，代表该类index不需要训练，只需要add向量进去即可
            index.train(df)
            print(f"{data_key}.index 已经训练过了。")
        # 导入数据
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
def get_search_result_IVFPQR_QPS(data_choice:str='glove',n_piece:int=5,dim:int=25,nlist:int=100,M:int=5,nbits:int=8,M_refine:int=5,nbits_refine:int=8,
                                nprobe:int=1,k:int=10,number:int=100):
    # data_dict,data_name=data_piece(data_choice,n_piece,dim)
    data_name= data_info[data_choice][dim] 
    folder_path = create_index_folder_choice(data_name,n_piece,'IVFPQR',nlist,M,nbits,M_refine,nbits_refine)
    golve_test,_,_=get_test_data_QPS(data_choice,dim,number)# 这是一个双层数组，因为有这么多测试数据集
    # 测试数据提取
    search_id=[]
    search_distance=[]
    id_dict=get_id(data_choice,n_piece,dim)
    for i in range(n_piece):
        data_key=data_name+'_nlist'+str(nlist)+'_M'+str(M)+'_nbits'+str(nbits)+'_MR'+str(M_refine)+'_nbitsR'+str(nbits_refine)+'_n'+str(n_piece)+'_'+str(i+1)
        keys='npiece_'+str(i+1)
        print(f'正在处理数据集{data_key}')
        file_path = f"{folder_path}/{data_key}.index"
        index = faiss.read_index(file_path)
        # 设置搜索参数
        index.nprobe = nprobe
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
def get_QPS(data_choice:str='glove',n_piece:int=5,dim:int=25,nlist:int=100,M:int=5,nbits:int=8,
            M_refine:int=5,nbits_refine:int=8,nprobe:int=1,k:int=10,number:int=100):
    # glove_test,distance_true和neighbors_true暂时没用到
    _,neighbors_true,_=get_test_data_QPS(data_choice,dim,number)# 获取真实id
    start_time = time.time()
    test_id,_=get_search_result_IVFPQR_QPS(data_choice,n_piece,dim,nlist,M,nbits,M_refine,nbits_refine,nprobe,k,number)
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
# 这里我们通过改变查询参数nprobe来控制召回率
# 构造参数是：nlist,M,nbits,M_refine,nbits_refine

# 创建多组实验所采用的数据集
# 这里参数一多的话循环次数就多起来了，可以考虑用笛卡尔乘积的形式来存储一组实验参数
def get_multiple_data(data_choice:str='glove',dim:int=25,n_piece:list=[5],nlist:list=[128,512],M:list=[5],nbits:list=[8,12,16],
                      M_refine:list=[5],nbits_refine:list=[8,12,16]):
    # 计算笛卡尔积
    # 列表中的每个元组中的元素分别是：n_piece,nlist,M,nbits,M_refine,nbits_refine
    cartesian_product = list(product(n_piece,nlist,M,nbits,M_refine,nbits_refine))
    for i,j,m,n,mr,nr in cartesian_product:
        print(i,j,m,n,mr,nr)
        create_index_IVFPQR_QPS(data_choice,n_piece=i,dim=dim,nlist=j,M=m,nbits=n,M_refine=mr,nbits_refine=nr)


# 进行多组实验：输出一个字典，key是data_key，value是对应的recall和qps
def get_multiple_QPS(data_choice:str='glove',dim:int=25,n_piece:list=[5],nlist: list=[128,512],M:list=[5],
                     nbits:list=[8,12,16], M_refine:list=[5],nbits_refine:list=[8,12,16],nprobe:list=[1,2,4,6,8,10,15,20,50,100],k:int=10,number:int=1000):
    # 计算笛卡尔积
    # 列表中的每个元组中的元素分别是：n_piece,nlist,M,nbits,M_refine,nbits_refine
    cartesian_product = list(product(n_piece,nlist,M,nbits,M_refine,nbits_refine))
    result_dict={}
    for i,j,m,n,mr,nr in cartesian_product:
        data_name= data_info[data_choice][dim] 
        data_key=data_name+'_nlist'+str(j)+'_M'+str(m)+'_nbits'+str(n)+'_MR'+str(mr)+'_nbitsR'+str(nr)+'_n'+str(i)
        result_dict[data_key]={}
        qps=[]
        recall=[]
        for l in nprobe:
            mean_reacall,qps_=get_QPS(data_choice,n_piece=i,dim=dim,nlist=j,M=m,nbits=n,M_refine=mr,nbits_refine=nr,nprobe=l,k=k,number=number)
            qps.append(qps_)
            recall.append(mean_reacall)
        result_dict[data_key]['recall']=np.array(recall)
        result_dict[data_key]['qps']=np.array(qps)
    return result_dict

# 可视化-------------------------------------------
# 创建存储图片所需要的文件夹
def create_photo_store_qps(data_choice:str='glove',dim:int=25,faiss_style:str='IVFPQR')->str:
    data_name= data_info[data_choice][dim]
    folder_path = f"figure/{data_name}_{faiss_style}_QPS"
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        # 不存在时创建文件夹
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 不存在，已创建。")
    else:
        print(f"文件夹 '{folder_path}' 已经存在。")
    return folder_path

def plot_multiple_lines_qps( data_dict: dict, data_choice: str = 'glove', dim: int = 25,faiss_style:str='IVFPQR'):
    # 创建文件夹并确定文件路径
    data_name= data_info[data_choice][dim] 
    folder_path = create_photo_store_qps(data_choice, dim,faiss_style)
    file_path = f"{folder_path}/{faiss_style}_multiple_lines.png"
    num_lines = len(data_dict)
    # 创建图表对象
    plt.figure(figsize=(8, 5))
    # 遍历每条折线图
    for i, (key, inner_dict) in enumerate(data_dict.items()):
        # 提取数据
        x_values = inner_dict['recall']
        y_values = inner_dict['qps']
        # 选择不同颜色，可以根据需要修改颜色
        line_color = plt.cm.viridis(i / num_lines)
        # 在同一个图中绘制多条折线图，并设置标签位置在图外
        sub_label=key
        plt.plot(x_values, y_values, marker='o', label=sub_label, color=line_color)
    # 添加标签和标题
    set_xlabel, set_ylabel = 'Recall', 'query per second(1/s)'
    plt.xlabel(set_xlabel)
    plt.ylabel(set_ylabel)
    plt_title=data_name+'-'+faiss_style+'-QPS'
    plt.title(plt_title)

    # 添加图例，并设置位置在图外
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # 保存图片
    plt.savefig(file_path, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    '''
    这里分三步进行：创建多组实验的数据集：
    1. 通过npiece,nlist,M,nbits,M_refine来进行控制，每个参数类型都是一个列表
    2. 创建完数据集后，通过get_multiple_QPS函数来控制nprobe得到每组实验的数据
    3. 最后通过plot_multiple_lines_qps函数来进行可视化
    
    这里如果nbit和nbit_refine都是默认值8，则可以用index_factory来创建索引
        对应的string：f'IVF{nlist},PQ{M}+{M_refine}'

    这里的维度d不仅要整除m还必须上整除m_refine

    命名规范：
    总文件夹：index_IVFPQR_QPS
    每个数据的文件夹：glove-25-angular_n5_nlist128_M5_nbits8_MR5_nbitsR8
    每个数据的文件：glove-25-angular_nlist128_M5_nbits8_MR5_nbitsR8_n5_1.index
    存储图片的文件夹：figure/glove-25-angular_IVFPQR_QPS
    存储图片的文件：IVFPQR_QPS_multiple_lines.png

    如果命名不规范可能会导致查询不到对应的数据而报错
    '''
    # 创建多组实验所采用的数据集
    get_multiple_data(data_choice='glove',dim=25,n_piece=[5],nlist=[128,256,512],M=[5],nbits=[8],M_refine=[5],nbits_refine=[8])

    # 进行多组实验：输出一个双层字典，key是data_key，value是对应的recall和qps字典
    result_dict=get_multiple_QPS(data_choice='glove',dim=25,n_piece=[5],nlist=[128,256,512],M=[5],nbits=[8],M_refine=[5],nbits_refine=[8])
    # ,nprobe=[1,2,4,6,8,10,15,20,50,100],k=10,number=100可以省略，默认就可以
    # 可视化
    plot_multiple_lines_qps(result_dict, data_choice='glove', dim=25,faiss_style='IVFPQR')
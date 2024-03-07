# 介绍

这是关于faiss向量数据库测试的相关代码，使用的数据集是glove-25和sift-128,数据来源[erikbern/ann-benchmarks: Benchmarks of approximate nearest neighbor libraries in Python (github.com)](https://github.com/erikbern/ann-benchmarks)

支持的测试内容：

- [x] 支持HNSW64,其余的还没测试
- [ ] 支持创建faiss数据库的不同参数测试
- [x] 创建不同分片向量数据库的时间对比
- [x] 不同切片数据库在不同k（top k）值下的查询时间和召回率对比
- [x] 对比不同query数量下的查询时间和召回率对比
- [x] 对比不同分片数据库下，不同query数量下的查询时间和召回率对比
- [ ] QPS-Recall测试



# 使用说明

1. 安装相关环境，`conda create -n faiss-test python=3.9`,`pip install -r requirements.txt`

2. 如果仅使用HNSW 64，直接在faiss-test环境下运行`python faiss_get_report.py`

3. 如果想要修改创建数据库的类型，可以修改`faiss_get_report.py`下的

   1. ![image-20240307100900991](photo\change_faiss.png)

4. 创建数据库的相关文件是`faiss_database_create.py`创建的方式采用的是`faiss.index_factory(dim, param, measure)`具体可看

   1. [The index factory · facebookresearch/faiss Wiki (github.com)](https://github.com/facebookresearch/faiss/wiki/The-index-factory)

5. `faiss_test.py`主要是用来获取Recall，可以看数据是如何召回的

   1. 如果想要修改数据集，注意需要修改`faiss_get_report.py`下的`data_choice`和`dim`参数，这两个参数要与`config.py`中的字典`data_info`中信息匹配

   2. ![image-20240307101537924](photo\data_choice.png)

      ```python
      data_info={
          'glove':{25:'glove-25-angular',
                 50: 'glove-50-angular',
                 100: 'glove-100-angular',
                 200: 'glove-200-angular'},
          'sift':{128:'sift-128-euclidean'}
      }
      ```

      

6. 最后输出的结果是一组图片，存放在figure文件夹中，index文件夹存的是向量数据库

7. 这里还有关于faiss的一些个人学习资料，具体可看同文件夹下的：`Faiss-HNSW.md`


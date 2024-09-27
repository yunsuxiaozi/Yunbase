## Yunbase

在打数据挖掘的算法比赛中,有很多操作是每场比赛都要做的,从数据预处理到k折交叉验证,这些操作中有很多是重复的。每次都写重复的代码有点麻烦,所以我这里提取了这些操作中共性的部分,写了Yunbase这个类。(Yun取我的网名匀速小子的第一个字,base就是作为算法比赛的baseline)

### 快速上手

1.克隆项目到本地

> git clone https://github.com/yunsuxiaozi/Yunbase.git

2.导入Yunbase

>  from Yunbase.baseline import Yunbase

3.创建Yunbase类

> ```python
> yunbase=Yunbase(num_folds=5,
>                       models=[],
>                       FE=None,
>                       seed=2024,
>                       objective='regression',
>                       metric='rmse',
>                       nan_margin=0.95,
>                       group_col='p_num',
>                       target_col='bg+1:00',
>                )
> ```

- num_folds:k折交叉验证的折数
- models和FE是给用户灵活使用的。FE是一个特征工程的函数,你可以定义自己的特征工程,函数的使用方法形如:df=FE(df),models可以存储你自己的模型,例如:[(LGBMRegressor(**lgb_params),'lgb')]
- seed:是随机种子
- objective是任务类型,目前有'binary'(二分类),'multi_class'(多分类)和'regression'(回归)。
- metric:评估指标,目前只支持rmse,mse和accuracy.
- nan_margin:就是表格数据中某列缺失值大于多少,我们选择丢掉这列。
- group_col是针对groupkfold而设计的,如果一个任务你认为需要用到groupkfold,group_col就是groupkfold的group那列。
- target_col:我们需要预测的那列,也就是标签列。

4.模型的训练

目前支持csv文件的路径,或者已经读取出来的csv文件。

> ```python
> yunbase.fit(train_path_or_file="train.csv")
> ```

5.模型的推理

```python
test_preds=yunbase.predict(test_path_or_file="test.csv")
```

6.预测结果的保存

这里需要读取sample_submission,然后将target_col的值替换成test_preds

```python
yunbase.submit(submission_path='sample_submission.csv',test_preds=test_preds)
```

yunbase使用的参考教程

<a href="https://www.kaggle.com/code/yunsuxiaozi/brist1d-yunbase">yunbase</a>

### 后续工作

代码目前已经完成大致的框架,后续会继续改进,在修正bug的基础上增加新的功能。



 2024/9/27
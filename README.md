## Yunbase,算法比赛的第一次提交

在打数据挖掘的算法比赛中,有很多操作是每场比赛都要做的,从数据预处理到k折交叉验证,这些操作中有很多是重复的。每次都写重复的代码有点麻烦,所以我这里提取了这些操作中共性的部分,写了Yunbase这个类。(Yun取我的网名匀速小子的第一个字,base就是作为算法比赛的baseline)

### 快速上手

1.克隆项目到本地

```python
!git clone https://github.com/yunsuxiaozi/Yunbase.git
```

2.导入Yunbase

```python
from Yunbase.baseline import Yunbase
```

3.创建Yunbase类

```python
yunbase=Yunbase(  num_folds=5,
                  models=[],
                  FE=None,
                  drop_cols=[],
                  seed=2024,
                  objective='regression',
                  metric='mse',
                  nan_margin=0.95,
                  group_col=None,
                  num_classes=None,
                  target_col='target',
                  infer_size=10000,
                  save_oof_preds=True,
                  save_test_preds=True,
                  device='cpu',
                  one_hot_max=50,
                  custom_metric=None,
                  use_optuna_find_params=0,
                  optuna_direction=None,
                  early_stop=100,
                  use_pseudo_label=False,
                  use_high_correlation_feature=True,
            )
```

- num_folds:k折交叉验证的折数
- models和FE是给用户灵活使用的。FE是一个特征工程的函数,除了内置的特征工程外,你可以定义自己的特征工程,函数的使用方法:df=FE(df),models可以存储你自己的模型,例如:[(LGBMRegressor(**lgb_params),'lgb')],如果models=[],会使用内置的模型和参数。
- drop_cols是针对做好特征工程的train和test,如果你想要删除一些列,可以将列名添加进这个参数里。
- seed:是随机种子
- objective是任务类型,目前有'binary'(二分类),'multi_class'(多分类)和'regression'(回归)。
- metric:评估指标,目前只支持一些常用的评估指标。如果你要用的评估指标不支持,可以用custom_metric参数自定义,例如custom_metric(y_true,y_pred),在分类任务中,y_pred.shape=(len(y_true),num_classes)
- nan_margin:就是表格数据中某列缺失值大于多少,我们选择丢掉这列。
- group_col是针对groupkfold而设计的,如果一个任务你认为需要用到groupkfold,group_col就是groupkfold的group那列。
- num_classes:分类任务中需要指定的类别数,回归任务不需要填。
- target_col:我们需要预测的那列,也就是标签列。
- infer_size:在模型推理的时候可能会由于测试数据太大,无法一次性推理完成,所以这里采用batch推理的方法,比如一次推理10000个测试数据。
- save_oof_preds:设置为True或者False,就是要不要保存模型交叉验证的结果,可能有人会想要分析模型对于哪些样本预测的好,哪些样本预测的差,这时候可以保存oof_preds用于研究。
- save_test_preds:这里主要针对的是多分类任务,如果需要的是预测结果的概率,可以设置这个参数为True。
- device为cpu或者gpu,就是模型训练的时候是在cpu环境还是在gpu环境。
- one_hot_max就是设置一个特征列的nunique小于多少的时候会对这列进行onehotencoder.
- custom_metric:自定义评估指标。
- use_optuna_find_params,目前只支持用optuna找lgb模型的参数,设置为0就是不找参数,否则这个参数就是optuna迭代的次数
- optuna_direction,自定义评估指标的时候需要,指定评估指标是越大越好,还是越小越好
- early_stop:设置模型早停的次数。
- use_pseudo_label:是否使用伪标签,这是数据挖掘算法中的一种方法。
- use_high_correlation_feature:是否使用高相关性的特征。如果设置为False,会去找训练数据中高相关性的特征对,并删除其中一个。

4.模型的训练

目前支持csv文件的路径,或者已经读取出来的csv文件。

```python
yunbase.fit(train_path_or_file="train.csv")
```

5.模型的推理

```python
test_preds=yunbase.predict(test_path_or_file="test.csv"，weights=[1,1,1])
```

- weights:根据你设置的models的模型个数(k折交叉验证只算1个)设置模型的权重,如果没设置就是普通的求平均,weights里求和不需要等于1,因为内置了权重归一化。

6.预测结果的保存

这里需要读取sample_submission,然后将target_col列的值替换成test_preds

```python
yunbase.submit(submission_path='sample_submission.csv',test_preds=test_preds,save_name='yunbase')
```

- save_name你可以设置为你自己想保存的名字,例如submission.

训练数据和测试数据的访问

```python
yunbase.train,yunbase.test
```



### yunbase使用的参考教程(Kaggle最新的比赛)

<a href="https://www.kaggle.com/code/yunsuxiaozi/brist1d-yunbase">yunbase</a>



### 后续工作

代码目前已经完成大致的框架,后续会继续改进,在修正bug的基础上增加新的功能。

原则上,bug发现多少就修改多少,新功能想到多少就增加多少。

1.增加repeatkfold。

2.样本的欠采样和过采样。

3.目前只支持dataframe,争取支持train_X,train_y,test_X.

Kaggle账号:https://www.kaggle.com/yunsuxiaozi

 更新时间2024/10/07(baseline.py和README可能不会同步更新)
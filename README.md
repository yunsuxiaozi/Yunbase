# 🚀Yunbase,first submission of your algorithm competition

<img src="yunbase.png" alt="yunbase title image" style="zoom:100%;" />

In the competition of data mining,there are many operations that need to be done in every time.Many of these operations,from data preprocessing to k-fold cross validation,are repetitive.It's a bit troublesome to write repetitive code every time,so I extracted the common parts among these operations and wrote the Yunbase class here.('Yun' is my name <b>yunsuxiaozi</b>,'base' is the baseline of competition)



### Get Started Quickly

1.git clone 

```python
!git clone https://github.com/yunsuxiaozi/Yunbase.git
```

2.download wheel in requirements.txt

```python
!pip download -r Yunbase/requirements.txt
```

3.install according to  requirements.txt

```python
!pip install -q --requirement yourpath/Yunbase/requirements.txt  \
--no-index --find-links file:yourpath
```

4.import Yunbase

```python
from Yunbase.baseline import Yunbase
```

5.create Yunbase.

All the parameters are below, and you can flexibly choose parameters according to the task.

```python
yunbase=Yunbase(num_folds:int=5,
                      models:list[tuple]=[],
                      FE=None,
                      drop_cols:list[str]=[],
                      seed:int=2024,
                      objective:str='regression',
                      metric:str='mse',
                      nan_margin:float=0.95,
                      group_col=None,
                      num_classes=None,
                      target_col:str='target',
                      infer_size:int=10000,
                      save_oof_preds:bool=True,
                      save_test_preds:bool=True,
                      device:str='cpu',
                      one_hot_max:int=50,
                      custom_metric=None,
                      use_optuna_find_params:int=0,
                      optuna_direction=None,
                      early_stop:int=100,
                      use_pseudo_label:bool=False,
                      use_high_corr_feat:bool=True,
                      cross_cols:list[str]=[],
                      labelencoder_cols:list[str]=[],
                      list_cols:list[str]=[],
                      list_gaps:list[int]=[1],
                      word2vec_models:list[tuple]=[],
                      use_svd:bool=False,
                      text_cols:list[str]=[],
                      print_feature_importance:bool=False,
                      log:int=100,
                      exp_mode:bool=False,
                      use_reduce_memory:bool=False,
            )
```

- num_folds:<b>int</b>.the number of folds for k-fold cross validation.

- models:<b>list of models</b>.Built in 3 GBDTs as baseline, you can also use custom models,such as models=[(LGBMRegressor(**lgb_params),'lgb')].
  
- FE:<b>function</b>.In addition to the built-in feature engineer, you can also customize feature engineer.For example:

     ```python
     def FE(df):
         return df.drop(['id'],axis=1)
     ```

     

- drop_cols:<b>list</b>.The column to be deleted after all feature engineering is completed.

- seed:<b>int</b>.random seed.

- objective:<b>str</b>.what task do you want to do?<b>regression</b>,<b>binary</b> or <b>multi_class</b>?

- metric:<b>str</b>.metric to evaluate your model.

- nan_margin:<b>float</b>.when the proportion of missing values in a column is greater than, we delete this column.

- group_col:<b>str</b>.if you want to use groupkfold,then define this group_col.

- num_classes:<b>int</b>.if objectibe is multi_class or binary,you should define this class.

- target_col:<b>str</b>.the column that you want to predict.

- infer_size:<b>int</b>.the test data might be large,we can predict in batches.

- save_oof_preds:<b>bool</b>.you can save OOF for your own offline study.

- save_test_preds:<b>bool</b>.you can save test_preds.For multi classification tasks, the predicted result is the category.If you need to save the probability of the test_data,you can use save_test_preds.                         

- device:<b>str</b>.GBDT can training on GPU,you can set this parameter like NN.

- one_hot_max:<b>int</b>.If the nunique of a column is less than a certain value, perform one hot encoder.

- custom_metric:<b>function</b>.your custom_metric.

     <b>Attention:when objective is multi_class,y_pred in custom_metric(y_true,y_pred) is probability.</b>

- use_optuna_find_params:<b>int</b>.count of use optuna find best params,0 is not use optuna to find params.Currently only LGBM is supported.

- optuna_direction:<b>str</b>.'minimize' or 'maximize',when you use custom metric,you need to define.the direction of optimization.

- early_stop:<b>int</b>.Common parameters of GBDT.

- use_pseudo_label:<b>bool</b>.Whether to use pseudo labels.When it is true,adding the test data to the training data and training again after obtaining the predicted results of the test data.

- use_high_corr_feat:<b>bool</b>.whether to use high correlation features or not. 

- cross_cols:<b>list[str]</b>.Construct features using addition, subtraction, multiplication, and division brute force for these columns of features.

- labelencoder_cols:<b>list</b>.Convert categorical string variables into [1,2,……,n].

- list_cols:<b>list</b>.If the data in a column is a list or str(list), this can be used to extract features.

- list_gaps:<b>list</b>.extract features for list_cols.example=[1,2,4].

- word2vec_models:<b>list</b>.Use models such as tfidf to extract features of string columns.For example:word2vec_models=[(TfidfVectorizer(),col,model_name)].
  
- use_svd:<b>bool</b>.use truncated  singular value decomposition to word2vec features.
  
- text_cols:<b>list</b>.extract features of words, sentences, and paragraphs from text here.

- print_feature_importance:<b>bool</b>.after model training,whether print feature importance or not.

- log:<b>int</b>.log trees are trained in the GBDT model to output a validation set score once.

- exp_mode:<b>bool</b>.In regression tasks, the distribution of target_col is a long tail distribution, and this parameter can be used to perform log transform on the target_col.

- use_reduce_memory:<b>bool</b>.if use function reduce_mem_usage(),then set this parameter True.

6.yunbase training

At present, it supports read csv, parquet files according to path, or csv files that have already been read.

```python
yunbase.fit(train_path_or_file:str|pd.DataFrame|pl.DataFrame='train.csv',
            sample_weight=1,category_cols:list[str]=[],
            target2idx:dict|None=None,
           )
```

- train_path_or_file:You can use the file path or pass in the already loaded file.
- sample_weight:If you want to weight the samples, you can pass in a numpy.array of the same size as the training data.
- category_cols:You can specify which columns to convert to 'category' in the training data.
- target2idx:The dictionary mapped in the classification task, if you want to predict a person's gender, you can specify {'Male ': 0,' Female ': 1}.If you do not specify it yourself, it will be mapped to 0, 1,... n in order of the number of times each target appears.

7.yunbase inference

```python
test_preds=yunbase.predict(test_path_or_file:str|pd.DataFrame|pl.DataFrame='test.csv',weights=None)
```

- weights:This is setting the weights for model ensemble. For example, if you specify lgb, xgb, and cat, you can set weights to [3,4,3].

8.save test_preds to submission.csv

```python
yunbase.submit(submission_path_or_file='submission.csv',test_preds=np.ones(3),save_name='yunbase')
```

- save_name can be changed.if you set  'submission',it will give you a csv file named 'submission.csv'.

9.ensemble

```python
yunbase.ensemble(solution_paths_or_files,weights=None)
```

- For example:

  ```python
  solution_paths_or_files=[
  'submission1.csv',
  'submission2.csv',
  'submission3.csv'
  ]
  weights=[3,3,4]
  ```

10.If train and inference need to be separated.

```python
#model save
yunbase.pickle_dump(yunbase,'yunbase.model')

import dill#serialize and deserialize objects (such as saving and loading tree models)
def pickle_load(path):
    #open path,binary read
    with open(path, mode="rb") as f:
        data = dill.load(f)
        return data
yunbase=Yunbase()
yunbase=pickle_load("yunbase.model")
yunbase.model_save_path=your_model_save_path
```

11.train data and test data can be seen as below.

```python
yunbase.train,yunbase.test
```

##### <a href="https://www.kaggle.com/code/yunsuxiaozi/yunbase">Here</a> is a static version that can be used to play Kaggle competition.You can refer to this <a href="https://www.kaggle.com/code/yunsuxiaozi/brist1d-yunbase">notebook</a> to learn usage of Yunbase. 

## TimeSeries Purged CV

```python
yunbase.purged_cross_validation(
    train_path_or_file:str|pd.DataFrame|pl.DataFrame='train.csv',                             test_path_or_file:str|pd.DataFrame|pl.DataFrame='test.csv',
    date_col:str='date',train_gap_each_fold:int=31,#one month
    train_test_gap:int=7,#a week
    train_date_range:int=0,test_date_range:int=0,
    category_cols:list[str]=[],
    use_seasonal_features:bool=True,
    weight_col:str='weight',
    use_weighted_metric:bool=False,
                           ) 
```

Demo notebook:<a href="https://www.kaggle.com/code/yunsuxiaozi/rsfc-yunbase">Rohlik Yunbase</a>

### follow-up work

The code has now completed a rough framework and will continue to be improved by adding new functions based on bug fixes.

<b>In principle, fix as many bugs as I discover and add as many new features as I think of.</b>

1.add kfold such as <b>repeat</b>kfold.

2.<b>Undersampling, oversampling</b>.

3.fit function to <b>np.array</b>.(such as model.fit(train_X,train_y),model.predict(test_X))

4.add more common <b>metric</b>.

5.In addition to kfold, <b>single model</b> training and inference are also implemented.

6.hill climbing to find <b>blending</b> weight.

Waiting for updates.

Kaggle:https://www.kaggle.com/yunsuxiaozi

 update time:2024/11/20(baseline.py and README may not synchronize updates)
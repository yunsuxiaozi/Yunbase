## Yunbase,First submission of the algorithm competition

In the competition of data mining,there are many operations that need to be done in every time.Many of these operations, from data preprocessing to k-fold cross validation, are repetitive.It's a bit troublesome to write repetitive code every time, so I extracted the common parts among these operations and wrote the Yunbase class here。(Yun is my name,base is the baseline of competition)

### Get Started Quickly

1.clone 

```python
!git clone https://github.com/yunsuxiaozi/Yunbase.git
```

2.import Yunbase

```python
from Yunbase.baseline import Yunbase
```

3.create Yunbase,All the parameters are below, and you can flexibly choose parameters according to the task

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
                  use_high_corr_feat=True,
                  labelencoder_cols=[],
                  list_cols=[],
                  list_gaps=[1],
                  word2vec_models=[],
            )
```

- num_folds:the number of folds for k-fold cross validation.
- models:Built in 3 GBDTs as baseline, you can also use custom models,
                                 such as models=[(LGBMRegressor(**lgb_params),'lgb')]
- FE:In addition to the built-in feature engineer, you can also customize feature engineer.
- drop_cols:The column to be deleted after all feature engineering is completed.
- seed:random seed.
- objective:what task do you want to do?regression,binary or multi_class?
- metric:metric to evaluate your model.
- nan_margin:when the proportion of missing values in a column is greater than, we delete this column.
- group_col:if you want to use groupkfold,then define this group_col.
- num_classes:if objectibe is multi_class,you should define this class.
- target_col:the column that you want to predict.s
- infer_size:the test data might be large,we can predict in batches.
- save_oof_preds:you can save OOF for offline study.
- save_test_preds:you can save test_preds.For multi classification tasks, the predicted result is the category.If you need to save the probability of the test_data,you can save test_preds.                         
- device:GBDT can training on GPU,you can set this parameter like NN.
- one_hot_max:If the nunique of a column is less than a certain value, perform one hot encoder.
- custom_metric:your custom_metric,when objective is multi_class,y_pred in custom(y_true,y_pred) is probability.
- use_optuna_find_params:count of use optuna find best params,0 is not use optuna to find params.Currently only LGBM is supported.
- optuna_direction:'minimize' or 'maximize',when you use custom metric,you need to define.the direction of optimization.
- early_stop:Common parameters of GBDT.
- use_pseudo_label:Whether to use pseudo labels.When it is true,adding the test data to the training data and training again after obtaining the predicted results of the test data.
- use_high_corr_feat:whether to use high correlation features or not. 
- labelencoder_cols:Convert categorical string variables into [1,2,……,n].
- list_cols:If the data in a column is a list or str(list), this can be used to extract features.
- list_gaps:extract features for list_cols.example=[1,2,4].
- word2vec_models:Use models such as tfidf to extract features of string columns 
                                 example:word2vec_models=[(TfidfVectorizer(),col,model_name)]

4.Model training

At present, it supports read CSV, Parquet files, or CSV files that have already been read.

```python
yunbase.fit(train_path_or_file="train.csv")
```

5.Model inference

```python
test_preds=yunbase.predict(test_path_or_file="test.csv"，weights=[1,1,1],load_path='')
```

6.save test_preds to submission.csv

```python
yunbase.submit(submission_path='sample_submission.csv',test_preds=test_preds,save_name='yunbase')
```

- save_name can change,such as submission.

train data and test data can be seen as below.

```python
yunbase.train,yunbase.test
```



### You can refer to this <a href="https://www.kaggle.com/code/yunsuxiaozi/brist1d-yunbase">notebook</a> to learn usage of Yunbase.



### follow-up work

The code has now completed a rough framework and will continue to be improved by adding new features based on bug fixes.

In principle, fix as many bugs as I discover and add as many new features as I think of.

1.add repeatkfold。

2.Undersampling, oversampling, and weighting of samples.

3.model.fit(train_X,train_y),model.predict(test_X) np.array

Kaggle:https://www.kaggle.com/yunsuxiaozi

 update time:2024/10/19(baseline.py and README may not synchronize updates)
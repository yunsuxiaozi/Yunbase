"""
@author:yunsuxiaozi
@start_time:2024/9/27
@update_time:2024/9/27
"""
import polars as pl#和pandas类似,但是处理大型数据集有更好的性能.
import pandas as pd#读取csv文件的库
import numpy as np#对矩阵进行科学计算的库
#这里使用groupkfold
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedGroupKFold,GroupKFold
#model lightgbm回归模型,日志评估
from  lightgbm import LGBMRegressor,LGBMClassifier,log_evaluation,early_stopping
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings#避免一些可以忽略的报错
warnings.filterwarnings('ignore')#filterwarnings()方法是用于设置警告过滤器的方法，它可以控制警告信息的输出方式和级别。
class Yunbase():
    def __init__(self,num_folds=5,
                      models=[],
                      FE=None,
                      seed=2024,
                      objective='regression',
                      metric='mse',
                      nan_margin=0.95,
                      group_col=None,
                      num_classes=None,
                      target_col='target',
                ):
        """
        num_folds:是k折交叉验证的折数
        models:我这里已经设置了基本的模型作为baseline,你也可以使用你自己想使用的模型
        FE:除了我这里已经做了的基本的特征工程以外,你可以自定义你需要的特征工程的函数
        seed:随机种子
        objective:你想做的任务是什么?,regression,binary还是multi_class
        metric:你想使用的评估指标
        nan_margin:一列缺失值大于多少选择不要
        group_col:groupkfold需要有一列作为group
        num_classes:如果是分类任务,需要指定类别数量
        target_col:需要预测的那一列
        """
        self.num_folds=num_folds
        self.seed=seed
        self.models=models
        self.FE=FE
        self.objective=objective
        self.metric=metric
        self.nan_margin=nan_margin
        self.group_col=group_col
        self.target_col=target_col
        self.num_classes=num_classes
        self.pretrained_models={}#用字典的方式保存已经训练好的模型
        
    def get_details(self,):
        #目前支持的评估指标有
        metrics=['rmse','mse','accuracy']
        #目前支持的模型有
        models=['lgb']
        #目前支持的交叉验证方法有
        kfolds=['KFold','GroupKFold','StratifiedKFold','StratifiedGroupKFold']
        #目前支持的任务
        objectives=['binary','multi_class','regression']
        print(f"Currently supported metrics:{metrics}")
        print(f"Currently supported models:{models}")
        print(f"Currently supported kfolds:{kfolds}")
        print(f"Currently supported objectives:{objectives}")
        
    #对训练数据或者测试数据做特征工程,mode='train'或者'test'
    def Feature_Engineer(self,df,mode='train'):
        if self.FE!=None:
            #你想添加的特征工程
            df=self.FE(df)
        if mode=='train':
            #缺失值太多
            self.nan_cols=[col for col in df.columns if df[col].isna().mean()>self.nan_margin]
            #nunique=1
            self.unique_cols=[col for col in df.columns if df[col].nunique()==1]
            #如果一列是object列,那肯定不能放入模型进行学习
            self.object_cols=[col for col in df.columns if (df[col].dtype==object) and (col!=self.group_col)]
            #one_hot_cols
            self.one_hot_cols=[]
            for col in df.columns:
                if col!=self.target_col and col!=self.group_col:
                    if (df[col].nunique()<20) and (df[col].nunique()>2):
                        self.one_hot_cols.append([col,list(df[col].unique())]) 
        for i in range(len(self.one_hot_cols)):
            col,nunique=self.one_hot_cols[i]
            for u in nunique:
                df[f"{col}_{u}"]=(df[col]==u).astype(np.int8)
        
        #去除无用的列
        df.drop(self.nan_cols+self.unique_cols+self.object_cols,axis=1,inplace=True,errors='ignore')
        return df
    
    def Metric(self,y_true,y_pred):
        if self.metric=='rmse':
            return np.sqrt(np.mean((y_true-y_pred)**2))
        if self.metric=='mse':
            return np.mean((y_true-y_pred)**2)
        if self.metric=='accuracy':
            return np.mean(y_true==y_pred)
    def fit(self,train_path_or_file='train.csv'):
        try:#path试试
            self.train=pl.read_csv(train_path_or_file)
            self.train=self.train.to_pandas()
        except:#file
            self.train=train_path_or_file
        #提供的训练数据不是df表格
        if not isinstance(self.train, pd.DataFrame):
            raise ValueError("train_path_or_file is not pd.DataFrame")
        self.train=self.Feature_Engineer(self.train,mode='train')
        
        #二分类,多分类,回归
        if self.objective.lower() not in ['binary','multi_class','regression']:
            raise ValueError("Wrong or currently unsupported objective")
        
        #选择哪种交叉验证方法
        if self.objective.lower()=='binary' or self.objective.lower()=='multi_class':
            if self.group_col!=None:#group
                kf=StratifiedGroupKFold(n_splits=self.num_folds,random_state=self.seed,shuffle=True)
            else:
                kf=StratifiedKFold(n_splits=self.num_folds,random_state=self.seed,shuffle=True)
        else:#回归任务
            if self.group_col!=None:#group
                kf=GroupKFold(n_splits=self.num_folds)
            else:
                kf=KFold(n_splits=self.num_folds,random_state=self.seed,shuffle=True)
                
        #模型的训练,如果你自己准备了模型,那就用你的模型,否则就用我的模型
        if len(self.models)==0:
            metric=self.metric.lower()
            if self.objective.lower()=='multi_class':
                metric='multi_logloss'
            lgb_params={"boosting_type": "gbdt","metric": metric,#"objective": self.objective.lower(),
                        'random_state': self.seed,  "max_depth": 10,"learning_rate": 0.05,
                        "n_estimators": 10000,"colsample_bytree": 0.6,"colsample_bynode": 0.6,"verbose": -1,"reg_alpha": 0.2,
                        "reg_lambda": 5,"extra_trees":True,'num_leaves':64,"max_bin":255,
                        }
            if self.objective.lower()=='regression':
                self.models=[(LGBMRegressor(**lgb_params),'lgb')]
            else:
                self.models=[(LGBMClassifier(**lgb_params),'lgb')]
        
        X=self.train.drop([self.group_col,self.target_col],axis=1,errors='ignore')
        y=self.train[self.target_col]
        
        self.col2name={}
        for i in range(len(list(X.columns))):
            self.col2name[list(X.columns)[i]]=f'col_{i}'
        X=X.rename(columns=self.col2name)
        
        if self.group_col!=None:
            group=self.train[self.group_col]
        else:
            group=None
        for (model,model_name) in self.models:
            oof=np.zeros(len(y))
            for fold, (train_index, valid_index) in (enumerate(kf.split(X,y,group))):
                print(f"name:{model_name},fold:{fold}")

                X_train, X_valid = X.iloc[train_index].reset_index(drop=True), X.iloc[valid_index].reset_index(drop=True)
                y_train, y_valid = y.iloc[train_index].reset_index(drop=True), y.iloc[valid_index].reset_index(drop=True)

                if 'lgb' in model_name:
                    model.fit(X_train,y_train,eval_set=[(X_valid, y_valid)],
                             callbacks=[log_evaluation(100),early_stopping(200)]
                        ) 
                else:
                    model.fit(X_train,y_train,eval_set=[(X_valid, y_valid)],verbose=500) 
                oof[valid_index]=model.predict(X_valid)
                self.pretrained_models[f'{model_name}_fold{fold}']=model
            print(f"{self.metric}:{self.Metric(y.values,oof)}")
        
    def predict(self,test_path_or_file='test.csv'):
        try:#path试试
            self.test=pl.read_csv(test_path_or_file)
            self.test=self.test.to_pandas()
        except:#file
            self.test=test_path_or_file
        #提供的训练数据不是df表格
        if not isinstance(self.test, pd.DataFrame):
            raise ValueError("test_path_or_file is not pd.DataFrame")
        self.test=self.Feature_Engineer(self.test,mode='test')
        self.test=self.test.drop([self.group_col,self.target_col],axis=1,errors='ignore')
        self.test=self.test.rename(columns=self.col2name)
        if self.objective.lower()=='regression':
            test_preds=np.zeros((len(self.models)*self.num_folds,len(self.test)))
            fold=0
            for (model_name,model) in self.pretrained_models.items():
                test_preds[fold]=model.predict(self.test)
                fold+=1
            return test_preds.mean(axis=0)
        else:
            test_preds=np.zeros((len(self.models)*self.num_folds,len(self.test),self.num_classes))
            fold=0
            for (model_name,model) in self.pretrained_models.items():
                test_preds[fold]=model.predict_proba(self.test)
                fold+=1
            return np.argmax(test_preds.mean(axis=0),axis=1)
    def submit(self,submission_path='submission.csv',test_preds=None):
        submission=pd.read_csv(submission_path)
        submission[self.target_col]=test_preds
        submission.to_csv("yunbase.csv",index=None)
        submission.head()

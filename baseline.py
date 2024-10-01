
"""
@author:yunsuxiaozi
@start_time:2024/09/27
@update_time:2024/10/01
"""
import polars as pl#和pandas类似,但是处理大型数据集有更好的性能.
import pandas as pd#读取csv文件的库
import numpy as np#对矩阵进行科学计算的库
#这里使用groupkfold
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedGroupKFold,GroupKFold
from sklearn.metrics import roc_auc_score
#model lightgbm回归模型,日志评估
from  lightgbm import LGBMRegressor,LGBMClassifier,log_evaluation,early_stopping
from catboost import CatBoostRegressor,CatBoostClassifier
from xgboost import XGBRegressor,XGBClassifier
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
                      infer_size=10000,
                      save_oof_preds=True,
                      device='cpu',
                      one_hot_max=50,
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
        infer_size:测试数据可能内存太大,一次推理会出错,所以有了分批次预测.
        save_oof_preds:是否保存交叉验证的结果用于后续的研究
        device:将树模型放在GPU还是CPU上训练.
        one_hot_max:一列特征的nunique少于多少做onehot处理.
        """
        self.num_folds=num_folds
        if self.num_folds<2:
            raise ValueError("num_folds must be greater than 2")
        self.seed=seed
        self.models=models
        self.FE=FE
        
        self.objective=objective.lower()
        if self.objective not in ['binary','regression','multi_class']:
            raise ValueError("objective currently only supports binary、regression、multi_class")
        
        self.metric=metric.lower()
        self.nan_margin=nan_margin
        if self.nan_margin<0 or self.nan_margin>1:
            raise ValueError("self.nan_margin must be within the range of 0 to 1")
        self.group_col=group_col
        self.target_col=target_col
        self.infer_size=infer_size
        self.save_oof_preds=save_oof_preds
        self.num_classes=num_classes
        self.device=device.lower()
        if (self.objective=='binary') and self.num_classes!=2:
            raise ValueError("num_classes must be 2")
        elif self.objective=='multi_class' and self.num_classes==None:
            raise ValueError("num_classes must be a number")
        self.one_hot_max=one_hot_max
        self.pretrained_models={}#用字典的方式保存已经训练好的模型
        
    def get_details(self,):
        #目前支持的评估指标有
        metrics=['rmse','mse','accuracy','auc']
        #目前支持的模型有
        models=['lgb','cat','xgb']
        #目前支持的交叉验证方法有
        kfolds=['KFold','GroupKFold','StratifiedKFold','StratifiedGroupKFold']
        #目前支持的任务
        objectives=['binary','multi_class','regression']
        print(f"Currently supported metrics:{metrics}")
        print(f"Currently supported models:{models}")
        print(f"Currently supported kfolds:{kfolds}")
        print(f"Currently supported objectives:{objectives}")
        
    #遍历表格df的所有列修改数据类型减少内存使用
    def reduce_mem_usage(self,df, float16_as32=True):
        #memory_usage()是df每列的内存使用量,sum是对它们求和, B->KB->MB
        start_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

        for col in df.columns:#遍历每列的列名
            col_type = df[col].dtype#列名的type
            if col_type != object and str(col_type)!='category':#不是object也就是说这里处理的是数值类型的变量
                c_min,c_max = df[col].min(),df[col].max() #求出这列的最大值和最小值
                if str(col_type)[:3] == 'int':#如果是int类型的变量,不管是int8,int16,int32还是int64
                    #如果这列的取值范围是在int8的取值范围内,那就对类型进行转换 (-128 到 127)
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    #如果这列的取值范围是在int16的取值范围内,那就对类型进行转换(-32,768 到 32,767)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    #如果这列的取值范围是在int32的取值范围内,那就对类型进行转换(-2,147,483,648到2,147,483,647)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    #如果这列的取值范围是在int64的取值范围内,那就对类型进行转换(-9,223,372,036,854,775,808到9,223,372,036,854,775,807)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:#如果是浮点数类型.
                    #如果数值在float16的取值范围内,如果觉得需要更高精度可以考虑float32
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        if float16_as32:#如果数据需要更高的精度可以选择float32
                            df[col] = df[col].astype(np.float32)
                        else:
                            df[col] = df[col].astype(np.float16)  
                    #如果数值在float32的取值范围内，对它进行类型转换
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    #如果数值在float64的取值范围内，对它进行类型转换
                    else:
                        df[col] = df[col].astype(np.float64)
        #计算一下结束后的内存
        end_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        #相比一开始的内存减少了百分之多少
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

        return df
        
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
            self.nunique_2_cols=[]
            for col in df.columns:
                if col!=self.target_col and col!=self.group_col:
                    if (df[col].nunique()<self.one_hot_max) and (df[col].nunique()>2):
                        self.one_hot_cols.append([col,list(df[col].unique())]) 
                    elif df[col].nunique()==2:
                        self.nunique_2_cols.append([col,list(df[col].unique())[0]])
                    
        for i in range(len(self.one_hot_cols)):
            col,nunique=self.one_hot_cols[i]
            for u in nunique:
                df[f"{col}_{u}"]=(df[col]==u).astype(np.int8)
        for i in range(len(self.nunique_2_cols)):
            c,u=self.nunique_2_cols[i]
            df[f"{c}_{u}"]=(df[c]==u).astype(np.int8)
        
        #去除无用的列
        df.drop(self.nan_cols+self.unique_cols+self.object_cols,axis=1,inplace=True,errors='ignore')
        df=self.reduce_mem_usage(df, float16_as32=True)
        return df
    
    def Metric(self,y_true,y_pred):#对于分类任务是标签和预测的每个类别的概率
        if self.metric=='rmse':
            return np.sqrt(np.mean((y_true-y_pred)**2))
        if self.metric=='mse':
            return np.mean((y_true-y_pred)**2)
        if self.metric=='accuracy':
            #概率转换成最大的类别
            y_pred=np.argmax(y_pred,axis=1)
            return np.mean(y_true==y_pred)
        if self.metric=='auc':
            return roc_auc_score(y_true,y_pred[:,1])
    def fit(self,train_path_or_file='train.csv'):
        try:#path试试
            self.train=pl.read_csv(train_path_or_file)
            self.train=self.train.to_pandas()
        except:#csv_file
            self.train=train_path_or_file
        #提供的训练数据不是df表格
        if not isinstance(self.train, pd.DataFrame):
            raise ValueError("train_path_or_file is not pd.DataFrame")
        self.train=self.Feature_Engineer(self.train,mode='train')
        
        #二分类,多分类,回归
        if self.objective not in ['binary','multi_class','regression']:
            raise ValueError("Wrong or currently unsupported objective")
        
        #选择哪种交叉验证方法
        if self.objective=='binary' or self.objective=='multi_class':
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
            
            metric=self.metric
            if self.objective=='multi_class':
                metric='multi_logloss'
            lgb_params={"boosting_type": "gbdt","metric": metric,
                        'random_state': self.seed,  "max_depth": 10,"learning_rate": 0.05,
                        "n_estimators": 10000,"colsample_bytree": 0.6,"colsample_bynode": 0.6,"verbose": -1,"reg_alpha": 0.2,
                        "reg_lambda": 5,"extra_trees":True,'num_leaves':64,"max_bin":255,
                        }
            
            #catboost的metric设置
            if self.metric=='multi_logloss':
                metric='Logloss'
            elif self.metric=='auc':
                metric='AUC'
            elif self.metric in ['rmse','mse']:
                metric='RMSE'
            elif self.metric=='accuracy':
                metric='Accuracy'
            cat_params={'eval_metric'         : metric,
                       'bagging_temperature' : 0.50,
                       'iterations'          : 10000,
                       'learning_rate'       : 0.08,
                       'max_depth'           : 12,
                       'l2_leaf_reg'         : 1.25,
                       'min_data_in_leaf'    : 24,
                       'random_strength'     : 0.25, 
                       'verbose'             : 0,
                      }
            
            xgb_params={'random_state': 2024, 'n_estimators': 10000, 
                        'learning_rate': 0.01, 'max_depth': 10,
                        'reg_alpha': 0.08, 'reg_lambda': 0.8, 
                        'subsample': 0.95, 'colsample_bytree': 0.6, 
                        'min_child_weight': 3,'early_stopping_rounds':100,
                       }
            if self.device in ['cuda','gpu']:#gpu常见的写法,目前只有lgb模型
                lgb_params['device']='gpu'
                lgb_params['gpu_use_dp']=True
                cat_params['task_type']="GPU"
                xgb_params['tree_method']='gpu_hist'
                
            if self.objective=='regression':
                self.models=[(LGBMRegressor(**lgb_params),'lgb'),
                             (CatBoostRegressor(**cat_params),'cat'),
                             (XGBRegressor(**xgb_params),'xgb')
                            ]
            else:
                self.models=[(LGBMClassifier(**lgb_params),'lgb'),
                             (CatBoostClassifier(**cat_params),'cat'),
                             (XGBClassifier(**xgb_params),'xgb'),
                            ]
        
        X=self.train.drop([self.group_col,self.target_col],axis=1,errors='ignore')
        y=self.train[self.target_col]
        
        self.col2name={}
        for i in range(len(list(X.columns))):
            self.col2name[list(X.columns)[i]]=f'col_{i}'
        X=X.rename(columns=self.col2name)
        
        for col in X.columns:
            if X[col].dtype==object:
                X[col]=X[col].astype(np.float32)
                
        #分类任务搞个target2idx,idx2target
        if self.objective!='regression':
            self.target2idx={}
            self.idx2target={}
            y_unique=sorted(list(y.unique()))
            for i in range(len(y_unique)):
                self.target2idx[y_unique[i]]=i
                self.idx2target[i]=y_unique[i]
            y=y.apply(lambda k:self.target2idx[k])
        
        if self.group_col!=None:
            group=self.train[self.group_col]
        else:
            group=None
        for (model,model_name) in self.models:
            if self.objective=='regression':
                oof_preds=np.zeros(len(y))
            else:
                oof_preds=np.zeros((len(y),self.num_classes))
            for fold, (train_index, valid_index) in (enumerate(kf.split(X,y,group))):
                print(f"name:{model_name},fold:{fold}")

                X_train, X_valid = X.iloc[train_index].reset_index(drop=True), X.iloc[valid_index].reset_index(drop=True)
                y_train, y_valid = y.iloc[train_index].reset_index(drop=True), y.iloc[valid_index].reset_index(drop=True)

                if 'lgb' in model_name:
                    model.fit(X_train,y_train,eval_set=[(X_valid, y_valid)],
                             callbacks=[log_evaluation(100),early_stopping(200)]
                        ) 
                elif 'cat' in model_name:
                    model.fit(X_train, y_train,
                          eval_set=(X_valid, y_valid),
                          early_stopping_rounds=100, verbose=200)
                elif 'xgb' in model_name:
                    model.fit(X_train,y_train,eval_set=[(X_valid, y_valid)],verbose=200)
                else:#假设你还有其他的模型
                    model.fit(X_train,y_train) 
                
                if self.objective=='regression':
                    oof_preds[valid_index]=model.predict(X_valid)
                else:
                    oof_preds[valid_index]=model.predict_proba(X_valid)
                self.pretrained_models[f'{model_name}_fold{fold}']=model
            print(f"{self.metric}:{self.Metric(y.values,oof_preds)}")
            if self.save_oof_preds:#如果需要保存oof_preds
                np.save(f"{model_name}_seed{self.seed}_fold{self.num_folds}.npy",oof_preds)
        
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
        for col in self.test.columns:
            if self.test[col].dtype==object:
                self.test[col]=self.test[col].astype(np.float32)
        if self.objective=='regression':
            test_preds=np.zeros((len(self.models)*self.num_folds,len(self.test)))
            fold=0
            for (model_name,model) in self.pretrained_models.items():
                test_pred=np.zeros(len(self.test))
                for i in range(0,len(self.test),self.infer_size):
                    test_pred[i:i+self.infer_size]=model.predict(self.test[i:i+self.infer_size])
                test_preds[fold]=test_pred
                fold+=1
            return test_preds.mean(axis=0)
        else:#分类任务到底要的是什么
            test_preds=np.zeros((len(self.models)*self.num_folds,len(self.test),self.num_classes))
            fold=0
            for (model_name,model) in self.pretrained_models.items():
                test_pred=np.zeros((len(self.test),self.num_classes))
                for i in range(0,len(self.test),self.infer_size):
                    test_pred[i:i+self.infer_size]=model.predict_proba(self.test[i:i+self.infer_size])
                test_preds[fold]=test_pred
                fold+=1
            if self.metric=='auc':
                return test_preds.mean(axis=0)[:,1]
            test_preds=np.argmax(test_preds.mean(axis=0),axis=1)
            return test_preds
    def submit(self,submission_path='submission.csv',test_preds=None):
        submission=pd.read_csv(submission_path)
        submission[self.target_col]=test_preds
        if self.objective!='regression':
            if self.metric!='auc':
                submission[self.target_col]=submission[self.target_col].apply(lambda x:idx2target[x])
        submission.to_csv("yunbase.csv",index=None)
        submission.head()

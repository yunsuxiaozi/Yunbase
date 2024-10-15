"""
@author:yunsuxiaozi
@start_time:2024/09/27
@update_time:2024/10/15
"""
import polars as pl#和pandas类似,但是处理大型数据集有更好的性能.
import pandas as pd#读取csv文件的库
import numpy as np#对矩阵进行科学计算的库
#这里使用groupkfold
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedGroupKFold,GroupKFold
from sklearn.preprocessing import LabelEncoder#类别标签转换成数值.


#二分类常用的评估指标
from sklearn.metrics import roc_auc_score,f1_score,matthews_corrcoef
#model lightgbm回归模型,日志评估
from  lightgbm import LGBMRegressor,LGBMClassifier,log_evaluation,early_stopping
from catboost import CatBoostRegressor,CatBoostClassifier
from xgboost import XGBRegressor,XGBClassifier
import dill#对对象进行序列化和反序列化(例如保存和加载树模型)
import optuna#自动超参数优化框架
import warnings#避免一些可以忽略的报错
warnings.filterwarnings('ignore')#filterwarnings()方法是用于设置警告过滤器的方法，它可以控制警告信息的输出方式和级别。
class Yunbase():
    def __init__(self,num_folds=5,
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
                      labelencoder_cols=[]
                ):
        """
        num_folds:是k折交叉验证的折数
        models:我这里已经设置了基本的模型作为baseline,你也可以使用你自己想使用的模型
        FE:除了我这里已经做了的基本的特征工程以外,你可以自定义你需要的特征工程的函数.
        drop_cols:所有的特征工程(你的自定义特征工程+内置的特征工程)完成之后想要删除的列.
        seed:随机种子
        objective:你想做的任务是什么?,regression,binary还是multi_class
        metric:你想使用的评估指标.
        nan_margin:一列缺失值大于多少选择不要
        group_col:groupkfold需要有一列作为group
        num_classes:如果是分类任务,需要指定类别数量
        target_col:需要预测的那一列
        infer_size:测试数据可能内存太大,一次推理会出错,所以有了分批次预测.
        save_oof_preds:是否保存交叉验证的结果用于后续的研究.
        save_test_preds:是否保存测试数据的预测结果
        device:将树模型放在GPU还是CPU上训练.
        one_hot_max:一列特征的nunique少于多少做onehot处理.
        custom_metric:自定义评估指标,对于分类任务,输入的是y_true和预测出每个类别的概率分布.
        use_optuna_find_params:使用optuna找参数的迭代次数,如果为0则说明不找,目前只支持lightgbm模型.
        optuna_direction:'minimize'或者'maximize',评估指标是最大还是最小
        early_stop:早停的次数,如果模型迭代多少次没有改善就会停下来.
        use_pseudo_label:是否用伪标签,就是在得到测试数据的预测结果后将测试数据加入训练数据再训练一次.
        use_high_correlation_feature:bool类型的变量,你是否要保留训练数据中高相关性的特征,如果要保留,为True,反之为False.
        labelencoder_cols:一般是字符串类型的类别型变量,转换成[1,2,3]
        """
        
        #目前支持的评估指标有
        self.supported_metrics=['custom_metric',
                                'mae','rmse','mse','medae','rmsle',#回归任务
                                'auc','f1_score','mcc',#二分类任务
                                'accuracy','logloss',#多分类任务(分类任务)
                               ]
        #目前支持的模型有
        self.supported_models=['lgb','cat','xgb']
        #目前支持的交叉验证方法有
        self.supported_kfolds=['KFold','GroupKFold','StratifiedKFold','StratifiedGroupKFold']
        #目前支持的任务
        self.supported_objectives=['binary','multi_class','regression']
        
        print(f"Currently supported metrics:{self.supported_metrics}")
        print(f"Currently supported models:{self.supported_models}")
        print(f"Currently supported kfolds:{self.supported_kfolds}")
        print(f"Currently supported objectives:{self.supported_objectives}")
        
        self.num_folds=num_folds
        if self.num_folds<2:
            raise ValueError("num_folds must be greater than 2")
        self.seed=seed
        self.models=models
        self.FE=FE
        self.drop_cols=drop_cols
        
        self.objective=objective.lower()
        #二分类,多分类,回归
        if self.objective not in self.supported_objectives:
            raise ValueError("Wrong or currently unsupported objective")
        
        self.custom_metric=custom_metric#function
        if self.custom_metric!=None:
            self.metric='custom_metric'
        else:
            self.metric=metric.lower()
        if self.metric not in self.supported_metrics and self.custom_metric==None:
            raise ValueError("Wrong or currently unsupported metric,You can customize the evaluation metrics using 'custom_metric'")
        
        self.nan_margin=nan_margin
        if self.nan_margin<0 or self.nan_margin>1:
            raise ValueError("nan_margin must be within the range of 0 to 1")
        self.group_col=group_col
        self.target_col=target_col
        self.infer_size=infer_size
        self.save_oof_preds=save_oof_preds
        if self.save_oof_preds not in [True,False]:
            raise ValueError("save_oof_preds must be True or False")  
        self.save_test_preds=save_test_preds
        if self.save_test_preds not in [True,False]:
            raise ValueError("save_test_preds must be True or False")
        self.num_classes=num_classes
        self.device=device.lower()
        if (self.objective=='binary') and self.num_classes!=2:
            raise ValueError("num_classes must be 2")
        elif self.objective=='multi_class' and self.num_classes==None:
            raise ValueError("num_classes must be a number")
        self.one_hot_max=one_hot_max
        self.use_optuna_find_params=use_optuna_find_params
        self.optuna_direction=optuna_direction
        #如果你要用optuna找参数并且自定义评估指标,却不说评估指标要最大还是最小,就要报错.
        if (self.use_optuna_find_params) and (self.custom_metric!=None) and self.optuna_direction not in ['minimize','maximize']:
            raise ValueError("optuna_direction must be 'minimize' or 'maximize'")
        self.model_paths=[]#保存模型的路径
        self.early_stop=early_stop
        self.test=None#初始化时没有测试数据.
        self.use_pseudo_label=use_pseudo_label
        self.use_high_correlation_feature=use_high_correlation_feature
        self.labelencoder_cols=labelencoder_cols

    #保存训练好的树模型,obj是保存的模型,path是需要保存的路径
    def pickle_dump(self,obj, path):
        #打开指定的路径path,binary write(二进制写入)
        with open(path, mode="wb") as f:
            #将obj对象保存到f,使用协议版本4进行序列化
            dill.dump(obj, f, protocol=4)
    def pickle_load(self,path):
        #打开指定的路径path,binary read(二进制读取)
        with open(path, mode="rb") as f:
            #按照制定路径去加载模型
            data = dill.load(f)
            return data
        
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
        
    #对训练数据或者测试数据做特征工程,mode='train'或者'test' ,drop_cols是其他想删除的列名
    def base_FE(self,df,mode='train',drop_cols=[]):
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
        print("one hot encoder")          
        for i in range(len(self.one_hot_cols)):
            col,nunique=self.one_hot_cols[i]
            for u in nunique:
                df[f"{col}_{u}"]=(df[col]==u).astype(np.int8)
        for i in range(len(self.nunique_2_cols)):
            c,u=self.nunique_2_cols[i]
            df[f"{c}_{u}"]=(df[c]==u).astype(np.int8)
            
        #labelencoder
        print("label encoder")
        for col in self.labelencoder_cols:
            if mode=='train':#训练的时候
                le=LabelEncoder()
                le.fit(df[col])#对id这列进行拟合
                self.pickle_dump(le,f'le_{col}.model')
            else:
                le=self.pickle_load(f'le_{col}.model')   
            df[col+"_le"] = le.transform(df[col])
            
        if (mode=='train') and (self.use_high_correlation_feature==False):#如果需要删除高相关性的特征
            self.drop_high_correlation_feats(df)
        
        #去除无用的列
        print("drop useless cols")
        df.drop(self.nan_cols+self.unique_cols+self.object_cols+drop_cols,axis=1,inplace=True,errors='ignore')
        df=self.reduce_mem_usage(df, float16_as32=True)
        print("-"*30)
        return df
    
    def Metric(self,y_true,y_pred):#对于分类任务是标签和预测的每个类别的概率
        #如果你有自定义的评估指标,那就用你的评估指标
        if self.custom_metric!=None:
            return self.custom_metric(y_true,y_pred)
        if self.objective=='regression':
            if self.metric=='medae':
                return np.median(np.abs(y_true-y_pred))
            elif self.metric=='mae':
                return np.mean(np.abs(y_true-y_pred))
            elif self.metric=='rmse':
                return np.sqrt(np.mean((y_true-y_pred)**2))
            elif self.metric=='mse':
                return np.mean((y_true-y_pred)**2)
            elif self.metric=='rmsle':
                   return np.sqrt(np.mean((np.log1p(y_pred)-np.log1p(y_true))**2))
        else:
            if self.metric=='accuracy':
                #转换成概率最大的类别
                y_pred=np.argmax(y_pred,axis=1)
                return np.mean(y_true==y_pred)
            elif self.metric=='auc':
                return roc_auc_score(y_true,y_pred[:,1])
            elif self.metric=='f1_score':
                #转换成概率最大的类别
                y_pred=np.argmax(y_pred,axis=1)
                return f1_score(y_true, y_pred)
            elif self.metric=='mcc':
                #转换成概率最大的类别
                y_pred=np.argmax(y_pred,axis=1)
                return matthews_corrcoef(y_true, y_pred)
            elif self.metric=='logloss':
                eps=1e-15
                label=np.zeros_like(y_pred)
                for i in range(len(label)):
                    label[i][y_true[i]-1]=1
                y_true=label
                y_pred=np.clip(y_pred,eps,1-eps)
                return -np.mean(np.sum(y_true*np.log(y_pred),axis=-1))
        
    #用optuna找lgb模型的参数,暂时不支持custom_metric
    def optuna_lgb(self,X,y,group,kf,metric):
        def objective(trial):
            params = {
                "boosting_type": "gbdt","metric": metric,
                'random_state': self.seed,
                'n_estimators': trial.suggest_int('n_estimators', 500,1500),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),#对数分布的建议值
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),#浮点数
                'subsample': trial.suggest_float('subsample', 0.5, 1),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.5, log=True),
                'num_leaves' : trial.suggest_int('num_leaves', 8, 64),#整数
                'min_child_samples': trial.suggest_int('min_child_samples', 2, 100),
                "extra_trees":True,
                "verbose": -1
            }
            if self.device in ['cuda','gpu']:#gpu常见的写法
                params['device']='gpu'
                params['gpu_use_dp']=True
            model_name='lgb'
            if self.objective=='regression':
                model=LGBMRegressor(**params)
            else:
                model=LGBMClassifier(**params)
            oof_preds,metric_score=self.cross_validation(X,y,group,kf,model,model_name,use_optuna=True)
            return metric_score
        #优化最大值还是最小值
        if self.metric in ['accuracy','auc','f1_score','mcc']:
            direction='maximize'
        elif self.metric in ['medae','mae','rmse','mse','logloss','rmsle']:
            direction='minimize'
        else:
            direction=self.optuna_direction
            
        #创建的研究命名,找最大值.
        study = optuna.create_study(direction=direction, study_name='find best lgb_params')
        #目标函数,尝试的次数  
        study.optimize(objective, n_trials=self.use_optuna_find_params)
        best_params=study.best_trial.params
        best_params["boosting_type"]="gbdt"
        best_params["extra_trees"]=True
        best_params["metric"]=metric
        best_params['random_state']=self.seed
        best_params['verbose']=-1
        print(f"best_params={best_params}")
        return best_params
    
    #这个function会返回 oof_preds和metric_score,用于optuna找参数的.如果在使用optuna找参数,就不用保存预训练模型.
    def cross_validation(self,X,y,group,kf,model,model_name,use_optuna=False):
        log=100
        if use_optuna:
            log=10000
        if self.objective=='regression':
            oof_preds=np.zeros(len(y))
        else:
            oof_preds=np.zeros((len(y),self.num_classes))
        for fold, (train_index, valid_index) in (enumerate(kf.split(X,y,group))):
            print(f"name:{model_name},fold:{fold}")

            X_train, X_valid = X.iloc[train_index].reset_index(drop=True), X.iloc[valid_index].reset_index(drop=True)
            y_train, y_valid = y.iloc[train_index].reset_index(drop=True), y.iloc[valid_index].reset_index(drop=True)

            #如果决定使用伪标签,并且已经得到测试数据的预测结果
            #初始化的self.test=None,只有predict函数里self.test才会有数据,这时已经fit过了
            if (self.use_pseudo_label) and (type(self.test)==pd.DataFrame):
                test_X=self.test.drop([self.group_col,self.target_col],axis=1,errors='ignore')
                test_y=self.test[self.target_col]
                X_train=pd.concat((X_train,test_X),axis=0)
                y_train=pd.concat((y_train,test_y),axis=0)
            
            if 'lgb' in model_name:
                model.fit(X_train,y_train,eval_set=[(X_valid, y_valid)],
                         callbacks=[log_evaluation(log),early_stopping(self.early_stop)]
                    ) 
            elif 'cat' in model_name:
                model.fit(X_train, y_train,
                      eval_set=(X_valid, y_valid),
                      early_stopping_rounds=self.early_stop, verbose=log)
            elif 'xgb' in model_name:
                model.fit(X_train,y_train,eval_set=[(X_valid, y_valid)],verbose=log)
            else:#假设你还有其他的模型
                model.fit(X_train,y_train) 

            if self.objective=='regression':
                oof_preds[valid_index]=model.predict(X_valid)
            else:
                oof_preds[valid_index]=model.predict_proba(X_valid)
            if not use_optuna:#如果没有在找参数
                self.pickle_dump(model,f'{model_name}_fold{fold}.model')
                self.model_paths.append((model_name,fold))
        metric_score=self.Metric(y.values,oof_preds)
        return oof_preds,metric_score
    
    def drop_high_correlation_feats(self,df):
        #target_col和group_col都是模型训练要用的,不能删,object特征计算不了相关性
        #这里相关性定死0.99,毕竟低于这个值的特征还是有信息的,可以用PCA之类的降维方法.
        #如果你需要删除其他高相关性的特征,可以自行添加进初始化参数drop_cols中.
        numerical_cols=[col for col in df.columns if (col not in [self.target_col,self.group_col]) and df[col].dtype!=object]
        corr_matrix=df[numerical_cols].corr().values
        drop_cols=[]
        for i in range(len(corr_matrix)):
            for j in range(i+1,len(corr_matrix)):
                if abs(corr_matrix[i][j])>=0.99:
                    drop_cols.append(numerical_cols[j])
        #加入drop_cols中,后续特征工程结束一起drop
        print(f"drop_cols={drop_cols}")
        self.drop_cols+=drop_cols
    
    def fit(self,train_path_or_file='train.csv'):
        self.train_path_or_file=train_path_or_file
        try:#path试试
            self.train=pl.read_csv(train_path_or_file)
            self.train=self.train.to_pandas()
        except:#csv_file 或者parquet
            try:
                self.train=pl.read_parquet(train_path_or_file)
                self.train=self.train.to_pandas()
            except:
                self.train=train_path_or_file
        #如果是polars文件,转成pandas
        if isinstance(self.train, pl.DataFrame):
            self.train=self.train.to_pandas()
        #提供的训练数据不是df表格
        if not isinstance(self.train, pd.DataFrame):
            raise ValueError("train_path_or_file is not pd.DataFrame")
        print(f"len(train):{len(self.train)}")
        self.train=self.base_FE(self.train,mode='train',drop_cols=self.drop_cols)
        
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
        
        X=self.train.drop([self.group_col,self.target_col],axis=1,errors='ignore')
        y=self.train[self.target_col]
        
        self.col2name={}
        for i in range(len(list(X.columns))):
            self.col2name[list(X.columns)[i]]=f'col_{i}'
        X=X.rename(columns=self.col2name)
        print(f"feature_count:{len(list(X.columns))}")
        
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
        #存储训练集真实的标签,因为有target2idx的操作,用于最后算final_score    
        self.target=y.values
        
        #模型的训练,如果你自己准备了模型,那就用你的模型,否则就用我的模型
        if len(self.models)==0:
            
            metric=self.metric
            if self.objective=='multi_class':
                metric='multi_logloss'
            #lightgbm不支持f1_score,但是最后调用Metric的时候会计算f1_score
            if metric in ['f1_score','mcc','logloss']:
                metric='auc'
            elif metric=='medae':
                metric='mae'
            elif metric=='rmsle':
                metric='mse'
            if self.custom_metric!=None:#用custom_metric
                if self.objective=='regression':
                    metric='rmse'
                elif self.objective=='binary':
                    metric='auc'
                elif self.objective=='multi_class':
                    metric='multi_logloss'
            lgb_params={"boosting_type": "gbdt","metric": metric,
                        'random_state': self.seed,  "max_depth": 10,"learning_rate": 0.05,
                        "n_estimators": 10000,"colsample_bytree": 0.6,"colsample_bynode": 0.6,"verbose": -1,"reg_alpha": 0.2,
                        "reg_lambda": 5,"extra_trees":True,'num_leaves':64,"max_bin":255,
                        }
            #找到新的参数
            if self.use_optuna_find_params:#如果要用optuna找lgb模型的参数
                lgb_params=self.optuna_lgb(X,y,group,kf,metric)
             
            #catboost的metric设置
            # Valid options are: 'Logloss', 'CrossEntropy', 'CtrFactor', 'Focal', 'RMSE', 'LogCosh', 
            # 'Lq', 'MAE', 'Quantile', 'MultiQuantile', 'Expectile', 'LogLinQuantile', 'MAPE', 
            # 'Poisson', 'MSLE', 'MedianAbsoluteError', 'SMAPE', 'Huber', 'Tweedie', 'Cox', 
            # 'RMSEWithUncertainty', 'MultiClass', 'MultiClassOneVsAll', 'PairLogit', 'PairLogitPairwise',
            # 'YetiRank', 'YetiRankPairwise', 'QueryRMSE', 'GroupQuantile', 'QuerySoftMax', 
            # 'QueryCrossEntropy', 'StochasticFilter', 'LambdaMart', 'StochasticRank', 
            # 'PythonUserDefinedPerObject', 'PythonUserDefinedMultiTarget', 'UserPerObjMetric',
            # 'UserQuerywiseMetric', 'R2', 'NumErrors', 'FairLoss', 'AUC', 'Accuracy', 'BalancedAccuracy',
            # 'BalancedErrorRate', 'BrierScore', 'Precision', 'Recall', 'F1', 'TotalF1', 'F', 'MCC', 
            # 'ZeroOneLoss', 'HammingLoss', 'HingeLoss', 'Kappa', 'WKappa', 'LogLikelihoodOfPrediction',
            # 'NormalizedGini', 'PRAUC', 'PairAccuracy', 'AverageGain', 'QueryAverage', 'QueryAUC',
            # 'PFound', 'PrecisionAt', 'RecallAt', 'MAP', 'NDCG', 'DCG', 'FilteredDCG', 'MRR', 'ERR', 
            # 'SurvivalAft', 'MultiRMSE', 'MultiRMSEWithMissingValues', 'MultiLogloss', 'MultiCrossEntropy',
            # 'Combination'. 
            if self.metric=='multi_logloss':
                metric='Accuracy'
            elif self.metric=='logloss':
                metric='Logloss'
            elif self.metric in ['mse','rmsle']:
                metric='RMSE'
            elif self.metric=='accuracy':
                metric='Accuracy'
            elif self.metric=='f1_score':
                metric='F1'
            elif self.metric in ['auc','rmse','mcc','mae']:#catboost里是大写的评估指标
                metric=metric.upper()
            elif self.metric=='medae':
                metric='MAE'
            if self.custom_metric!=None:#用custom_metric
                if self.objective=='regression':
                    metric='RMSE'
                elif self.objective=='binary':
                    metric='Logloss'
                else:
                    metric='Accuracy'
                    
            cat_params={
                       'random_state':self.seed,
                       'eval_metric'         : metric,
                       'bagging_temperature' : 0.50,
                       'iterations'          : 10000,
                       'learning_rate'       : 0.08,
                       'max_depth'           : 12,
                       'l2_leaf_reg'         : 1.25,
                       'min_data_in_leaf'    : 24,
                       'random_strength'     : 0.25, 
                       'verbose'             : 0,
                      }
            
            xgb_params={'random_state': self.seed, 'n_estimators': 10000, 
                        'learning_rate': 0.01, 'max_depth': 10,
                        'reg_alpha': 0.08, 'reg_lambda': 0.8, 
                        'subsample': 0.95, 'colsample_bytree': 0.6, 
                        'min_child_weight': 3,'early_stopping_rounds':self.early_stop,
                       }

            if self.device in ['cuda','gpu']:#gpu常见的写法
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
                
            print(f"lgb_params:{lgb_params}")
            print(f"xgb_params:{xgb_params}")
            print(f"cat_params:{cat_params}")
            
        for (model,model_name) in self.models:
            oof_preds,metric_score=self.cross_validation(X,y,group,kf,model,model_name,use_optuna=False)
            print(f"{self.metric}:{metric_score}")
            if self.save_oof_preds:#如果需要保存oof_preds
                np.save(f"{model_name}_seed{self.seed}_fold{self.num_folds}.npy",oof_preds)
        
    def predict(self,test_path_or_file='test.csv',weights=None):
        #weights:[1]*len(self.models),几个模型的交叉验证就几个权重,下面会扩展到num_folds倍,后续也会对权重进行归一化处理.
        n=len(self.models)
        #如果你不设置权重,就按照普通的求平均来操作
        if weights==None:
            weights=np.ones(n)

        if len(weights)!=n:
            raise ValueError(f"length of weights must be len(models)")
        weights=np.array([w for w in weights for f in range(self.num_folds)],dtype=np.float32)
        #归一化
        weights=weights*(self.num_folds*n)/np.sum(weights)

        #计算oof分数             
        oof_preds=np.zeros_like(np.load(f"{self.models[0//self.num_folds][1]}_seed{self.seed}_fold{self.num_folds}.npy"))
        for i in range(0,len(weights),self.num_folds):
            oof_pred=np.load(f"{self.models[i//self.num_folds][1]}_seed{self.seed}_fold{self.num_folds}.npy")
            oof_preds+=weights[i]*oof_pred
        oof_preds=oof_preds/n
        print(f"final_{self.metric}:{self.Metric(self.target,oof_preds)}")
        
        try:#path试试
            self.test=pl.read_csv(test_path_or_file)
            self.test=self.test.to_pandas()
        except:#
            try:#path试试
                self.test=pl.read_parquet(test_path_or_file)
                self.test=self.test.to_pandas()
            except:#
                self.test=test_path_or_file
        #如果是polars文件,转成pandas
        if isinstance(self.test, pl.DataFrame):
            self.test=self.test.to_pandas()
        #提供的训练数据不是df表格
        if not isinstance(self.test, pd.DataFrame):
            raise ValueError("test_path_or_file is not pd.DataFrame")
        print(f"len(test):{len(self.test)}")
        self.test=self.base_FE(self.test,mode='test',drop_cols=self.drop_cols)
        self.test=self.test.drop([self.group_col,self.target_col],axis=1,errors='ignore')
        self.test=self.test.rename(columns=self.col2name)
        for col in self.test.columns:
            if self.test[col].dtype==object:
                self.test[col]=self.test[col].astype(np.float32)
        if self.objective=='regression':
            test_preds=np.zeros((len(self.models)*self.num_folds,len(self.test)))
            cnt=0
            for (model_name,fold) in self.model_paths:
                model=self.pickle_load(f'{model_name}_fold{fold}.model')
                test_pred=np.zeros(len(self.test))
                for i in range(0,len(self.test),self.infer_size):
                    test_pred[i:i+self.infer_size]=model.predict(self.test[i:i+self.infer_size])
                test_preds[cnt]=test_pred
                cnt+=1
            test_preds=np.mean([test_preds[i]*weights[i] for i in range(len(test_preds))],axis=0)
            
            #伪标签代码
            if self.use_pseudo_label:
                self.test[self.target_col]=test_preds
                self.model_paths=[]
                self.fit(self.train_path_or_file)
                
                test_preds=np.zeros((len(self.models)*self.num_folds,len(self.test)))
                cnt=0
                for (model_name,fold) in self.model_paths:
                    model=self.pickle_load(f'{model_name}_fold{fold}.model')
                    test_pred=np.zeros(len(self.test))
                    for i in range(0,len(self.test),self.infer_size):
                        test_pred[i:i+self.infer_size]=model.predict(self.test.drop([self.target_col],axis=1)[i:i+self.infer_size])
                    test_preds[cnt]=test_pred
                    cnt+=1
                test_preds=np.mean([test_preds[i]*weights[i] for i in range(len(test_preds))],axis=0)
            
            if self.save_test_preds:
                np.save('test_preds.npy',test_preds)
            return test_preds
        else:#分类任务到底要的是什么
            test_preds=np.zeros((len(self.models)*self.num_folds,len(self.test),self.num_classes))
            cnt=0
            for (model_name,fold) in self.model_paths:
                model=self.pickle_load(f'{model_name}_fold{fold}.model')
                test_pred=np.zeros((len(self.test),self.num_classes))
                for i in range(0,len(self.test),self.infer_size):
                    test_pred[i:i+self.infer_size]=model.predict_proba(self.test[i:i+self.infer_size])
                test_preds[cnt]=test_pred
                cnt+=1   
            test_preds=np.mean([test_preds[i]*weights[i] for i in range(len(test_preds))],axis=0)#(len(test),self.num_classes)
            
            #伪标签代码
            if self.use_pseudo_label:
                self.test[self.target_col]=np.argmax(test_preds,axis=1)
                self.model_paths=[]
                self.fit(self.train_path_or_file)

                test_preds=np.zeros((len(self.models)*self.num_folds,len(self.test),self.num_classes))
                fold=0
                for (model_name,fold) in self.model_paths:
                    model=self.pickle_load(f'{model_name}_fold{fold}.model')
                    test_pred=np.zeros((len(self.test),self.num_classes))
                    for i in range(0,len(self.test),self.infer_size):
                        test_pred[i:i+self.infer_size]=model.predict_proba(self.test.drop([self.target_col],axis=1)[i:i+self.infer_size])
                    test_preds[fold]=test_pred
                    fold+=1
                test_preds=np.mean([test_preds[i]*weights[i] for i in range(len(test_preds))],axis=0)
            if self.metric=='auc':
                return test_preds[:,1]
            if self.save_test_preds:
                np.save('test_preds.npy',test_preds)
            test_preds=np.argmax(test_preds,axis=1)
            return test_preds

    #集成很多solution
    def ensemble(self,solution_paths_or_files,weights=None):
        #如果你不设置权重,就按照普通的求平均来操作
        n=len(solution_paths_or_files)
        if weights==None:
            weights=np.ones(n)
        if len(weights)!=n:
            raise ValueError(f"length of weights must be len(solution_paths_or_files)")
        #归一化
        weights=weights/np.sum(weights)

        #连续值加权求和
        if (self.objective=='regression') or(self.metric=='auc'):
            final_solutions=[]
            for i in range(n):
                try:#path试试
                    solution=pl.read_csv(solution_paths_or_files[i])
                    solution=solution.to_pandas()
                except:#csv_file
                    solution=solution_paths_or_files[i]
                #提供的训练数据不是df表格
                if not isinstance(solution, pd.DataFrame):
                    raise ValueError("solution_paths_or_files is not pd.DataFrame")
                final_solutions.append(weights[i]*solution[self.target_col].values)
            final_solutions=np.sum(final_solutions,axis=0)
            return final_solutions
        else:#离散值(分类任务)求众数
            #n个solution,m个数据
            solutions=[]
            for i in range(n):
                try:#path试试
                    solution=pl.read_csv(solution_paths_or_files[i])
                    solution=solution.to_pandas()
                except:#csv_file
                    solution=solution_paths_or_files[i]
                #提供的训练数据不是df表格
                if not isinstance(solution, pd.DataFrame):
                    raise ValueError("solution_paths_or_files is not pd.DataFrame")
                solutions.append(solution[self.target_col].values)
            final_solutions=[]
            for i in range(len(solutions[0])):
                solution2count={}
                #第i个数据第j个solution
                for j in range(n):
                    if solutions[j][i] in solution2count.keys():
                        solution2count[ solutions[j][i] ]+=weights[j]
                    else:
                        solution2count[ solutions[j][i] ]=weights[j]
                solution2count=dict(sorted(solution2count.items(),key=lambda x:-x[1]))
                final_solutions.append(list(solution2count.keys())[0])
            final_solutions=np.array(final_solutions)
            return final_solutions
                
    def submit(self,submission_path='submission.csv',test_preds=None,save_name='yunbase'):
        submission=pd.read_csv(submission_path)
        submission[self.target_col]=test_preds
        if self.objective!='regression':
            if self.metric!='auc':
                submission[self.target_col]=submission[self.target_col].apply(lambda x:self.idx2target[x])
        submission.to_csv(f"{save_name}.csv",index=None)
        submission.head()

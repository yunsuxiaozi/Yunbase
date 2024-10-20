"""
@author:yunsuxiaozi
@start_time:2024/09/27
@update_time:2024/10/20
"""
import polars as pl#similar to pandas, but with better performance when dealing with large datasets.
import pandas as pd#read csv,parquet
import numpy as np#for scientific computation of matrices
#current supported kfold
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedGroupKFold,GroupKFold
import ast#parse Python list strings  transform '[a,b,c]' to [a,b,c]
#metrics
from sklearn.metrics import roc_auc_score,f1_score,matthews_corrcoef
#models(lgb,xgb,cat)
from  lightgbm import LGBMRegressor,LGBMClassifier,log_evaluation,early_stopping
from catboost import CatBoostRegressor,CatBoostClassifier
from xgboost import XGBRegressor,XGBClassifier
import dill#serialize and deserialize objects (such as saving and loading tree models)
import optuna#automatic hyperparameter optimization framework
import warnings#avoid some negligible errors
#The filterwarnings () method is used to set warning filters, which can control the output method and level of warning information.
warnings.filterwarnings('ignore')
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
                      use_high_corr_feat=True,
                      labelencoder_cols=[],
                      list_cols=[],
                      list_gaps=[1],
                      word2vec_models=[],
                ):
        """
        num_folds             :the number of folds for k-fold cross validation.
        models                :Built in 3 GBDTs as baseline, you can also use custom models,
                               such as models=[(LGBMRegressor(**lgb_params),'lgb')]
        FE                    :In addition to the built-in feature engineer, you can also customize feature engineer.
        drop_cols             :The column to be deleted after all feature engineering is completed.
        seed                  :random seed.
        objective             :what task do you want to do?regression,binary or multi_class?
        metric                :metric to evaluate your model.
        nan_margin            :when the proportion of missing values in a column is greater than, we delete this column.
        group_col             :if you want to use groupkfold,then define this group_col.
        num_classes           :if objectibe is multi_class,you should define this class.
        target_col            :the column that you want to predict.s
        infer_size            :the test data might be large,we can predict in batches.
        save_oof_preds        :you can save OOF for offline study.
        save_test_preds       :you can save test_preds.For multi classification tasks, 
                               the predicted result is the category.If you need to save the probability of the test_data,
                               you can save test_preds.
        device                :GBDT can training on GPU,you can set this parameter like NN.
        one_hot_max           :If the nunique of a column is less than a certain value, perform one hot encoder.
        custom_metric         :your custom_metric,when objective is multi_class,y_pred in custom(y_true,y_pred) is probability.            
        use_optuna_find_params:count of use optuna find best params,0 is not use optuna to find params.
                               Currently only LGBM is supported.
        optuna_direction      :'minimize' or 'maximize',when you use custom metric,you need to define 
                               the direction of optimization.
        early_stop            :Common parameters of GBDT.
        use_pseudo_label      :Whether to use pseudo labels.When it is true,adding the test data 
                               to the training data and training again after obtaining the predicted 
                               results of the test data.
        use_high_corr_feat    :whether to use high correlation features or not. 
        labelencoder_cols     :Convert categorical string variables into [1,2,……,n].
        list_cols             :If the data in a column is a list or str(list), this can be used to extract features.
        list_gaps             :extract features for list_cols.example=[1,2,4]
        word2vec_models       :Use models such as tfidf to extract features of string columns 
                               example:word2vec_models=[(TfidfVectorizer(),col,model_name)]
        """
        
        #currented supported metric
        self.supported_metrics=['custom_metric',#your custom_metric
                                'mae','rmse','mse','medae','rmsle',#regression
                                'auc','f1_score','mcc',#binary metric
                                'accuracy','logloss',#multi_class or classification
                               ]
        #current supported metric
        self.supported_models=['lgb','cat','xgb']
        #current supported kfold.
        self.supported_kfolds=['KFold','GroupKFold','StratifiedKFold','StratifiedGroupKFold']
        #current supported objective.
        self.supported_objectives=['binary','multi_class','regression']
        
        print(f"Currently supported metrics:{self.supported_metrics}")
        print(f"Currently supported models:{self.supported_models}")
        print(f"Currently supported kfolds:{self.supported_kfolds}")
        print(f"Currently supported objectives:{self.supported_objectives}")
        
        self.num_folds=num_folds
        if self.num_folds<2:#kfold must greater than 1
            raise ValueError("num_folds must be greater than 1")
        self.seed=seed
        self.models=models
        self.FE=FE
        self.drop_cols=drop_cols
        
        self.objective=objective.lower()
        #binary multi_class,regression
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
        if self.infer_size<=0 or type(self.infer_size) is not int:
            raise ValueError("infer size must be greater than 0 and must be int")  
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
        elif (self.objective=='multi_class') and (self.num_classes==None):
            raise ValueError("num_classes must be a number(int)")
        self.one_hot_max=one_hot_max
        self.use_optuna_find_params=use_optuna_find_params
        self.optuna_direction=optuna_direction
        if (self.use_optuna_find_params) and (self.custom_metric!=None) and self.optuna_direction not in ['minimize','maximize']:
            raise ValueError("optuna_direction must be 'minimize' or 'maximize'")
        self.model_paths=[]#training model path
        self.early_stop=early_stop
        self.test=None#test data will be replaced when call predict function.
        self.use_pseudo_label=use_pseudo_label
        self.use_high_corr_feat=use_high_corr_feat
        self.labelencoder_cols=labelencoder_cols
        self.load_path=""#If train and inference in different notebook,we should change the path model load.
        self.list_cols=list(set(list_cols))
        self.list_gaps=sorted(list_gaps)
        self.word2vec_models=word2vec_models
        self.word2vec_cols=[]#origin cols that need to use in tfidf model.
        self.col2name=None#Due to the presence of special characters in some column names, 
        #they cannot be directly passed into the LGB model training, so conversion is required

    #save models after training
    def pickle_dump(self,obj, path):
        #open path,binary write
        with open(path, mode="wb") as f:
            dill.dump(obj, f, protocol=4)
    #load models when inference
    def pickle_load(self,path):
        #open path,binary read
        with open(path, mode="rb") as f:
            data = dill.load(f)
            return data
        
    #Traverse all columns of df, modify data types to reduce memory usage
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
        #calculate memory after optimization
        end_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

        return df
        
    #basic Feature Engineer,mode='train' or 'test' ,drop_cols is other cols you want to delete.
    def base_FE(self,df,mode='train',drop_cols=[]):
        if self.FE!=None:
            #use your custom metric first
            df=self.FE(df)
        if mode=='train':
            #missing value 
            self.nan_cols=[col for col in df.columns if df[col].isna().mean()>self.nan_margin]
            #nunique=1
            self.unique_cols=[col for col in df.drop(self.list_cols,axis=1,errors='ignore').columns if(df[col].nunique()==1)]
            #object dtype
            self.object_cols=[col for col in df.columns if (df[col].dtype==object) and (col!=self.group_col)]
            #one_hot_cols
            self.one_hot_cols=[]
            self.nunique_2_cols=[]
            for col in df.columns:
                if col not in [self.target_col,self.group_col]+self.list_cols:
                    if (df[col].nunique()<self.one_hot_max) and (df[col].nunique()>2):
                        self.one_hot_cols.append([col,list(df[col].unique())]) 
                    elif df[col].nunique()==2:
                        self.nunique_2_cols.append([col,list(df[col].unique())[0]])
        print("< one hot encoder >")          
        for i in range(len(self.one_hot_cols)):
            col,nunique=self.one_hot_cols[i]
            for u in nunique:
                df[f"{col}_{u}"]=(df[col]==u).astype(np.int8)
        for i in range(len(self.nunique_2_cols)):
            c,u=self.nunique_2_cols[i]
            df[f"{c}_{u}"]=(df[c]==u).astype(np.int8)
        
        if len(self.list_cols):
            print("< list column's feature >")
            for col in self.list_cols:
                try:#if str(list)transform '[a,b]' to [a,b]
                    df[col]=df[col].apply(lambda x:ast.literal_eval(x))
                except:#origin data is list or data can't be parsed.
                    #find first data is not nan,if data.dtype!=list, then error
                    for i in range(len(df)):
                        v=df[col].values[i]
                        if v==v:#find first data isn't nan
                            if not isinstance(v, list):
                                raise ValueError(f"col '{col}' not a list")
                #origin feats 
                df[f'{col}_len']=df[col].apply(len)
                df[f'first_{col}']=df[col].apply(lambda x:x[0])
                df[f'last_{col}']=df[col].apply(lambda x:x[-1])
                df[f'mean_{col}']=df[col].apply(lambda x:np.nanmean(x))
                df[f'median_{col}']=df[col].apply(lambda x:np.nanmedian(x))
                df[f'max_{col}']=df[col].apply(lambda x:np.nanmax(x))
                df[f'min_{col}']=df[col].apply(lambda x:np.nanmin(x))
                df[f'std_{col}']=df[col].apply(lambda x:np.nanstd(x))
                df[f'sum_{col}']=df[col].apply(lambda x:np.nansum(x))
                df[f'ptp_{col}']=df[f'max_{col}']-df[f'min_{col}']
                df[f'mean_{col}/std_{col}']=df[f'mean_{col}']/df[f'std_{col}']
                #list gap_feats
                def get_list(l):
                    if len(self.list_gaps)==0:
                        return l
                    elif len(l)<self.list_gaps[-1]:
                        return l+[np.nan]*(self.list_gaps[-1]-len(l))
                    else:
                        return l
                df[col]=df[col].apply(lambda l:get_list(l))
                for gap in self.list_gaps:
                    v=df[col].values
                    for i in range(gap):
                        for j in range(len(v)):
                            v[j]=np.diff(v[j])
                    df[f'first_{col}_gap{gap}']=[vi[0] if len(vi) else np.nan for vi in v]
                    df[f'last_{col}_gap{gap}']=[vi[-1] if len(vi) else np.nan for vi in v]
                    df[f'mean_{col}_gap{gap}']=[np.nanmean(vi) if len(vi) else np.nan for vi in v]
                    df[f'median_{col}_gap{gap}']=[np.nanmedian(vi) if len(vi) else np.nan for vi in v]
                    df[f'max_{col}_gap{gap}']=[np.nanmax(vi) if len(vi) else np.nan for vi in v]
                    df[f'min_{col}_gap{gap}']=[np.nanmin(vi) if len(vi) else np.nan for vi in v]
                    df[f'std_{col}_gap{gap}']=[np.nanstd(vi) if len(vi) else np.nan for vi in v]
                    df[f'sum_{col}_gap{gap}']=[np.nansum(vi) if len(vi) else np.nan for vi in v]
                    
        if len(self.word2vec_models):#word2vec transform
            self.word2vec_cols=[]
            for (model,col,model_name) in self.word2vec_models:
                self.word2vec_cols.append(col)
            #set to duplicate removal
            self.word2vec_cols=list(set(self.word2vec_cols))
           
        if (mode=='train') and (self.use_high_corr_feat==False):#drop high correlation features
            print("< drop high correlation feature >")
            self.drop_high_correlation_feats(df)
        
        print("< drop useless cols >")
        total_drop_cols=self.nan_cols+self.unique_cols+self.object_cols+drop_cols
        total_drop_cols=[col for col in total_drop_cols if col not in self.word2vec_cols+self.labelencoder_cols]
        df.drop(total_drop_cols,axis=1,inplace=True,errors='ignore')
        df=self.reduce_mem_usage(df, float16_as32=True)
        print("-"*30)
        return df
    
    #Feature engineering that needs to be done internally in cross validation.
    def CV_FE(self,df,mode='train',fold=0):
        #labelencoder
        if len(self.labelencoder_colnames):
            print("< label encoder >")
            for col in self.labelencoder_colnames:
                #load model when model is existed,fit when model isn't exist.
                try:
                    le=self.pickle_load(self.load_path+f'le_{col}_fold{fold}.model')
                except:
                    value=df[col].values
                    le={}
                    for v in value:
                        if v in le.keys():
                            le[v]=len(le)
                    self.pickle_dump(le,f'le_{col}_fold{fold}.model')
                df[col+"_le"] = df[col].apply(lambda x:le.get(x,-1))

        if len(self.word2vec_models):
            print("< word2vec >")
            for (model,col,model_name) in self.word2vec_models:
                col=self.col2name[col]
                #load when model is existed.fit when model isn't existed.
                try:
                    model=self.pickle_load(self.load_path+f'{model_name}_{col}_fold{fold}.model')
                except:
                    model.fit(df[col])
                    self.pickle_dump(model,f'{model_name}_{col}_fold{fold}.model') 
                word2vec_feats=model.transform(df[col]).toarray()
                for i in range(word2vec_feats.shape[1]):
                    df[f"{col}_{model_name}_{i}"]=word2vec_feats[:,i]
        df.drop(self.word2vec_colnames+self.labelencoder_colnames,axis=1,inplace=True)
        #after this operation,df will be dropped into model,so we need to Convert object to floating-point numbers
        for col in df.columns:
            if (df[col].dtype==object):
                df[col]=df[col].astype(np.float32)
        #replace inf to nan
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df  
    
    def Metric(self,y_true,y_pred):#for multi_class,labeland proability
        #use cutom_metric when you define.
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
                y_pred=np.argmax(y_pred,axis=1)#transform probability to label
                return np.mean(y_true==y_pred)
            elif self.metric=='auc':
                return roc_auc_score(y_true,y_pred[:,1])
            elif self.metric=='f1_score':
                y_pred=np.argmax(y_pred,axis=1)#transform probability to label
                return f1_score(y_true, y_pred)
            elif self.metric=='mcc':
                y_pred=np.argmax(y_pred,axis=1)#transform probability to label
                return matthews_corrcoef(y_true, y_pred)
            elif self.metric=='logloss':
                eps=1e-15
                label=np.zeros_like(y_pred)
                for i in range(len(label)):
                    label[i][y_true[i]-1]=1
                y_true=label
                y_pred=np.clip(y_pred,eps,1-eps)
                return -np.mean(np.sum(y_true*np.log(y_pred),axis=-1))
    
    def optuna_lgb(self,X,y,group,kf,metric):
        def objective(trial):
            params = {
                "boosting_type": "gbdt","metric": metric,
                'random_state': self.seed,
                'n_estimators': trial.suggest_int('n_estimators', 500,1500),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
                'subsample': trial.suggest_float('subsample', 0.5, 1),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.5, log=True),
                'num_leaves' : trial.suggest_int('num_leaves', 8, 64),
                'min_child_samples': trial.suggest_int('min_child_samples', 2, 100),
                "extra_trees":True,
                "verbose": -1
            }
            if self.device in ['cuda','gpu']:#gpu mode when training
                params['device']='gpu'
                params['gpu_use_dp']=True
            model_name='lgb'
            if self.objective=='regression':
                model=LGBMRegressor(**params)
            else:
                model=LGBMClassifier(**params)
            oof_preds,metric_score=self.cross_validation(X,y,group,kf,model,model_name,use_optuna=True)
            return metric_score
        #'minimize' or 'maximize'
        if self.metric in ['accuracy','auc','f1_score','mcc']:
            direction='maximize'
        elif self.metric in ['medae','mae','rmse','mse','logloss','rmsle']:
            direction='minimize'
        else:
            direction=self.optuna_direction
            
        study = optuna.create_study(direction=direction, study_name='find best lgb_params') 
        study.optimize(objective, n_trials=self.use_optuna_find_params)
        best_params=study.best_trial.params
        best_params["boosting_type"]="gbdt"
        best_params["extra_trees"]=True
        best_params["metric"]=metric
        best_params['random_state']=self.seed
        best_params['verbose']=-1
        print(f"best_params={best_params}")
        return best_params
    
    # return oof_preds and metric_score
    # can use optuna to find params.If use optuna,then not save models.
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

            X_train=self.CV_FE(X_train,mode='train',fold=fold)
            X_valid=self.CV_FE(X_valid,mode='test',fold=fold)
            
            if (self.use_pseudo_label) and (type(self.test)==pd.DataFrame):
                test_copy=self.CV_FE(self.test.copy(),mode='test',fold=fold)
                test_X=test_copy.drop([self.group_col,self.target_col],axis=1,errors='ignore')
                test_y=test_copy[self.target_col]
                X_train=pd.concat((X_train,test_X),axis=0)
                y_train=pd.concat((y_train,test_y),axis=0)
            
            if 'lgb' in model_name:
                model.fit(X_train,y_train,eval_set=[(X_valid, y_valid)],
                         callbacks=[log_evaluation(log),early_stopping(self.early_stop)]
                    )
                if use_optuna==False:#print feature importance when not use optuna to find params.
                    #here we only care origin features in X.
                    columns,importances=[self.name2col[x] for x in list(X.columns)],model.feature_importances_[:len(X)]
                    useless_cols=[]
                    col2importance={}
                    for i in range(len(columns)):
                        if importances[i]==0:
                            useless_cols.append(columns[i])
                        else:
                            col2importance[columns[i]]=importances[i]
                    #descending order
                    col2importance = dict(sorted(col2importance.items(), key=lambda x: x, reverse=True))
                    print(f"feature_importance:{col2importance}")
                    print(f"useless_cols={useless_cols}")
            elif 'cat' in model_name:
                model.fit(X_train, y_train,
                      eval_set=(X_valid, y_valid),
                      early_stopping_rounds=self.early_stop, verbose=log)
            elif 'xgb' in model_name:
                model.fit(X_train,y_train,eval_set=[(X_valid, y_valid)],verbose=log)
            else:#other models
                model.fit(X_train,y_train) 

            if self.objective=='regression':
                oof_preds[valid_index]=model.predict(X_valid)
            else:
                oof_preds[valid_index]=model.predict_proba(X_valid)
            if not use_optuna:#not find_params(training)
                self.pickle_dump(model,f'{model_name}_fold{fold}.model')
                self.model_paths.append((model_name,fold))
        metric_score=self.Metric(y.values,oof_preds)
        return oof_preds,metric_score
    
    def drop_high_correlation_feats(self,df):
        #target_col and group_col is for model training,don't delete.object feature is string.
        #Here we choose 0.99,other feature with high correlation can use Dimensionality reduction such as PCA.
        #if you want to delete other feature with high correlation,add into drop_cols when init.
        numerical_cols=[col for col in df.columns if (col not in [self.target_col,self.group_col]) and df[col].dtype!=object]
        corr_matrix=df[numerical_cols].corr().values
        drop_cols=[]
        for i in range(len(corr_matrix)):
            for j in range(i+1,len(corr_matrix)):
                if abs(corr_matrix[i][j])>=0.99:
                    drop_cols.append(numerical_cols[j])
        #add drop_cols to self.drop_cols,they will be dropped in the final part of the function base_FE.
        print(f"drop_cols={drop_cols}")
        self.drop_cols+=drop_cols
    
    def fit(self,train_path_or_file='train.csv'):
        print("fit......")
        print("< load train data >")
        #read csv,parquet or csv_file
        self.train_path_or_file=train_path_or_file
        try:
            self.train=pl.read_csv(self.train_path_or_file)
            self.train=self.train.to_pandas()
        except:
            try:
                self.train=pl.read_parquet(self.train_path_or_file)
                self.train=self.train.to_pandas()
            except:#file.copy()
                self.train=self.train_path_or_file.copy()
        #polars to pandas.
        if isinstance(self.train, pl.DataFrame):
            self.train=self.train.to_pandas()
        if not isinstance(self.train, pd.DataFrame):
            raise ValueError("train_path_or_file is not pd.DataFrame")
        self.train=self.train.drop_duplicates()
        print(f"train.shape:{self.train.shape}")
        print("< Feature Engineer >")
        self.train=self.base_FE(self.train,mode='train',drop_cols=self.drop_cols)
        
        #choose cross validation
        if self.objective!='regression':
            if self.group_col!=None:#group
                kf=StratifiedGroupKFold(n_splits=self.num_folds,random_state=self.seed,shuffle=True)
            else:
                kf=StratifiedKFold(n_splits=self.num_folds,random_state=self.seed,shuffle=True)
        else:#regression
            if self.group_col!=None:#group
                kf=GroupKFold(n_splits=self.num_folds)
            else:
                kf=KFold(n_splits=self.num_folds,random_state=self.seed,shuffle=True)
        
        X=self.train.drop([self.group_col,self.target_col],axis=1,errors='ignore')
        y=self.train[self.target_col]
        
        #special characters in columns'name will lead to errors when GBDT model training.
        self.col2name={}
        self.name2col={}
        for i in range(len(list(X.columns))):
            self.col2name[list(X.columns)[i]]=f'col_{i}'
            self.name2col[f'col_{i}']=list(X.columns)[i]
        X=X.rename(columns=self.col2name)
        
        self.word2vec_colnames=[self.col2name[col] for col in self.word2vec_cols]
        self.labelencoder_colnames=[self.col2name[col] for col in self.labelencoder_cols]
        
        print(f"feature_count:{len(list(X.columns))}")
                
        #classification:target2idx,idx2target
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
        #save true label in train data to calculate final score  
        self.target=y.values
        
        #if you don't use your own models,then use built-in models.
        print("< load models >")
        if len(self.models)==0:
            
            metric=self.metric
            if self.objective=='multi_class':
                metric='multi_logloss'
            #lightgbm don't support f1_score,but we will calculate f1_score as Metric.
            if metric in ['f1_score','mcc','logloss']:
                metric='auc'
            elif metric=='medae':
                metric='mae'
            elif metric=='rmsle':
                metric='mse'
            if self.custom_metric!=None:
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
            #find new params then use optuna
            if self.use_optuna_find_params:
                lgb_params=self.optuna_lgb(X,y,group,kf,metric)
             
            #catboost's metric
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
            elif self.metric in ['auc','rmse','mcc','mae']:
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

            if self.device in ['cuda','gpu']:#gpu's name
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

        print("< model training >")
        for (model,model_name) in self.models:
            oof_preds,metric_score=self.cross_validation(X,y,group,kf,model,model_name,use_optuna=False)
            print(f"{self.metric}:{metric_score}")
            if self.save_oof_preds:#if oof_preds is needed
                np.save(f"{model_name}_seed{self.seed}_fold{self.num_folds}.npy",oof_preds)
        
    def predict(self,test_path_or_file='test.csv',weights=None,load_path=''):
        print("predict......")
        print("< weight normalization >")
        #if train and inference in different notebook,then
        self.load_path=load_path
        #weights:[1]*len(self.models)
        n=len(self.models)
        #if you don't set weights,then calculate mean value as result.
        if weights==None:
            weights=np.ones(n)
        if len(weights)!=n:
            raise ValueError(f"length of weights must be len(models)")
        weights=np.array([w for w in weights for f in range(self.num_folds)],dtype=np.float32)
        #normalization
        weights=weights*(self.num_folds*n)/np.sum(weights)

        #calculate oof score       
        oof_preds=np.zeros_like(np.load(f"{self.models[0//self.num_folds][1]}_seed{self.seed}_fold{self.num_folds}.npy"))
        for i in range(0,len(weights),self.num_folds):
            oof_pred=np.load(f"{self.models[i//self.num_folds][1]}_seed{self.seed}_fold{self.num_folds}.npy")
            oof_preds+=weights[i]*oof_pred
        oof_preds=oof_preds/n
        print(f"final_{self.metric}:{self.Metric(self.target,oof_preds)}")
        print("< load test data >")
        #parse csv,parquet
        try:
            self.test=pl.read_csv(test_path_or_file)
            self.test=self.test.to_pandas()
        except:
            try:#parquet
                self.test=pl.read_parquet(test_path_or_file)
                self.test=self.test.to_pandas()
            except:#file.copy
                self.test=test_path_or_file.copy()
        #polars to pandas
        if isinstance(self.test, pl.DataFrame):
            self.test=self.test.to_pandas()
        if not isinstance(self.test, pd.DataFrame):
            raise ValueError("test_path_or_file is not pd.DataFrame")
        print(f"test.shape:{test.shape}")
        print("< Feature Engineer >")
        self.test=self.base_FE(self.test,mode='test',drop_cols=self.drop_cols)
        self.test=self.test.drop([self.group_col,self.target_col],axis=1,errors='ignore')
        self.test=self.test.rename(columns=self.col2name)
        if self.objective=='regression':
            print("< prediction on test data >")
            test_preds=np.zeros((len(self.models)*self.num_folds,len(self.test)))
            cnt=0
            for (model_name,fold) in self.model_paths:
                model=self.pickle_load(self.load_path+f'{model_name}_fold{fold}.model')
                test_copy=self.CV_FE(self.test.copy(),mode='test',fold=fold)
                test_pred=np.zeros(len(self.test))
                for i in range(0,len(self.test),self.infer_size):
                    test_pred[i:i+self.infer_size]=model.predict(test_copy[i:i+self.infer_size])
                test_preds[cnt]=test_pred
                cnt+=1
            test_preds=np.mean([test_preds[i]*weights[i] for i in range(len(test_preds))],axis=0)
            
            #use pseudo label
            if self.use_pseudo_label:
                self.test[self.target_col]=test_preds
                self.model_paths=[]
                self.fit(self.train_path_or_file)
                
                test_preds=np.zeros((len(self.models)*self.num_folds,len(self.test)))
                cnt=0
                for (model_name,fold) in self.model_paths:
                    model=self.pickle_load(self.load_path+f'{model_name}_fold{fold}.model')
                    test_copy=self.CV_FE(self.test.copy(),mode='test',fold=fold)
                    test_pred=np.zeros(len(self.test))
                    for i in range(0,len(self.test),self.infer_size):
                        test_pred[i:i+self.infer_size]=model.predict(test_copy.drop([self.target_col],axis=1)[i:i+self.infer_size])
                    test_preds[cnt]=test_pred
                    cnt+=1
                test_preds=np.mean([test_preds[i]*weights[i] for i in range(len(test_preds))],axis=0)
            
            if self.save_test_preds:
                np.save('test_preds.npy',test_preds)
            return test_preds
        else:#classification 
            test_preds=np.zeros((len(self.models)*self.num_folds,len(self.test),self.num_classes))
            cnt=0
            for (model_name,fold) in self.model_paths:
                model=self.pickle_load(self.load_path+f'{model_name}_fold{fold}.model')
                test_copy=self.CV_FE(self.test.copy(),mode='test',fold=fold)
                test_pred=np.zeros((len(self.test),self.num_classes))
                for i in range(0,len(self.test),self.infer_size):
                    test_pred[i:i+self.infer_size]=model.predict_proba(test_copy[i:i+self.infer_size])
                test_preds[cnt]=test_pred
                cnt+=1   
            test_preds=np.mean([test_preds[i]*weights[i] for i in range(len(test_preds))],axis=0)#(len(test),self.num_classes)
            
            #use pseudo label
            if self.use_pseudo_label:
                self.test[self.target_col]=np.argmax(test_preds,axis=1)
                self.model_paths=[]
                self.fit(self.train_path_or_file)

                test_preds=np.zeros((len(self.models)*self.num_folds,len(self.test),self.num_classes))
                fold=0
                for (model_name,fold) in self.model_paths:
                    model=self.pickle_load(self.load_path+f'{model_name}_fold{fold}.model')
                    test_copy=self.CV_FE(self.test.copy(),mode='test',fold=fold)
                    test_pred=np.zeros((len(self.test),self.num_classes))
                    for i in range(0,len(self.test),self.infer_size):
                        test_pred[i:i+self.infer_size]=model.predict_proba(test_copy.drop([self.target_col],axis=1)[i:i+self.infer_size])
                    test_preds[fold]=test_pred
                    fold+=1
                test_preds=np.mean([test_preds[i]*weights[i] for i in range(len(test_preds))],axis=0)
            if self.metric=='auc':
                return test_preds[:,1]
            if self.save_test_preds:
                np.save('test_preds.npy',test_preds)
            test_preds=np.argmax(test_preds,axis=1)
            return test_preds

    #ensemble some solutions.
    def ensemble(self,solution_paths_or_files,weights=None):
        #If you don't set weights,then use mean value as result.
        n=len(solution_paths_or_files)
        if weights==None:
            weights=np.ones(n)
        if len(weights)!=n:
            raise ValueError(f"length of weights must be len(solution_paths_or_files)")
        #normalization
        weights=weights/np.sum(weights)

        #Weighted Sum of Continuous Values
        if (self.objective=='regression') or(self.metric=='auc'):
            final_solutions=[]
            for i in range(n):
                try:
                    solution=pl.read_csv(solution_paths_or_files[i])
                    solution=solution.to_pandas()
                except:#csv_file
                    solution=solution_paths_or_files[i]
                if not isinstance(solution, pd.DataFrame):
                    raise ValueError("solution_paths_or_files is not pd.DataFrame")
                final_solutions.append(weights[i]*solution[self.target_col].values)
            final_solutions=np.sum(final_solutions,axis=0)
            return final_solutions
        else:#classification find mode
            #n solutions,m datas
            solutions=[]
            for i in range(n):
                try:
                    solution=pl.read_csv(solution_paths_or_files[i])
                    solution=solution.to_pandas()
                except:#csv_file
                    solution=solution_paths_or_files[i]
                if not isinstance(solution, pd.DataFrame):
                    raise ValueError("solution_paths_or_files is not pd.DataFrame")
                solutions.append(solution[self.target_col].values)
            final_solutions=[]
            for i in range(len(solutions[0])):
                solution2count={}
                #data[i] solution[j]
                for j in range(n):
                    if solutions[j][i] in solution2count.keys():
                        solution2count[ solutions[j][i] ]+=weights[j]
                    else:
                        solution2count[ solutions[j][i] ]=weights[j]
                solution2count=dict(sorted(solution2count.items(),key=lambda x:-x[1]))
                final_solutions.append(list(solution2count.keys())[0])
            final_solutions=np.array(final_solutions)
            return final_solutions

    #save test_preds to submission.csv
    def submit(self,submission_path='submission.csv',test_preds=None,save_name='yunbase'):
        submission=pd.read_csv(submission_path)
        submission[self.target_col]=test_preds
        if self.objective!='regression':
            if self.metric!='auc':
                submission[self.target_col]=submission[self.target_col].apply(lambda x:self.idx2target[x])
        submission.to_csv(f"{save_name}.csv",index=None)

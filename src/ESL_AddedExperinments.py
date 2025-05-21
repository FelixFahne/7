#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import sklearn
import os
import collections
from sklearn import metrics,model_selection,linear_model
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LinearRegression,BayesianRidge,LogisticRegression
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[3]:


def transform(x):
    if x=='Unknown':
        return 'Dialogue level'
    else:
        return x

def assign_tone(row):
    if row['backchannels'] > 0 or row['code-switching for communicative purposes'] > 0 or row['collaborative finishes'] > 0:
        return 'Informal'
    elif row['subordinate clauses'] > 0 or row['impersonal subject + non-factive verb + NP'] > 0:
        return 'Formal'
    else:
        return 'Neutral'  # Default to Neutral if none of the criteria match


def get_new_labels(counts_df,new_column='P-V',old_column=['topic extension with the same content',
                                                          'topic extension under the previous direction']):
    cc_dd=counts_df[old_column].astype(int).values
    cc_dd=['-'.join([str(j) for j in i]) for i in cc_dd]
    counts_df[new_column]=cc_dd
    return counts_df

def get_result(test_y,test_pred):
    acc=metrics.accuracy_score(test_y,test_pred)
    pre=metrics.precision_score(test_y,test_pred)
    rec=metrics.recall_score(test_y,test_pred)
    f1s=metrics.f1_score(test_y,test_pred)
    return [acc,pre,rec,f1s]

def get_result(test_y,test_pred):
    acc=metrics.accuracy_score(test_y,test_pred)
    pre=metrics.precision_score(test_y,test_pred,average='weighted')
    rec=metrics.recall_score(test_y,test_pred,average='weighted')
    f1s=metrics.f1_score(test_y,test_pred,average='weighted')
    return [acc,pre,rec,f1s]

def get_result_2(test_y,test_pred):
    mae=metrics.mean_absolute_error(test_y,test_pred)
    mse=metrics.mean_squared_error(test_y,test_pred)
    mape=metrics.mean_absolute_percentage_error(test_y,test_pred)
    r2=metrics.r2_score(test_y,test_pred)
    return [mae,mse,mape,r2]

def get_normal(data):
    #============归一化===============
    normal_list=[]
    for i in data.values:
        if i[2]=='Token level':
            norm=len(i[3].split('&&&&')[1].split())/Token_level_label_count[i[1]]
        elif i[2]=='Utterance level':
            norm=len(i[3].split('&&&&')[1].split())/Utterance_level_label_count[i[1]]
        else:
            norm=-1
        normal_list.append(norm)

    data['normal']=normal_list
    return data


score_dict={'topic extension with clear new context':[5,'Y1'],
           'topic extension under the previous direction':[4,'Y1'],
           'topic extension with the same content':[3,'Y1'],
            'repeat and no topic extension':[2,'Y1'],
            'no topic extension and stop the topic at this point':[1,'Y1'],

            'overall tone choice: very informal':[5,'Y2'],
           'overall tone choice: quite informal, but some expressions are still formal':[4,'Y2'],
           'overall tone choice: quite formal and some expressions are not that formal':[3,'Y2'],
            'overall tone choice: quite formal and some expressions are not that formal':[2,'Y2'],
            'overall tone choice: very formal':[1,'Y2'],

            'Co1':[5,'Y3'],
           'Co2':[4,'Y3'],
           'Co3':[3,'Y3'],
            'Co4':[2,'Y3'],
            'Co5':[1,'Y3'],

           'Cc1':[5,'Y4'],
           'Cc2':[4,'Y4'],
           'Cc3':[3,'Y4'],
            'Cc4':[2,'Y4'],
            'Cc5':[1,'Y4'],

            'conversation opening':[3,'Y3'],
            'onversation closing':[3,'Y4']
           }


# In[4]:


data_path='./data_csv_sample/'
data=pd.DataFrame({})
for file in os.listdir(data_path):
    file_path=data_path+file
    data_i=pd.read_csv(file_path)
    data=pd.concat([data,data_i])


new_labellevel=[]
for i in data.values:
    _,i_label,i_labellevel,_=i
    if i_labellevel=='Unknown':
        if i_label=='adjectives/ adverbs expressing possibility':
            new_v='Utterance level'
        else:
            new_v='Dialogue level'
    else:
        new_v=i_labellevel

    new_labellevel.append(new_v)

data['LabelLevel']=new_labellevel
# data['LabelLevel']=data['LabelLevel'].apply(transform)
data.index=range(len(data))


# In[ ]:


#=============统计频次============
Token_level=data[data['LabelLevel']=='Token level']
Token_level_label_count=collections.Counter(Token_level['Label'])

Utterance_level=data[data['LabelLevel']=='Utterance level']
Utterance_level_label_count=collections.Counter(Utterance_level['Label'])

Dialogue_level=data[data['LabelLevel']=='Dialogue level']
Dialogue_level_label_count=collections.Counter(Dialogue_level['Label'])

print(len(Utterance_level_label_count)+len(Token_level_label_count))
valid_keys_list=list(Utterance_level_label_count.keys())+list(Token_level_label_count.keys())


# In[ ]:


all_people_df=pd.DataFrame({})
all_labels_df=pd.DataFrame({})
all_classs_df=pd.DataFrame({})
for index in range(0,len(data),10):
    index_data=data[index:index+10]
    index_data['word_num']=[len(i.split('&&&&')[1].split()) for i in index_data['Content']]#统计各句子的特殊符号之间的word数

    #===========================================================提取feature=================
    #===========统计每一个Label对应的平均词汇数============
    test_feature={}
    for label in index_data['Label'].unique():
        index_data_label=index_data[index_data['Label']==label]
        index_data_label_1=pd.DataFrame([collections.Counter(index_data_label['word_num'])],index=[label])/len(index_data_label)
        test_feature[label]=sum(np.array(index_data_label_1.columns)*index_data_label_1.values[0])/(index_data_label_1.sum().sum())
    test_feature=pd.DataFrame([test_feature],index=[index])
    all_people_df=pd.concat([all_people_df,test_feature])#===========特征矩阵

    #===========================================================提取score=================
    #===========查表获取分数===========
    new_score_list=[]
    for label in index_data['Label']:
        if label in score_dict:
            new_score=score_dict[label]
        else:
            new_score=[None,None]
        new_score_list.append(new_score)
    index_data[['score','type']]=new_score_list

    #============求取平均分数=============
    type_score={}
    type_class={}
    for type_ in ['Y1','Y2','Y3','Y4']:
        type_data=index_data[index_data['type']==type_]
        if len(type_data)==0:
            type_score[type_]=3
        else:
            type_score[type_]=index_data['score'].mean()

        type_data_big3=type_data[type_data['score']>3]
        type_data_sma3=type_data[type_data['score']<3]
        if len(type_data_big3)+len(type_data_sma3)==0:
            type_class[type_]=3
        else:
            if len(type_data_sma3)>0:
                type_data_sma3=type_data_sma3.sort_values(by=['score','word_num'])
                type_data_sma3_s=type_data_sma3.values[0,-2]
                type_data_sma3=type_data_sma3[type_data_sma3['score']==type_data_sma3_s]
                type_data_sma3_w=type_data_sma3.values[-1,-3]
                type_data_sma3=type_data_sma3[type_data_sma3['word_num']==type_data_sma3_w]
            if len(type_data_big3)>0:
                type_data_big3=type_data_big3.sort_values(by=['score','word_num'])
                type_data_big3_s=type_data_big3.values[-1,-2]
                type_data_big3=type_data_big3[type_data_big3['score']==type_data_big3_s]
                type_data_big3_w=type_data_big3.values[-1,-3]
                type_data_big3=type_data_big3[type_data_big3['word_num']==type_data_big3_w]
            type_data_concat=pd.concat([type_data_sma3,type_data_big3])
            type_data_concat=type_data_concat.sort_values(by=['word_num'])
            type_data_concat_s=int(type_data_concat.values[-1,-2])
            type_class[type_]=type_data_concat_s


    type_score_df=pd.DataFrame([type_score],index=[index])
    all_labels_df=pd.concat([all_labels_df,type_score_df])#===========分数矩阵



    type_class_df=pd.DataFrame([type_class],index=[index])
    all_classs_df=pd.concat([all_classs_df,type_class_df])#===========分数矩阵

data_x=all_people_df[valid_keys_list]
data_x=data_x.fillna(0)
data_y=all_labels_df.copy()

columns_list=[]
for i in range(1,5):
    for j in range(1,6):
        columns_list.append(f'Y{i}_{j}')

data_class=all_classs_df.copy()


# In[ ]:


train_x,test_x,train_y,test_y,train_c,test_c=train_test_split(data_x,data_y,data_class,train_size=0.8)
train_x_normal=(train_x-train_x.mean())/train_x.std()
test_x_normal=(test_x-train_x.mean())/train_x.std()


# In[ ]:


train_x_normal.isna().sum().sum(),test_x_normal.isna().sum().sum()


# In[8]:


topk=10


# In[9]:


def save_csv(data,path):
    data.to_csv(f'./{path}.csv',encoding='utf_8_sig')


# # Task1

# In[10]:


model_dict={'Linear':LinearRegression(),'Logit':LogisticRegression(),
           'RF_R':RandomForestRegressor(),'RF_C':RandomForestClassifier(),
           'NB_R':BayesianRidge(),'NB_C':BernoulliNB()}


# In[11]:


def get_all_result(train_x_normal,train_y,test_x_normal,test_y,model_name):
    global model
    test_pred_dict={}
    test_true_dict={}
    train_pred_dict={}
    train_true_dict={}
    train_result_linear={}
    test_result_linear={}
    model_infortance=pd.DataFrame({})
    for co in train_y.columns:
        train_true=train_y[co]
        test_true=test_y[co]
        model=model_dict[model_name]
        model.fit(train_x_normal,train_true)#训练

        train_pred=model.predict(train_x_normal)#预测
        test_pred=model.predict(test_x_normal)

        #整理
        if len(train_pred)==1:
            train_pred_dict[co]=train_pred
            test_pred_dict[co]=test_pred
        else:
            train_pred_dict[co]=train_pred[0]
            test_pred_dict[co]=test_pred[0]
        train_true_dict[co]=train_true
        test_true_dict[co]=test_true

        if model_name in ['Linear','NB_R','RF_R']:
            result_func=get_result_2
            metrics_list=['mae','mse','mape','r2']
        else:
            result_func=get_result
            metrics_list=['acc','pre','rec','f1s']
        train_result_linear[co]=result_func(train_true,train_pred)#评估
        test_result_linear[co]=result_func(test_true,test_pred)

        #--------得到重要度---------
        importance=None
        if model_name in ['Linear','NB_R']:
            importance=pd.DataFrame(model.coef_,index=model.feature_names_in_,columns=[co])
        elif model_name in ['RF_C','RF_R']:
            importance=pd.DataFrame(model.feature_importances_,index=model.feature_names_in_,columns=[co])
        elif model_name in ['Logit','NB_C']:
            importance=pd.DataFrame(model.coef_[0],index=model.feature_names_in_,columns=[co])

        importance_=abs(importance)
        importance_=importance_.sort_values(by=[co])
        importance=importance.loc[importance_[-topk:].index]
        model_infortance=pd.concat([model_infortance,importance],axis=1)

    train_result=pd.DataFrame(train_result_linear,index=metrics_list)
    test_result=pd.DataFrame(test_result_linear,index=metrics_list)
    test_pred_df=pd.DataFrame(test_pred_dict,index=test_y.index)
    test_true_df=pd.DataFrame(test_true_dict,index=test_y.index)
    train_pred_df=pd.DataFrame(train_pred_dict,index=train_y.index)
    train_true_df=pd.DataFrame(train_true_dict,index=train_y.index)
    return [train_result,test_result],[test_true_df,test_pred_df],[train_true_df,train_pred_df],model_infortance


# In[12]:


#================Linear===========
model_name='Linear'
result,test_true_pred,train_true_pred,importance=get_all_result(train_x_normal,train_y,
                                                                test_x_normal,test_y,
                                                                model_name)
importance,result


# In[13]:


#=================RF===============
model_name='RF_R'
result,test_true_pred,train_true_pred,importance=get_all_result(train_x_normal,train_y,
                                                                test_x_normal,test_y,
                                                                model_name)
importance,result[1]


# In[14]:


#=================RF===============
model_name='NB_R'
result,test_true_pred,train_true_pred,importance=get_all_result(train_x_normal,train_y,
                                                                test_x_normal,test_y,
                                                                model_name)
importance,result[1]


# In[15]:


#===============BP网络=============
def get_model_R(input_dim,output_dim):
    inputs=tf.keras.layers.Input(shape=(input_dim,))
    dense=tf.keras.layers.Dense(64,activation='relu')(inputs)
    dense=tf.keras.layers.Dense(64,activation='relu')(dense)
    outputs=tf.keras.layers.Dense(output_dim,activation='sigmoid')(dense)

    model=tf.keras.models.Model(inputs,outputs)
    model.compile(optimizer='adam',loss='mse')
    return model

def get_model_C(input_dim,output_dim):
    inputs=tf.keras.layers.Input(shape=(input_dim,))
    dense=tf.keras.layers.Dense(64,activation='relu')(inputs)
    dense=tf.keras.layers.Dense(64,activation='relu')(dense)
    outputs=tf.keras.layers.Dense(5,activation='softmax')(dense)

    model=tf.keras.models.Model(inputs,outputs)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
    return model


# In[16]:


def get_BP_result(train_x_normal,train_y,test_x_normal,test_y,model_name):
    input_dim=train_x_normal.shape[1]
    output_dim=train_y.shape[1]

    if model_name=='BP_R':
        train_y_normal=train_y/5
        model_bp=get_model_R(input_dim,output_dim)
    else:
        train_y_normal=train_y.copy()
        model_bp=get_model_C(input_dim,output_dim)

    stopearly=tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True,monitor='val_loss')
    history=model_bp.fit(train_x_normal,train_y_normal,
                         validation_split=0.1,batch_size=16,
                         epochs=100,callbacks=[stopearly],verbose=0)

    test_pred_bp=model_bp.predict(test_x_normal)
    train_pred_bp=model_bp.predict(train_x_normal)

    if model_name=='BP_R':
        test_pred_bp=test_pred_bp*5
        train_pred_bp=train_pred_bp*5
    else:
        train_pred_bp=(train_pred_bp>=0.5).astype(int)
        test_pred_bp=(test_pred_bp>=0.5).astype(int)

    test_pred_bp=pd.DataFrame(test_pred_bp,index=test_y.index,columns=test_y.columns)
    test_true_bp=test_y.copy()
    train_pred_bp=pd.DataFrame(train_pred_bp,index=train_y.index,columns=train_y.columns)
    train_true_bp=train_y.copy()

    train_result_dict={}
    test_result_dict={}

    if model_name in ['BP_R']:
        result_func=get_result_2
        metrics_list=['mae','mse','mape','r2']
    else:
        result_func=get_result
        metrics_list=['acc','pre','rec','f1s']

    for co in train_pred_bp.columns:
        true_train=train_true_bp[co]
        pred_train=train_pred_bp[co]
        true_test=test_true_bp[co]
        pred_test=test_pred_bp[co]

        result_train=result_func(true_train,pred_train)
        result_test=result_func(true_test,pred_test)
        train_result_dict[co]=result_train
        test_result_dict[co]=result_test

    train_result=pd.DataFrame(train_result_dict,index=metrics_list)
    test_result=pd.DataFrame(test_result_dict,index=metrics_list)
    model_infortance=None
    return [train_result,test_result],[test_true_bp,test_pred_bp],[train_true_bp,train_pred_bp],model_infortance


# In[17]:


model_name='BP_R'
result,test_true_pred,train_true_pred,importance=get_BP_result(train_x_normal,train_y,
                                                               test_x_normal,test_y,model_name)
result


# # Task2

# In[46]:


task_2_importanc={}
task_2_test_true_pred={}
task_2_train_true_pred={}


# In[23]:


def get_result(test_y,test_pred):
    acc=metrics.accuracy_score(test_y,test_pred)
    pre=metrics.precision_score(test_y,test_pred,average='weighted')
    rec=metrics.recall_score(test_y,test_pred,average='weighted')
    f1s=metrics.f1_score(test_y,test_pred,average='weighted')
    return [acc,pre,rec,f1s]


# In[47]:


model_name='Logit'
result,test_true_pred,train_true_pred,importance=get_all_result(train_x_normal,train_c,
                                                                test_x_normal,test_c,
                                                                model_name)
task_2_importanc[model_name]=importance
task_2_test_true_pred[model_name]=test_true_pred
task_2_train_true_pred[model_name]=train_true_pred
importance,result


# In[49]:


model_name='RF_C'
result,test_true_pred,train_true_pred,importance=get_all_result(train_x_normal,train_c,
                                                                test_x_normal,test_c,
                                                                model_name)
task_2_importanc[model_name]=importance
task_2_test_true_pred[model_name]=test_true_pred
task_2_train_true_pred[model_name]=train_true_pred
importance,result


# In[50]:


model_name='NB_C'
result,test_true_pred,train_true_pred,importance=get_all_result(train_x_normal,train_c,
                                                                test_x_normal,test_c,
                                                                model_name)
task_2_importanc[model_name]=importance
task_2_test_true_pred[model_name]=test_true_pred
task_2_train_true_pred[model_name]=train_true_pred
importance,result


# In[21]:


def get_model_C(input_dim,output_dim):
    inputs=tf.keras.layers.Input(shape=(input_dim,))
    dense=tf.keras.layers.Dense(64,activation='relu')(inputs)
    dense=tf.keras.layers.Dense(64,activation='relu')(dense)
    outputs=tf.keras.layers.Dense(5,activation='softmax')(dense)

    model=tf.keras.models.Model(inputs,outputs)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
    return model


# In[22]:


def get_BP_result_C(train_x_normal,train_c,test_x_normal,test_c,model_name):
    input_dim=train_x_normal.shape[1]
    output_dim=train_c.shape[1]

    result_func=get_result
    metrics_list=['acc','pre','rec','f1s']

    train_pred_dict,test_pred_dict={},{}
    train_result_dict,test_result_dict={},{}
    for co in train_c.columns:
        train_y_normal=train_c[[co]]-1
        test_y_normal=test_c[[co]]-1
        model_bp=get_model_C(input_dim,output_dim)
        stopearly=tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True,monitor='val_loss')

        history=model_bp.fit(train_x_normal,train_y_normal,
                         validation_split=0.1,batch_size=16,
                         epochs=100,callbacks=[stopearly],verbose=0)

        test_pred_bp=model_bp.predict(test_x_normal)
        train_pred_bp=model_bp.predict(train_x_normal)

        train_pred_bp=train_pred_bp.argmax(1)+1
        test_pred_bp=test_pred_bp.argmax(1)+1
        train_pred_dict[co]=train_pred_bp
        test_pred_dict[co]=test_pred_bp

        train_result_dict[co]=result_func(train_y_normal+1,train_pred_bp)
        test_result_dict[co]=result_func(test_y_normal+1,test_pred_bp)

    train_result=pd.DataFrame(train_result_dict,index=metrics_list)
    test_result=pd.DataFrame(test_result_dict,index=metrics_list)

    train_pred_bp=pd.DataFrame(train_pred_dict,index=train_y_normal.index)
    test_pred_bp=pd.DataFrame(test_pred_dict,index=test_y_normal.index)
    train_true_bp=train_c.copy()
    test_true_bp=test_c.copy()
    return [train_result,test_result],[test_true_bp,test_pred_bp],[train_true_bp,train_pred_bp],None


# In[23]:


model_name='BP_C'
result,test_true_pred,train_true_pred,importance=get_BP_result_C(train_x_normal,train_c,
                                                               test_x_normal,test_c,model_name)
result


# In[40]:


#=============整理topk重要度的集合===============
type_df=collections.defaultdict(dict)
for model_n,df in task_2_importanc.items():
    for type_ in df.columns:
        type_df[type_][model_n]=df[type_]

for type_,df in type_df.items():
    df=pd.DataFrame(df)
    save_csv(df,f'task_2_{type_}_importance_df')

#===========预测结果与真实结果的t-test检验==============
def get_t_test_result(task_2_train_true_pred):
    from scipy import stats
    import scipy.stats
    t_test_result={}
    for model_n,true_pred in task_2_train_true_pred.items():
        true,pred=true_pred
        type_t={}
        for type_ in true.columns:
            true_t=true[type_].values
            pred_t=pred[type_].values
            t, pval = scipy.stats.ttest_ind(true_t, pred_t)
            type_t[type_]=pval
        t_test_result[model_n]=type_t
    t_test_result=pd.DataFrame(t_test_result)
    return t_test_result

t_test_result=get_t_test_result(task_2_test_true_pred)
t_train_result=get_t_test_result(task_2_train_true_pred)
save_csv(t_test_result,'t_test_result_test')
save_csv(t_train_result,'t_test_result_train')


# # Task3：替换x，只用Token

# In[24]:


train_x_normal_=train_x_normal[list(Token_level_label_count.keys())]
test_x_normal_=test_x_normal[list(Token_level_label_count.keys())]


# In[25]:


#================Linear===========
model_name='Linear'
result,test_true_pred,train_true_pred,importance=get_all_result(train_x_normal_,train_y,
                                                                test_x_normal_,test_y,
                                                                model_name)
importance,result


# In[26]:


#=================RF===============
model_name='RF_R'
result,test_true_pred,train_true_pred,importance=get_all_result(train_x_normal_,train_y,
                                                                test_x_normal_,test_y,
                                                                model_name)
importance,result[1]


# In[27]:


#=================RF===============
model_name='NB_R'
result,test_true_pred,train_true_pred,importance=get_all_result(train_x_normal_,train_y,
                                                                test_x_normal_,test_y,
                                                                model_name)
importance,result[1]


# In[28]:


model_name='BP_R'
result,test_true_pred,train_true_pred,importance=get_BP_result(train_x_normal_,train_y,
                                                               test_x_normal_,test_y,model_name)
result


# In[29]:


model_name='Logit'
result,test_true_pred,train_true_pred,importance=get_all_result(train_x_normal_,train_c,
                                                                test_x_normal_,test_c,
                                                                model_name)
importance,result


# In[30]:


model_name='RF_C'
result,test_true_pred,train_true_pred,importance=get_all_result(train_x_normal_,train_c,
                                                                test_x_normal_,test_c,
                                                                model_name)
importance,result


# In[31]:


model_name='NB_C'
result,test_true_pred,train_true_pred,importance=get_all_result(train_x_normal_,train_c,
                                                                test_x_normal_,test_c,
                                                                model_name)
importance,result


# In[32]:


model_name='BP_C'
result,test_true_pred,train_true_pred,importance=get_BP_result_C(train_x_normal_,train_c,
                                                               test_x_normal_,test_c,model_name)
result


# # task4：改变x，使用Utrance

# In[33]:


train_x_normal_=train_x_normal[list(Utterance_level_label_count.keys())]
test_x_normal_=test_x_normal[list(Utterance_level_label_count.keys())]


# In[34]:


#================Linear===========
model_name='Linear'
result,test_true_pred,train_true_pred,importance=get_all_result(train_x_normal_,train_y,
                                                                test_x_normal_,test_y,
                                                                model_name)
importance,result


# In[35]:


#=================RF===============
model_name='RF_R'
result,test_true_pred,train_true_pred,importance=get_all_result(train_x_normal_,train_y,
                                                                test_x_normal_,test_y,
                                                                model_name)
importance,result[1]


# In[36]:


#=================RF===============
model_name='NB_R'
result,test_true_pred,train_true_pred,importance=get_all_result(train_x_normal_,train_y,
                                                                test_x_normal_,test_y,
                                                                model_name)
importance,result[1]


# In[37]:


model_name='BP_R'
result,test_true_pred,train_true_pred,importance=get_BP_result(train_x_normal_,train_y,
                                                               test_x_normal_,test_y,model_name)
result


# In[38]:


model_name='Logit'
result,test_true_pred,train_true_pred,importance=get_all_result(train_x_normal_,train_c,
                                                                test_x_normal_,test_c,
                                                                model_name)
importance,result


# In[39]:


model_name='RF_C'
result,test_true_pred,train_true_pred,importance=get_all_result(train_x_normal_,train_c,
                                                                test_x_normal_,test_c,
                                                                model_name)
importance,result


# In[40]:


model_name='NB_C'
result,test_true_pred,train_true_pred,importance=get_all_result(train_x_normal_,train_c,
                                                                test_x_normal_,test_c,
                                                                model_name)
importance,result


# In[41]:


model_name='BP_C'
result,test_true_pred,train_true_pred,importance=get_BP_result_C(train_x_normal_,train_c,
                                                               test_x_normal_,test_c,model_name)
result


# In[ ]:





# In[ ]:





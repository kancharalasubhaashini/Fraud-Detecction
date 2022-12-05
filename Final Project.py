#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit,KFold
import time
plt.style.use(style="seaborn")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


get_ipython().run_cell_magic('time', '', 'train_identity=pd.read_csv(r"train_identity.csv.zip")\ntrain_transaction=pd.read_csv(r"train_transaction.csv.zip")\ntest_identity=pd.read_csv(r"test_identity.csv.zip")\ntest_transaction=pd.read_csv(r"test_transaction.csv.zip")')


# In[3]:


train_identity.info()
train_transaction.info()


# Downcast types to reduce memory usage

# In[4]:


def downcast_dtypes(df):
    _start=df.memory_usage(deep=True).sum()/1024**2
    float_cols=[c for c in df if df[c].dtypes=="float64"]
    int_cols=[c for c in df if df[c].dtype in ["int64","int32"]]
    df[float_cols]=df[float_cols].astype(np.float32)
    df[int_cols]=df[int_cols].astype(np.int16)
    _end=df.memory_usage(deep=True).sum()/1024**2
    saved=(_start-_end)/_start*100
    print(f"Saved{saved:.2f}%")
    return df


# In[5]:


train_identity=downcast_dtypes(train_identity)
train_transaction=downcast_dtypes(train_transaction)
test_identity=downcast_dtypes(test_identity)
test_transaction=downcast_dtypes(test_transaction)


# In[6]:


train_identity.info()
train_transaction.info()


# In[7]:


get_ipython().run_cell_magic('time', '', 'train_identity=pd.read_csv(r"train_identity.csv.zip")\ntrain_transaction=pd.read_csv(r"train_transaction.csv.zip")\ntest_identity=pd.read_csv(r"test_identity.csv.zip")\ntest_transaction=pd.read_csv(r"test_transaction.csv.zip")')


# In[8]:


#merging
train=pd.merge(train_transaction,train_identity,on="TransactionID",how='left')
test=pd.merge(test_transaction,test_identity,on="TransactionID",how='left')
del train_transaction,train_identity,test_transaction,test_identity


# In[9]:


print("Train  are cols {} and rows {}".format(train.shape[0], train.shape[1]))
print("test are cols {} and rows {}".format(test.shape[0], test.shape[1]))


# In[10]:


train.describe()


# In[11]:


train.describe(include=['O'])


# In[12]:


train[["ProductCD", "isFraud"]].groupby(['ProductCD'], as_index=False).mean().sort_values(by='isFraud', ascending=False)


# In[13]:


plt.pie(train.isFraud.value_counts(), labels=['Not Fraud', 'Fraud'],autopct='%0.1f%%')
plt.axis('equal')
plt.show()


# In[14]:


sns.countplot(test['card6'])
plt.show()


# In[15]:


train[["card6", "isFraud"]].groupby(['card6'], as_index=False).mean().sort_values(by='isFraud', ascending=False)


# In[16]:


sns.countplot(test['card4'])
plt.show()


# In[17]:


train[["card4", "isFraud"]].groupby(['card4'], as_index=False).mean().sort_values(by='isFraud', ascending=False)


# In[18]:


sns.countplot(train['ProductCD'])
plt.show()


# In[19]:


total_fraud = train.loc[(train['isFraud'] == 1),].shape[0]
fraud_percentage = (total_fraud*100)/train.shape[0]
print("Total fraud percentage ",format(fraud_percentage,'.2f'),"%")


# In[20]:


#total fraud amount
format(train.loc[(train['isFraud'] == 1),'TransactionAmt'].sum(), '.2f')


# In[21]:


#subset of columns
imp_features=["TransactionAmt",
 "productCD","card1","card2","card3","card5","card6","addr1","addr2",
 "dist1","dist2","P_emaildomain","R_emaildomain","C1","C2","C4","C5",
 "C6","C7","C8","C9","C10","C11","C12","C13","C14","D1","D2","D3","D4",
 "D5","D10","D11","D15","M1","M2","M3","M4","M6","M7","M8","M9","V1",
 "V3","V4","V6","V8","V11","V13","V14","V17","V20","V23","V26","V27",
 "V30","V36","V37","V40","V41","V44","V47","V48","V54","V56","V59",
 "V62","V65","V67","V68","V70","V76","V78","V80","V82","V86","V88",
 "V89","V91","V107","V108","V111","V115","V117","V120","V121","V123",
 "V124","V127","V129","V130","V136","V138","V139","V142","V156","V160",
 "V162","V165","V166","V169","V171","V173","V175","V176","V178","V180",
 "V182","V185","V187","V188","V198","V203","V205",
 "V207","V209","V210","V215","V218","V220","V221","V223","V224",
 "V226","V228","V229","V234","V235","V238","V240","V250","V252",
 "V253","V257","V258","V260","V261","V264","V266","V267","V271",
 "V274","V277","V281","V283","V284","V285","V286","V289","V291",
 "V294","V296","V297","V301","V303","V305","V307","V309","V310",
"V314","V320","DeviceType","DeviceInfo","isFraud",]


# In[22]:


len(imp_features)


# In[23]:


cols_to_drop_train=[col for col in train.columns if col not in imp_features]
cols_to_drop_test=[col for col in test.columns if col not in imp_features]
print(f"{len(cols_to_drop_train)} features from train are going to be dropped")
print(f"{len(cols_to_drop_test)} features from test are going to be dropped")


# In[24]:


train=train.drop(cols_to_drop_train,axis=1)
test=test.drop(cols_to_drop_test,axis=1)


# In[25]:


def clean_inf_nan(df):
    return df.replace([np.inf, -np.inf], np.nan)


# In[26]:


train=clean_inf_nan(train)
test=clean_inf_nan(test)


# In[27]:


train.fillna(0,inplace=True)
test.fillna(0,inplace=True)


# In[28]:


for col in train.columns:
    if train[col].dtypes == "object":
        le=LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))


# In[29]:


print(train.shape)
print(test.shape)


# In[30]:


x_train =train.drop("isFraud",axis=1).copy()
x_test=test.copy()
y_train=train["isFraud"].copy()


# In[31]:


x_train.shape,x_test.shape,y_train.shape


# In[32]:


from sklearn.model_selection import train_test_split
x_train_split,x_test_split,y_train_split,y_test_split=train_test_split(x_train,y_train,test_size=0.3,random_state=7)


# In[33]:


model_name = []
model_score = []


# Decision tree Model

# In[34]:


from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()


# In[35]:


decision_tree.fit(x_train_split,y_train_split)


# In[36]:


print("Roc Auc Score Decision tree:",roc_auc_score(y_test_split,decision_tree.predict(x_test_split)))


# Random forest

# In[37]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(max_depth=45,max_features=30,n_estimators=500,n_jobs=-1,min_samples_leaf=200)


# In[38]:


random_forest.fit(x_train_split,y_train_split)


# In[39]:


print("Roc Auc Score random forest:",roc_auc_score(y_test_split,random_forest.predict(x_test_split)))


# In[ ]:





# Gradient Bossting

# In[46]:


from sklearn.ensemble import GradientBoostingClassifier


# In[47]:


gradient = GradientBoostingClassifier()
gradient.fit(x_train_split,y_train_split)


# In[48]:


print("Roc Auc Score Gradient Bossting:",roc_auc_score(y_test_split,gradient.predict(x_test_split)))


# In[53]:


from sklearn.metrics import confusion_matrix


# In[ ]:





# In[56]:


pd.DataFrame(confusion_matrix(y_test_split, decision_tree.predict(x_test_split)), columns=['Predicted Benign', "Predicted Malignant"], index=['Actual Benign', 'Actual Malignant'])


# In[90]:


from sklearn import metrics
#tn, fp, fn, tp = confusion_matrix(y_test_split,decision_tree.predict(x_test_split)).ravel()
print(f'True Positives: {tp}')
print(f'False Positives: {fp}')
print(f'True Negatives: {tn}')
print(f'False Negatives: {fn}')

confusion_matrix = metrics.confusion_matrix(y_test_split,gradient.predict(x_test_split))
sns.heatmap(confusion_matrix,annot=True,fmt=".0f", cmap="crest")

cm_display.plot()
plt.show()


# In[67]:


probas = decision_tree.predict_proba(x_test_split)[:, 1]


# In[68]:


def get_preds(threshold, probabilities):
    return [1 if prob > threshold else 0 for prob in probabilities]


# In[69]:


roc_values = []
for thresh in np.linspace(0, 1, 100):
    preds = get_preds(thresh, probas)
    tn, fp, fn, tp = confusion_matrix(y_test_split,decision_tree.predict(x_test_split)).ravel()
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    roc_values.append([tpr, fpr])
tpr_values, fpr_values = zip(*roc_values)


# In[70]:


ig, ax = plt.subplots(figsize=(10,7))
ax.plot(fpr_values, tpr_values)
ax.plot(np.linspace(0, 1, 100),
         np.linspace(0, 1, 100),
         label='baseline',
         linestyle='--')
plt.title('Receiver Operating Characteristic Curve', fontsize=18)
plt.ylabel('TPR', fontsize=16)
plt.xlabel('FPR', fontsize=16)
plt.legend(fontsize=12)


# In[71]:


roc_auc_score(y_test_split,decision_tree.predict(x_test_split))


# In[ ]:





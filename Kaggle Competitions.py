#!/usr/bin/env python
# coding: utf-8

# # Churn Rate Prediction

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("/Users/adedayo/Desktop/datasets/Churn_Modelling.csv")

df.head(5)


# In[3]:


df.info()


# In[4]:


df['RowNumber'] = df['RowNumber'].astype(str)

df['CustomerId'] = df['CustomerId'].astype(str)


# In[5]:


df.describe()


# In[6]:


num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

for cols in num_cols:
    plt.figure(figsize = (8,6))
    sns.histplot(df[cols], kde = True)
    plt.title(f'Distiribution of {cols}')
    plt.xlabel(cols)
    plt.ylabel('Frequency')

plt.show()


# In[7]:


sns.boxplot(x = 'Exited', y = 'Age', data = df)


# In[8]:


geo_churn_rate = df['Geography'].value_counts()

sns.barplot(x = geo_churn_rate.index, y = geo_churn_rate.values)

plt.show()


# In[9]:


gender_churn_rate = df['Gender'].value_counts()

sns.barplot(x = gender_churn_rate.index, y = gender_churn_rate.values)

plt.show()


# In[10]:


sns.countplot(df['Exited'])

plt.title('distribution of customer churn')

plt.show()


# In[11]:


plt.figure(figsize = (8,6))

sns.heatmap(df.corr(), annot= True, cmap='coolwarm')

plt.show()


# In[12]:


plt.figure(figsize = (8,8))

sns.scatterplot(x = 'Age', y = 'Balance', data = df)

plt.title('Age vs Balance')

plt.show()


# In[13]:


plt.figure(figsize=(8,8))

sns.scatterplot(x = 'CreditScore', y = 'EstimatedSalary', data = df)

plt.title('Credit Score vs Salary')

plt.show()


# In[14]:


geo_rate = df.groupby('Geography')['Exited'].mean()

plt.figure(figsize=(8,6))

plt.pie(x = geo_rate, labels= geo_rate.index, autopct='%1.1f%%')

plt.title('Geo Churn Rate')

plt.show()


# In[15]:


gender_rate = df.groupby('Gender')['Exited'].mean()

plt.figure(figsize = (8,6))

plt.pie(x = gender_rate, labels = gender_rate.index, autopct='%1.1f%%')

plt.show()


# In[16]:


sns.pairplot(df)


# In[17]:


sns.countplot(x = df['Geography'], hue = df['Exited'])

plt.show()


# In[18]:


sns.countplot(x = df['Gender'], hue = df['Exited'])

plt.show()


# In[19]:


sns.countplot(x = df['HasCrCard'], hue = df['Exited'])

plt.show()


# In[20]:


sns.countplot(x = df['Tenure'], hue = df['Exited'])

plt.show()


# In[21]:


sns.countplot(df['IsActiveMember'], hue = df['Exited'])

plt.show()


# In[22]:


geo = pd.get_dummies(df['Geography'], drop_first=True)

gender = pd.get_dummies(df['Gender'], drop_first=True)

df = pd.concat([df, gender, geo], axis = 1)

df.head(5)


# In[23]:


df.drop(['Geography', 'Gender', 'Surname', 'RowNumber', 'CustomerId'], axis = 1, inplace = True)

df.head(5)


# In[24]:


X = df.drop(['Exited'], axis = 1)

y = df['Exited']


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# ### Logistic Regression

# In[27]:


from sklearn.linear_model import LogisticRegression


# In[28]:


logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)


# In[29]:


pred = logmodel.predict(X_test)

pred


# In[30]:


from sklearn.metrics import classification_report

print(classification_report(y_test, pred))


# In[31]:


results = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
print(results)


# ### Random Forest

# In[32]:


from sklearn.ensemble import RandomForestClassifier


# In[33]:


rfClassifier = RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=4,max_depth=None,
                                     random_state=42)

rfClassifier.fit(X_train,y_train)


# In[34]:


y_pred = rfClassifier.predict(X_test)

y_pred


# In[35]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[36]:


result2 = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})

result2


# ## Personal Loan Prediction

# In[37]:


df = pd.read_csv("/Users/adedayo/Desktop/datasets/Bank_Personal_Loan_Modelling.csv")

df.head()


# In[38]:


df.info()


# In[39]:


df.drop(['ID', 'ZIP_Code'], axis = 1, inplace = True)

df.head(2)


# In[40]:


sns.countplot(x = df['Personal_Loan'], hue = df['Education'])

plt.show()


# In[41]:


sns.countplot(x = df['Personal_Loan'], hue = df['CD_Account'])

plt.show()


# In[42]:


sns.countplot(x = df['Personal_Loan'], hue = df['CreditCard'])

plt.show()


# In[43]:


sns.countplot(x = df['Personal_Loan'], hue = df['Online'])

plt.show()


# In[44]:


sns.countplot(x = df['Personal_Loan'], hue = df['Family'])

plt.show()


# In[45]:


sns.countplot(x = df['Personal_Loan'], hue = df['Securities_Account'])

plt.show()


# In[46]:


num_cols = ['Age', 'Experience', 'Income']

for cols in num_cols:
    plt.figure(figsize=(8,6))
    sns.histplot(df[cols], kde = True)
    plt.title(f'Distribution of {cols}')
    plt.xlabel(cols)
    plt.ylabel('Frequency')
plt.show()        


# In[47]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.show()


# In[48]:


sns.pairplot(df)


# In[49]:


X = df.drop(['Personal_Loan'], axis = 1)

y = df['Personal_Loan']


# In[50]:


from sklearn.model_selection import train_test_split


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# #### Logistic Regression

# In[52]:


from sklearn.linear_model import LogisticRegression


# In[53]:


logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)


# In[54]:


pred = logmodel.predict(X_test)

pred


# In[55]:


from sklearn.metrics import classification_report

print(classification_report(y_test,pred))


# In[56]:


result = pd.DataFrame({'Actual':y_test, 'Predicted':pred})

result


# ### Random Forest

# In[57]:


from sklearn.ensemble import RandomForestClassifier


# In[58]:


rfClassifier = RandomForestClassifier(n_estimators = 100, min_samples_split=2, min_samples_leaf=4, random_state=42)

rfClassifier.fit(X_train,y_train)


# In[59]:


pred = rfClassifier.predict(X_test)

pred


# In[60]:


result = pd.DataFrame({'Actual':y_test,'Predicted':pred})

result


# In[61]:


from sklearn.metrics import classification_report

print(classification_report(y_test,pred))


# ## Home Loan Prediction

# In[62]:


df = pd.read_csv("/Users/adedayo/Desktop/datasets/train.csv")

df.head(5)


# In[63]:


df.info()


# In[64]:


df.isnull().sum()


# In[65]:


df.dropna(inplace = True)


# In[66]:


df.info()


# In[67]:


df.describe()


# In[68]:


num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

for cols in num_cols:
    plt.figure(figsize = (8,7))
    sns.histplot(df[cols], kde = True)
    plt.title(f'distribution of {cols}')
    plt.xlabel(cols)
    plt.ylabel('Frequency')
    
plt.show()


# In[69]:


df.drop(['Loan_ID'], axis = 1, inplace = True)

gender = pd.get_dummies(df['Gender'], drop_first= True)

married = pd.get_dummies(df['Married'], prefix = 'married', drop_first= True)

dependents = pd.get_dummies(df['Dependents'], prefix = 'dependents', drop_first= True)

edu = pd.get_dummies(df['Education'], drop_first= True)

self_employed = pd.get_dummies(df['Self_Employed'], prefix = 'self_employed', drop_first= True)

property_area = pd.get_dummies(df['Property_Area'], drop_first=True)


df = pd.concat([df, gender, married, dependents, edu, self_employed, property_area], axis = 1)

df.drop(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'], axis = 1, inplace=True)


df.head(5)


# In[70]:


df['Loan_Status'] = df['Loan_Status'].map({'Y':1, 'N':0})

df['Loan_Status'].value_counts()


# In[71]:


plt.figure(figsize=(10,8))

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.show()


# In[72]:


df.head(3)


# In[73]:


X = df.drop(['Loan_Status'], axis = 1)

y = df['Loan_Status']


# In[74]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# In[75]:


from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)


# In[76]:


prediction = logmodel.predict(X_test)

prediction


# In[77]:


test = pd.read_csv("/Users/adedayo/Desktop/datasets/test.csv")

test.head(5)


# In[78]:


test.drop(['Loan_ID'], axis = 1, inplace = True)

gender = pd.get_dummies(test['Gender'], drop_first= True)

married = pd.get_dummies(test['Married'], prefix = 'married', drop_first= True)

dependents = pd.get_dummies(test['Dependents'], prefix = 'dependents', drop_first= True)

edu = pd.get_dummies(test['Education'], drop_first= True)

self_employed = pd.get_dummies(test['Self_Employed'], prefix = 'self_employed', drop_first= True)

property_area = pd.get_dummies(test['Property_Area'], drop_first=True)




test = pd.concat([test, gender, married, dependents, edu, self_employed, property_area], axis = 1)

test.drop(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'], axis = 1, inplace=True)


test.head(5)


# In[79]:


test.dropna(inplace = True)

test.head()


# In[80]:


pred = logmodel.predict(test)

pred


# In[81]:


from sklearn.metrics import classification_report

print(classification_report(y_test,prediction))


# In[82]:


test


# In[83]:


result = pd.DataFrame({'actual':y_test,'predictions':prediction})

result


# In[84]:


from sklearn.ensemble import RandomForestClassifier


# In[85]:


rfClassifier = RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=4, random_state=42)


# In[86]:


rfClassifier.fit(X_train,y_train)


# In[87]:


randompred = rfClassifier.predict(X_test)

randompred


# In[88]:


from sklearn.metrics import classification_report

print(classification_report(y_test,randompred))


# In[89]:


from sklearn.tree import DecisionTreeClassifier


# In[90]:


treeclass = DecisionTreeClassifier(random_state=42)

treeclass.fit(X_train,y_train)


# In[91]:


treepred = treeclass.predict(X_test)

treepred


# In[92]:


from sklearn.metrics import classification_report

print(classification_report(y_test,treepred))


# In[93]:


treeclass.predict(test)


# In[94]:


from xgboost import XGBClassifier


# In[95]:


xgmodel = XGBClassifier(n_estimator = 100, max_depth = 6)

xgmodel.fit(X_train,y_train)


# In[96]:


xgmodelpred = xgmodel.predict(X_test)

xgmodelpred


# In[97]:


from sklearn.metrics import classification_report

print(classification_report(y_test,xgmodelpred))


# In[98]:


loan_status = xgmodel.predict(test)

Loan_status = pd.DataFrame({'Loan_status':loan_status})

Loan_status


# In[99]:


test = pd.concat([test,Loan_status], axis =1)

test.head(5)


# In[100]:


test['Loan_status'].value_counts()


# In[101]:


importance = rfClassifier.feature_importances_


# In[102]:


importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})
importance_df = importance_df.sort_values('Importance', ascending=False)

importance_df


# In[103]:


df = pd.read_csv("/Users/adedayo/Desktop/datasets/Default_Fin.csv")

df.head(5)


# In[104]:


df.info()


# In[105]:


df.drop(['Index'], axis = 1, inplace = True)

df.head(5)


# In[106]:


df['Defaulted?'].value_counts()


# In[107]:


df.describe()


# In[108]:


num_cols = ['Employed', 'Bank Balance', 'Annual Salary', 'Defaulted?']

for cols in num_cols:
    plt.figure(figsize = (8,8))
    sns.histplot(df[cols], kde = True)
    plt.title(f'Distribution of {cols}')
    plt.xlabel(cols)
    plt.ylabel('Frequency')
    
plt.show()


# In[109]:


sns.heatmap(df.corr(), annot=True, cmap='coolwarm')


# In[110]:


X = df.drop(['Defaulted?'], axis = 1)

y = df['Defaulted?']


# In[111]:


from sklearn.model_selection import train_test_split


# In[112]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[113]:


from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)


# In[114]:


logprediction = logmodel.predict(X_test)

logprediction


# In[115]:


print(classification_report(y_test,logprediction))


# In[116]:


result = pd.DataFrame({'Actual':y_test, 'Prediction':logprediction})

result


# In[117]:


from sklearn.ensemble import RandomForestRegressor


# In[118]:


rfClassifier = RandomForestClassifier(n_estimators = 100, min_samples_split=2, min_samples_leaf=4, random_state=42)


# In[119]:


rfClassifier.fit(X_train,y_train)


# In[120]:


rfpred = rfClassifier.predict(X_test)

rfpred


# In[121]:


print(classification_report(y_test,rfpred))


# In[122]:


from sklearn.tree import DecisionTreeClassifier


# In[123]:


dtree = DecisionTreeClassifier(random_state=42)


# In[124]:


dtree.fit(X_train,y_train)


# In[125]:


treepred = dtree.predict(X_test)

treepred


# In[126]:


print(classification_report(y_test,treepred))


# In[127]:


from xgboost import XGBClassifier


# In[128]:


xgmodel = XGBClassifier(n_estimators = 100, max_depth = 6)


# In[129]:


xgmodel.fit(X_train,y_train)


# In[130]:


xgpred = xgmodel.predict(X_test)

xgpred


# In[131]:


print(classification_report(y_test,xgpred))


# In[132]:


importance = rfClassifier.feature_importances_


# In[133]:


importance_df = pd.DataFrame({'features':X_train.columns, 'importance':importance})

importance_df


# In[134]:


train = pd.read_csv("/Users/adedayo/Desktop/datasets/Training Data.csv")

train.head(5)


# In[135]:


train.info()


# In[136]:


train.drop(['Profession', 'CITY', 'STATE'], axis = 1, inplace = True)


# In[137]:


train.head(5)


# In[138]:


house_ownership = pd.get_dummies(train['House_Ownership'], drop_first = True)

marital_status = pd.get_dummies(train['Married/Single'], drop_first = True)

own_car = pd.get_dummies(train['Car_Ownership'], prefix = 'car', drop_first = True)

train = pd.concat([train,house_ownership,marital_status,own_car],axis =1)

train.drop(['House_Ownership','Married/Single','Car_Ownership', 'Id'], axis = 1, inplace = True)


train.head(3)


# In[139]:


X = train.drop(['Risk_Flag'], axis = 1)

y = train['Risk_Flag']


# In[140]:


from sklearn.model_selection import train_test_split


# In[141]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# In[142]:


from sklearn.linear_model import LogisticRegression


# In[143]:


logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)


# In[144]:


logpredict = logmodel.predict(X_test)

logpredict


# In[145]:


from sklearn.metrics import classification_report

print(classification_report(y_test,logpredict))


# In[146]:


from sklearn.ensemble import RandomForestClassifier


# In[147]:


rfclassifier = RandomForestClassifier(n_estimators = 100, min_samples_split=2, min_samples_leaf=4, random_state = 42)

rfclassifier.fit(X_train,y_train)


# In[148]:


rfpred = rfclassifier.predict(X_test)

rfpred


# In[149]:


from sklearn.metrics import classification_report

print(classification_report(y_test,rfpred))


# In[150]:


from sklearn.tree import DecisionTreeClassifier


# In[151]:


treemodel = DecisionTreeClassifier(random_state=42)

treemodel.fit(X_train,y_train)


# In[152]:


treepred = treemodel.predict(X_test)

treepred


# In[153]:


from sklearn.metrics import classification_report

print(classification_report(y_test,treepred))


# In[154]:


from xgboost import XGBClassifier


# In[155]:


xgmodel = XGBClassifier(n_estimators = 100, max_depth = 4)

xgmodel.fit(X_train,y_train)


# In[156]:


xgpred = xgmodel.predict(X_test)

xgpred


# In[157]:


from sklearn.metrics import classification_report

print(classification_report(y_test,xgpred))


# In[158]:


importance = rfclassifier.feature_importances_


# In[159]:


importance_df = pd.DataFrame({'Feature':X_train.columns, 'Importance':importance})

importance_df


# In[160]:


test = pd.read_csv("/Users/adedayo/Desktop/datasets/Test Data.csv")

test.head(5)


# In[161]:


house_ownership = pd.get_dummies(test['House_Ownership'], drop_first = True)

marital_status = pd.get_dummies(test['Married/Single'], drop_first = True)

own_car = pd.get_dummies(test['Car_Ownership'], prefix = 'car', drop_first = True)

test = pd.concat([test,house_ownership,marital_status,own_car],axis =1)

test.drop(['House_Ownership','Married/Single','Car_Ownership', 'ID', 'Profession',
          'CITY', 'STATE'], axis = 1, inplace = True)


test.head(3)


# In[162]:


risk_flag = rfclassifier.predict(test)

risk_flag = pd.DataFrame({'risk_flag':risk_flag})

risk_flag


# In[163]:


test = pd.concat([test,risk_flag], axis = 1)

test


# In[164]:


test['risk_flag'].value_counts()


# In[165]:


df = pd.read_csv("/Users/adedayo/Downloads/archive (9)/training_set.csv")

df.head(5)


# In[166]:


df.info()


# In[167]:


df.dropna(inplace=True)

df.info()


# In[168]:


df.drop(['Loan_ID'], axis = 1, inplace = True)


# In[169]:


gender = pd.get_dummies(df['Gender'], drop_first=True)

married = pd.get_dummies(df['Married'], prefix = 'married', drop_first=True)

dependents = pd.get_dummies(df['Dependents'], prefix = 'dependents', drop_first = True)

edu = pd.get_dummies(df['Education'], drop_first=True)

self_employed = pd.get_dummies(df['Self_Employed'], prefix = 'self_employed', drop_first=True)

property_area = pd.get_dummies(df['property_Area'], drop_first=True)


df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})


df.drop(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'property_Area'], axis = 1, inplace = True)

df = pd.concat([df,gender,married,dependents,edu,self_employed, property_area], axis = 1)


df.head(5)


# In[170]:


df.info()


# In[171]:


X = df.drop(['Loan_Status'], axis = 1)

y = df['Loan_Status']


# In[172]:


from sklearn.model_selection import train_test_split


# In[173]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[174]:


from sklearn.ensemble import RandomForestClassifier


# In[175]:


rfclassifier = RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=4, random_state=42)

rfclassifier.fit(X_train,y_train)


# In[176]:


rfpred = rfclassifier.predict(X_test)

rfpred


# In[177]:


from sklearn.metrics import classification_report

print(classification_report(y_test,rfpred))


# In[178]:


from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)


# In[179]:


logpred = logmodel.predict(X_test)


logpred


# In[180]:


print(classification_report(y_test,logpred))


# In[181]:


from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(random_state=42)

dtree.fit(X_train,y_train)


# In[182]:


dpred = dtree.predict(X_test)

dpred


# In[183]:


print(classification_report(y_test,dpred))


# In[184]:


from xgboost import XGBClassifier

xgmodel = XGBClassifier(n_estimators = 50, max_depth = 6)

xgmodel.fit(X_train,y_train)


# In[185]:


xgpred = xgmodel.predict(X_test)

xgpred


# In[186]:


print(classification_report(y_test,xgpred))


# In[187]:


test = pd.read_csv("/Users/adedayo/Downloads/archive (9)/testing_set.csv")

test.head()


# In[188]:


gender = pd.get_dummies(test['Gender'], drop_first=True)

married = pd.get_dummies(test['Married'], prefix = 'married', drop_first=True)

dependents = pd.get_dummies(test['Dependents'], prefix = 'dependents', drop_first = True)

edu = pd.get_dummies(test['Education'], drop_first=True)

self_employed = pd.get_dummies(test['Self_Employed'], prefix = 'self_employed', drop_first=True)

property_area = pd.get_dummies(test['property_Area'], drop_first=True)


test.dropna(inplace = True)

loan_id = test['Loan_ID']

loan_id = pd.DataFrame({'loan_id':loan_id})


test.drop(['Gender', 'Loan_ID', 'Married', 'Dependents', 'Education', 'Self_Employed', 'property_Area'], axis = 1, inplace = True)

test = pd.concat([test,gender,married,dependents,edu,self_employed, property_area], axis = 1)


test.head(5)


# In[189]:


test.dropna(inplace = True)

test.info()


# In[190]:


logpred = logmodel.predict(test)

logpred


# In[191]:


loan_status = pd.DataFrame({'loan_status':logpred})
loan_status = loan_status.reset_index(drop=True)

loan_status


# In[192]:


loan_id = loan_id.reset_index(drop=True)

loan_id


# In[193]:



result = pd.concat([loan_id, loan_status], axis=1)

result


# In[194]:


df = pd.read_csv("/Users/adedayo/Downloads/advertising.csv")

df.head()


# In[195]:


df.info()


# In[196]:


num_cols = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage',
           'Male']

for cols in num_cols:
    plt.figure(figsize = (8,6))
    sns.histplot(df[cols], kde = True)
    plt.title(f'distribution of {cols}')
    plt.xlabel(cols)
    plt.ylabel('Frequency')

    plt.show()


# In[197]:


plt.figure(figsize=(8,8))

sns.countplot(df['Male'], hue = df['Clicked on Ad'])


# In[198]:


sns.scatterplot(x = df['Age'], y = df['Area Income'])


# In[199]:


gender_rate = df.groupby('Male')['Clicked on Ad'].mean()

plt.pie(x = gender_rate, labels = gender_rate.index, autopct = '%1.1f%%')


# In[200]:


sns.pairplot(df, hue = 'Clicked on Ad')


# In[201]:


plt.figure(figsize = (8,6))

sns.heatmap(df.corr(), annot=True, cmap = 'coolwarm')


# In[202]:


plt.figure(figsize = (8,6))

sns.scatterplot(x = df['Daily Time Spent on Site'], y = df['Daily Internet Usage'])


# In[203]:


df.info()


# In[204]:


df.drop(['City', 'Country', 'Ad Topic Line', 'Timestamp'], axis = 1, inplace = True)


# In[205]:


X = df.drop(['Clicked on Ad'], axis = 1)

y = df['Clicked on Ad']


# In[206]:


from sklearn.model_selection import train_test_split


# In[207]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[208]:


from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)


# In[209]:


logpred = logmodel.predict(X_test)

logpred


# In[210]:


print(classification_report(y_test,logpred))


# In[211]:


df = pd.read_csv("/Users/adedayo/Downloads/USA_Housing.csv")

df.head()


# In[212]:


df.info()


# In[213]:


df.drop(['Address'], axis = 1, inplace = True)


# In[214]:


df.head(3)


# In[215]:


num_cols = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
           'Area Population', 'Price']


for cols in num_cols:
    plt.figure(figsize = (8,6))
    sns.histplot(df[cols], kde = True)
    plt.title(f'distribution of {cols}')
    plt.xlabel(cols)
    plt.ylabel('Frequency')

plt.show()


# In[216]:


plt.scatter(x = df['Avg. Area Income'], y = df['Area Population'])


# In[217]:


plt.scatter(x = df['Avg. Area House Age'], y = df['Area Population'])


# In[218]:


plt.scatter(x = df['Avg. Area Number of Bedrooms'], y = df['Area Population'])


# In[219]:


plt.figure(figsize = (8,6))

sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm')


# In[220]:


sns.pairplot(df, hue = 'Price')


# In[221]:


df


# In[222]:


X = df.drop(['Price'], axis = 1)

y = df['Price']


# In[223]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[224]:


from sklearn.linear_model import LinearRegression

linmodel = LinearRegression()

linmodel.fit(X_train, y_train)


# In[225]:


linpred = linmodel.predict(X_test)

linpred


# In[226]:


linmodel.intercept_


# In[227]:


coef_df = pd.DataFrame(linmodel.coef_,X.columns, columns = ['Coefficient'])

coef_df


# In[228]:


plt.scatter(y_test,linpred)


# In[229]:


sns.displot((y_test-linpred))


# In[230]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test,linpred))
print('MSE:', metrics.mean_squared_error(y_test, linpred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, linpred)))


# In[ ]:





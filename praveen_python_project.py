#!/usr/bin/env python
# coding: utf-8

# In[152]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("housing.csv")


# In[ ]:





# In[127]:


data


# In[110]:


data.dropna


# In[111]:


data.info


# In[153]:


data.info()


# In[112]:


data.dropna()


# In[113]:


data.info()


# In[154]:


data.dropna(inplace=True)


# In[13]:


data.info()


# In[155]:


from sklearn.model_selection import train_test_split
x = data.drop(['median_house_value'],axis=1)
y = data['median_house_value']


# In[15]:


x


# In[16]:


y


# In[156]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[157]:


train_data = x_train.join(y_train)


# In[158]:


train_data


# In[20]:





# In[ ]:





# In[ ]:





# In[33]:


train_data


# In[34]:


a = data


# In[35]:


a


# In[36]:


data.info()


# In[37]:


a = data()


# In[38]:


a = data()


# In[39]:


a = data.info()


# In[40]:


a


# In[ ]:





# In[41]:


a


# In[43]:


a = data


# In[44]:


a


# In[45]:


train_data


# In[159]:


pd.get_dummies(train_data.ocean_proximity)


# In[160]:


train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity))


# In[120]:


train_data


# In[161]:


train_data.drop(['ocean_proximity'],axis=1)


# In[ ]:





# In[92]:


sns.heatmap(train_data.corr(),annot=True,cmap="GnBuYl")


# In[93]:


sns.heatmap(train_data.corr(),annot=True,cmap="YlGnBu")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[76]:


train_data = train_data.drop(['INLAND'],axis=1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[82]:


from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor()
regression.fit(x_train,y_train)


# In[121]:


test_data


# In[ ]:





# In[122]:


train_data


# In[123]:


from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor()
x_train = train_data.drop(["median_house_value"],axis=1)
y_train = train_data["median_house_value"]
regression.fit(x_train,y_train)


# In[100]:


import re
Output = re.findall(r'\d+\.\d+', train_data["INLAND"])


# In[124]:


train_data.drop(['ocean_proximity'],axis=1)


# In[137]:


from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor()
x_train = train_data.drop(["median_house_value"],axis=1)
y_train = train_data["median_house_value"]
regression.fit(x_train,y_train)


# In[138]:


x_train


# In[139]:


train_data


# In[140]:


from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor()
x_train = train_data.drop(["median_house_value"],axis=1)
y_train = train_data["median_house_value"]
regression.fit(x_train,y_train)


# In[141]:


train_data = train_data.drop(["ocean_proximity"],axis=1)


# In[142]:





# In[143]:


from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor()
x_train = train_data.drop(["median_house_value"],axis=1)
y_train = train_data["median_house_value"]
regression.fit(x_train,y_train)


# In[146]:


forest.score(x_test,y_test)


# In[147]:


train_data = train_data.drop(["ocean_proximity"],axis=1)


# In[148]:


train_data


# In[149]:


from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor()
x_train = train_data.drop(["median_house_value"],axis=1)
y_train = train_data["median_house_value"]
regression.fit(x_train,y_train)


# In[150]:


forest.score(x_test,y_test)


# In[151]:


from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor()
x_train = train_data.drop(["median_house_value"],axis=1)
y_train = train_data["median_house_value"]
regression.fit(x_train,y_train)


# In[162]:


from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor()
x_train = train_data.drop(["median_house_value"],axis=1)
y_train = train_data["median_house_value"]
regression.fit(x_train,y_train)


# In[163]:


train_data


# In[185]:


train_data = train_data.drop(['r_room'],axis=1)


# In[186]:


train_data


# In[167]:


train_data


# In[168]:


from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor()
x_train = train_data.drop(["median_house_value"],axis=1)
y_train = train_data["median_house_value"]
regression.fit(x_train,y_train)


# In[169]:


forest.score(x_test,y_test)


# In[170]:


forest.score(x_train,y_train)


# In[172]:


test_data = x_test.join(y_test)


# In[173]:


pd.get_dummies(test_data.ocean_proximity)


# In[174]:


test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity))


# In[177]:


test_data = test_data.drop(['ocean_proximity'],axis=1)


# In[191]:


from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor()
x_train = train_data.drop(["median_house_value"],axis=1)
y_train = train_data["median_house_value"]
regression.fit(x_train,y_train)


# In[179]:


x_test = test_data.drop(["median_house_value"],axis=1)
y_test = test_data["median_house_value"]


# In[190]:


forest.score(x_test,y_test)


# In[ ]:





# In[ ]:





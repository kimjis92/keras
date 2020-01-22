#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import linear_model

reg=linear_model.LinearRegression()
x_train=([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y_train=([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
reg.fit(x_train, y_train)

y_pred=reg.predict(([[11],[12],[13]]))


# In[2]:


y_pred


# In[ ]:





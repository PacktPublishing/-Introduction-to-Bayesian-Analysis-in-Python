#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pymc3 as pm
import numpy as np
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


size=100
true_intercept=1
true_slope=2
x=np.linspace(0,2,size)
true_regression_line=true_intercept+true_slope*x
y=true_regression_line+np.random.normal(scale=0.5,size=size)
y[x<0.25]*=2.5
data=dict(x=x,y=y)


# In[11]:


fig,ax=plt.subplots(figsize=(9,6))
ax.scatter(data['x'],data['y'],marker='x',label='sampled data')
ax.plot(x,true_regression_line,color='red',label='true regression line')
ax.set(xlabel='x',ylabel='y',title='Generated data and underlying model')
plt.legend()


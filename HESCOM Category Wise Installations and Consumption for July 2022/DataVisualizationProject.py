#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
 
# creating a data frame
df = pd.read_csv("HESCOM_Category_wise_installations_and_Consumption_for_June_2022.csv")


# In[44]:


df1=df[['Low Tension Installations (in Numbers)','Low Tension Consumption (in Million Units)']]
# Draw a scatter plot
df1.plot.scatter(x ='Low Tension Installations (in Numbers)', y ='Low Tension Consumption (in Million Units)',rot=90,title='Scatter Plot')


# In[47]:


df2=df[['High Tension  Installations (in Numbers)','High Tension Consumption (in Million Units)']]
df2.boxplot(by ='High Tension  Installations (in Numbers)', column =['High Tension Consumption (in Million Units)'],figsize=(10,5), grid = False)


# In[48]:


df3=df[['Category','Low Tension Installations (in Numbers)']]
# using a function df.plot.bar()
df3.plot.bar(x ='Category', y ='Low Tension Installations (in Numbers)',title='Bar Plot')


# In[ ]:





# In[ ]:





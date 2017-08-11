
# coding: utf-8

# In[59]:


import numpy as np
import pandas as pd


# reading the datasets

# In[221]:


bike=pd.read_csv("bike.csv")

cab=pd.read_csv("yellow_raw.csv")


# In[62]:


bike.head()


# In[63]:


bike.shape


# In[64]:


cab.head()


# In[65]:


cab.shape


# slicing 500k rows from bike and cabs data

# In[66]:


bike_citi=bike[:500000]


# In[67]:


pd.set_option('display.max_columns', 500)


# In[75]:


bike_citi.head()


# we will try to project the lat and long of pickup and drop off of cabs and bikes. so we drop the rest of the features.

# In[113]:


bike_citi.drop(['duration'],axis = 1, inplace = True)


# In[114]:


bike_citi.head()


# In[86]:


bike_citi.drop_bdate = pd.to_numeric(bike_citi['drop_bdate'], errors='coerse')


# In[96]:


bike_citi.dtypes


# In[100]:


cab_y=cab[:500000]


# In[101]:


cab_y.rename(columns={'pick_long': 'pick_blong','pick_lat': 'pick_blat','drop_long': 'drop_blong','drop_lat': 'drop_blat','pick_time': 'pick_btime','drop_time': 'drop_btime','drop_date': 'drop_bdate','pick_date': 'pick_bdate'}, inplace=True)


# In[102]:


cab_y.head()


# In[112]:


cab_y.drop(['drop_bdate'],axis = 1, inplace = True)


# In[116]:


cab_y.head()
cab_y.dtypes


# we concatenate both bike and cab datasets to apply the SOM algorithm

# In[117]:


data = pd.concat([bike_citi,cab_y])


# In[118]:


data.head()


# after combining we have now 1 million rows

# In[202]:


data.shape


# id column is created to reference any row from the data 

# In[120]:


data.insert(0, 'id', range(0, 0 + len(data)))
data


# In[121]:


data.dtypes


# extracting the features of the data 

# In[122]:


X = data.iloc[:].values


# scaling the features using sklearn

# In[123]:


from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range = (0,1))


# In[124]:


X = sc.fit_transform(X)


# implementation of som using minisom class 

# In[125]:


from minisom import MiniSom


# initialising the weights randomly close to 0 
# 
# the hyper parameters being sigma and learning rate changing them to 0.5 and 0.3 and running the model to 100 iterations

# In[176]:


som = MiniSom(x = 5, y = 5, input_len = 5, sigma = 0.5, learning_rate = 0.3)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)


# In[177]:


from pylab import bone, pcolor, colorbar, plot, show


# In[178]:


get_ipython().magic('matplotlib inline')


# visualisng the SOM mappings using pylab

# In[179]:


bone()
pcolor(som.distance_map().T)
colorbar()


# creating the mappings for each row**

# In[180]:


mappings = som.win_map(X)


# In[181]:


len(mappings)


# extracting the clusters based on the similar inter neuron distances.

# In[182]:


cluster1 = np.concatenate((mappings[(0,1)],mappings[(0,0)],mappings[(4,4)]), axis = 0)
cluster1 = sc.inverse_transform(cluster1)


# In[188]:


cluster8 = np.concatenate((mappings[(0,2)],mappings[(4,0)],mappings[(4,2)],mappings[(0,3)]), axis = 0)
cluster8 = sc.inverse_transform(cluster8)


# In[184]:


cluster3 = np.concatenate((mappings[(0,4)],mappings[(4,1)],mappings[(4,3)]), axis = 0)
cluster3 = sc.inverse_transform(cluster3)


# In[185]:


cluster4 = np.concatenate((mappings[(1,3)],mappings[(2,2)]), axis = 0)
cluster4 = sc.inverse_transform(cluster4)


# In[187]:


cluster7 = np.concatenate((mappings[(2,1)],mappings[(2,3)]), axis = 0)
cluster7 = sc.inverse_transform(cluster7)


# In[189]:


cluster6 = np.concatenate((mappings[(1,1)],mappings[(1,2)],mappings[(3,1)],mappings[(3,3)]), axis = 0)
cluster6 = sc.inverse_transform(cluster6)


# In[190]:


cluster9 = np.concatenate((mappings[(1,0)],mappings[(2,0)],mappings[(1,4)],mappings[(2,4)]), axis = 0)
cluster9 = sc.inverse_transform(cluster9)


# In[191]:


cluster10 = np.concatenate((mappings[(3,0)],mappings[(0,4)],mappings[(3,4)]), axis = 0)
cluster10 = sc.inverse_transform(cluster10)


# In[199]:


c10=pd.DataFrame(cluster10)
c10.head()


# In[201]:


len(c1+c3+c8+c7+c6+c9+c10+c4)


# In[211]:


c4.rename(columns={0: 'id'}, inplace=True)
c4.head()


# In[220]:


c4.head()


# In[223]:


bike1=bike[:500000]


# In[243]:


cab1=cab[:500000]


# In[260]:


bike1.head()


# In[245]:


cab1.rename(columns={'pick_long': 'pick_blong','pick_lat': 'pick_blat','drop_long': 'drop_blong','drop_lat': 'drop_blat','pick_time': 'pick_btime','drop_time': 'drop_btime','drop_date': 'drop_bdate','pick_date': 'pick_bdate'}, inplace=True)


# In[259]:


cab1.head()


# In[251]:


bike1.drop(['duration'],axis = 1, inplace = True)


# In[255]:


cab1['type']=1


# In[256]:


bike1['type']=0


# In[257]:


bike1['fare']=0
bike1['trip_dist']=0
bike1['passenger_count']=0
bike1['VendorID']=0


# In[258]:



cab1['duration']=0
cab1['pick_sid']=999999
cab1['drop_sid']=111111
cab1['user_type']='cabrider'


# In[261]:


data2 = pd.concat([bike1,cab1])


# In[262]:


data2.shape


# In[295]:


pd.set_option('display.max_rows', 10)


# merging the cluster dataframes with the original data plugging back the features of each row in the clusters 
# to understand and analyze the properties.

# In[309]:


m4 = pd.merge(left=data2,right=c4, left_on='id', right_on= 'id')


# In[265]:


m1.shape


# In[302]:


m1.head()


# In[290]:


m1['trip']=m1['pick_sname']+m1['drop_sname']


# In[294]:


m1['user_type'].value_counts()


# In[293]:


m1['type'].value_counts()


# In[297]:


m1['trip_dist'].value_counts()


# In[298]:


m1['pick_sname'].value_counts()


# In[299]:


m1['passenger_count'].value_counts()


# In[300]:


m1['fare'].value_counts()


# In[277]:


get_ipython().magic('matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
m1['VendorID'].value_counts().plot(kind='bar')


# In[319]:


get_ipython().magic('matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
m3['VendorID'].value_counts().plot(kind='bar')


# In[301]:


m1['drop_sname'].value_counts()


# In[316]:


m3['trip']=m3['pick_sname']+m3['drop_sname']
m3['passenger_count'].value_counts()


# In[311]:


m3['user_type'].value_counts()


# In[312]:


m3['type'].value_counts()


# In[313]:



m3['trip_dist'].value_counts()


# In[314]:



m3['fare'].value_counts()


# In[315]:



m3['pick_sname'].value_counts()


# In[317]:


m3['trip'].value_counts()


# In[318]:



m3['trip_dist'].value_counts()


# In[320]:


m8['trip']=m8['pick_sname']+m8['drop_sname']


# In[321]:


m8['user_type'].value_counts()


# In[322]:


m8['trip_dist'].value_counts()


# In[323]:


m8['fare'].value_counts()


# In[324]:


m8['pick_sname'].value_counts()


# In[325]:


m8['passenger_count'].value_counts()


# In[326]:


m8['drop_sname'].value_counts()


# In[327]:


m8['trip'].value_counts()


# In[329]:



m7['trip']=m7['pick_sname']+m7['drop_sname']


# In[330]:


m7['user_type'].value_counts()


# In[331]:


m7['trip_dist'].value_counts()


# In[332]:


m7['fare'].value_counts()


# In[333]:


m7['pick_sname'].value_counts()


# In[334]:


m7['passenger_count'].value_counts()


# In[335]:


m7['trip'].value_counts()


# In[337]:


m6['trip']=m6['pick_sname']+m6['drop_sname']


# In[338]:


m6['user_type'].value_counts()


# In[339]:


m6['trip_dist'].value_counts()


# In[340]:



m6['fare'].value_counts()


# In[341]:



m6['pick_sname'].value_counts()


# In[342]:


m6['trip'].value_counts()


# In[343]:



m9['trip']=m9['pick_sname']+m9['drop_sname']


# In[344]:


m9['user_type'].value_counts()


# In[345]:


m9['trip_dist'].value_counts()


# In[348]:



get_ipython().magic('matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
m9['VendorID'].value_counts().plot(kind='bar')


# In[346]:



m9['pick_sname'].value_counts()


# In[347]:


m9['trip'].value_counts()


# In[349]:


m10['trip']=m10['pick_sname']+m10['drop_sname']


# In[350]:


m10['user_type'].value_counts()


# In[351]:



get_ipython().magic('matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
m10['VendorID'].value_counts().plot(kind='bar')


# In[352]:



m4['trip']=m4['pick_sname']+m4['drop_sname']


# In[353]:


m4['user_type'].value_counts()


# In[354]:



m4['pick_sname'].value_counts()


# In[355]:



m4['trip'].value_counts()


# conclusion: 
#     
#     cluster properties are as follows:
# 
# c1
# 
# cab riders
# use cabs
# trip dist below 2
# most pick ups penn stat w42 st dyer ave
# no of passengers 1 most
# vendors id 2
# drop sname e 47 st park ave
# 
# 
# c3
# 
# passenger counts are mostly 1
# 
# most are cab riders
# 
# mostlycab
# 
# fare for the rides are 5.5 to 7
# 
# pick up stations are west st chambers st and carmine st & 6 ave
# 
# trip are from central park s and 6 ave central park, central park to ave5 ave & e 78st
# 
# trip dist are mostsly 0 to 1.10 miles
# 
# c8
# 
# more subsribers
# 
# dist are 0.7 to 1
# 
# fare are 5 to 6.5
# 
# pick ups at perishing square north
# 
# trips are central park s and 6 ave central park s & 6 ave
# 
# c7
# 
# 
# trips are perishing sq ne24 st & park ave s
# 
# perishing sq north most pick ups
# 
# c6
# 
# trips are Central Park S & 6 AveCentral Park S & 6 Ave  
# 
# pick up points are    West St & Chambers St 
# 
# users are subsribers
# 
# 
# c9
# 
# trips are Central Park S & 6 AveCentral Park S & 6 Ave 
# 
# pick ups are Pershing Square North    
# 
# good share of subsribers and cab riders
# 
# c10
# 
# only cab riders
# 
# c4
# 
# mostly subsribers
# 
# pershing sq north
# 
# central park s & 6 ave central

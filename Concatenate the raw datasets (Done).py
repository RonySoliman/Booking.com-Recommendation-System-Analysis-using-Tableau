#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# ## Libraries importing

# In[1]:


# Import the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from datetime import date


# ## Data importing

# In[2]:


#

city_A = pd.read_csv("Case_Study_Urgency_Message_Data.xlsx - City_A.csv")
city_B = pd.read_csv("Case_Study_Urgency_Message_Data.xlsx - City_B.csv")
city_C = pd.read_csv("Case_Study_Urgency_Message_Data.xlsx - City_C.csv")
city_D = pd.read_csv("Case_Study_Urgency_Message_Data.xlsx - City_D.csv")
city_E = pd.read_csv("Case_Study_Urgency_Message_Data.xlsx - City_E.csv")


# ## Make sure that all the columns have the same variables names.

# In[3]:


city_A.columns = ["ADR_USD", "hotel_id", "city_id", "star_rating", "accommadation_type_name", "chain_hotel", "booking_date",
                 "checkin_date", "checkout_date"]


# In[4]:


city_B.columns = ["ADR_USD", "hotel_id", "city_id", "star_rating", "accommadation_type_name", "chain_hotel", "booking_date",
                 "checkin_date", "checkout_date"]


# In[5]:


city_C.columns = ["ADR_USD", "hotel_id", "city_id", "star_rating", "accommadation_type_name", "chain_hotel", "booking_date",
                 "checkin_date", "checkout_date"]


# In[6]:


city_D.columns = ["ADR_USD", "hotel_id", "city_id", "star_rating", "accommadation_type_name", "chain_hotel", "booking_date",
                 "checkin_date", "checkout_date"]


# In[7]:


city_E.columns = ["ADR_USD", "hotel_id", "city_id", "star_rating", "accommadation_type_name", "chain_hotel", "booking_date",
                 "checkin_date", "checkout_date"]


# ## Dataframe concating

# In[8]:


#

df = pd.concat([city_A, city_B, city_C, city_D, city_E], axis= 0, ignore_index=True)


# In[9]:


#

df


# ## Convert the dataset into one original dataframe so I can start data processing

# In[10]:


df.to_csv("the_full_data.csv", index = False)


# In[ ]:





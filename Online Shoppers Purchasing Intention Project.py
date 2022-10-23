#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Potential Questions: 
# 
#     - 1. What's the page conversion rate?
#     - 2. How likely the page visitor is going to purchase products?
#     - 3. Are the Special days/months that can optimise more revenue?
#     - 4. What are the main proporations of Revenue factor?
#     

# # Data Features:
# 
# 
# `Administrative:` This is the number of pages of this type (administrative) that the user visited.
# 
# `Administrative_Duration:` This is the amount of time spent in this category of pages in seconds.
# 
# `Informational:` This is the number of pages of this type (informational) that the user visited.
# 
# `Informational_Duration:` This is the amount of time spent in this category of pages in seconds.
# 
# `ProductRelated:` This is the number of pages of this type (product related) that the user visited.
# 
# `ProductRelated_Duration:` This is the amount of time spent in this category of pages in seconds.
# 
# `BounceRates:` The percentage of visitors who enter the website through that page and exit without triggering any additional tasks.
# 
# `ExitRates:` The percentage of pageviews on the website that end at that specific page.
# 
# `PageValues:` The average value of the page averaged over the value of the target page and/or the completion of an eCommerce
# 
# `SpecialDay:` This value represents the closeness of the browsing date to special days or holidays (eg Mother's Day or Valentine's day) in which the sessions are more likely to be finalized with transaction.
# 
# `Month:` The month of the year.
# 
# `OperatingSystems`
# 
# `Browser`
# 
# `Region`
# 
# `TrafficType`
# 
# `VisitorType`
# 
# `Weekend`
# 
# `Revenue`

# ## <p style="background-color:#87CEEB; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Step One: Data Importing</p>

# In[1]:


# Import the essential libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load the raw dataset

df = pd.read_csv("online_shoppers_intention.csv")
df


# In[3]:


# Check the d-type of the data frame "df"
df.info()


# In[4]:


# Check the shape of the data-frame

df.describe().T


# ## <p style="background-color:#87CEEB; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Step Two: Data Wrangling</p>

# In[5]:


# Counting the missing values in the dataframe

missing_values = df.isnull().sum()/len(df)
missing_values.sort_values(inplace = True)
missing_values


# In[6]:


# Round the next variables in the data frame

df["ProductRelated_Duration"] = round(df["ProductRelated_Duration"],3)
df["BounceRates"] = round(df["BounceRates"], 3)
df["ExitRates"] = round(df["ExitRates"], 3)
df["PageValues"] = round(df["PageValues"], 3)
df["Administrative_Duration"] = round(df["Administrative_Duration"], 3)


# ## <p style="background-color:#87CEEB; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Step Three: Essential Visualisation</p>

# In[17]:


# Plot the Histogram

df.hist(figsize = (18,18), color = "darkseagreen")


# In[8]:


# Plot the Heatmap

plt.figure(figsize = (14,14))
sns.heatmap(df.corr(), annot = True, cmap="Blues")


# ## Interpretations:
# 
# 
#     - As we can see from the heatmap that the Revenue variable has corelations with the following variables:
#         - Page Values > moderate positive corelationship
#         - Product Related > weak positive corelationship
#         - Administrative > weak psotive corelationship

# ## <p style="background-color:#87CEEB; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Step Four: Exploratory Data Analysis (EDA)</p>

# ## Q1. What's the conversion rate for the page?

# In[9]:


# Calculating the Conversion Rate

CR = round(len(df[df["Revenue"] == True]) / len(df["Revenue"]), 4)*100
print("The Conversion Rate for this page is:",CR, "%")


# ## Q2. How likely the page visitor is going to purchase products?

# 
# #### To understand the possability of a page visitor to make a transaction, we need to highlight the time period the visitor will spend on the page, then how probability of making a transaction which follows the next table guidelines:
# 

# | Duration on the page | What it means |
# | :- | -: |
# | Less than 5 secs | Not interested |
# | Between 5 secs & 500 secs | Very interested |
# | Between 500 secs & 1000 secs | Interested |
# | More than 1000 secs | Probably went to get coffee |
# | More than 1500 secs | Interrupted or went away from the page without making a transaction |

# In[10]:


# Create scatter chart to highlight the proporation

plt.figure(figsize = (12,12))
sns.scatterplot(data = df, x = df["Informational_Duration"], y = df["PageValues"],  size = df["Revenue"],hue = df["Revenue"], 
                               palette=("lightblue", "red"), alpha=0.4, sizes=(60,600), legend = True)
plt.grid(axis = "x")
plt.legend(title = "Revenue Scale",fontsize = 14)
plt.show()


# ## Interpretation:
# 
#     - It's save to consider that the time period sepent if it's less than total of 1000 seconds, the highest possability for the vistior is going to do a transaction with the percentage of 15.47% conversion ratio, considering the page value has an average value above 50 compared to the other relative pages the visitor clicked.

# ## 3. Are the Special days/months that can optimise more revenue?

# In[11]:


#

plt.figure(figsize = (12,12))
sns.barplot(data = df, x = "SpecialDay", y = "Revenue", palette = "cubehelix")

plt.grid(axis = "y")
plt.show()


# ## Interpretation:
# 
#     - Based on the previous bar plot, we see that from scale 0 to 1 "as Special Day", if the day isn't relative to Mother's day or Valentine's day (when the value = 0), there is a larger possability that the visitor will do a transaction more than if the day is related to special day (when the value = 1). So we conclude, it's not relevant, and the next plot will highlight the specific months.

# In[12]:


plt.figure(figsize = (12,12))
sns.barplot(data = df, x = "Month", y = "Revenue", palette = 'Blues')
plt.grid(axis = "y")
plt.show()


# ## Interpretation:
# 
#     - We can see that the top 5 months that recorded the highest revenue are:
#         - November
#         - October
#         - September
#         - August
#         - July

# ## 4. What are the main proporations of Revenue factor?

# In[13]:


def scatter_chart(arg):
    plt.figure(figsize = (8,6))
    sns.barplot(x = df["Revenue"], y = arg, color = ("lightskyblue"))
    plt.grid(axis = "y")
    plt.show()
    
scatter_chart(df["PageValues"])


# In[14]:


scatter_chart(df["ProductRelated"])


# In[15]:


scatter_chart(df["Informational"])


# ## <p style="background-color:#87CEEB; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Step Five: Conclusion</p>

# ### - 1. What's the page conversion rate?
# 
#     - The conversion rate is 15.47%
#     
# ### - 2. How likely the page visitor is going to purchase products?
# 
#     - It's save to consider that the time period sepent if it's less than total of 1000 seconds, the highest possability for the vistior is going to do a transaction with the percentage of 15.47% conversion ratio, considering the page value has an average value above 50 compared to the other relative pages the visitor clicked.
# 
# ### - 3. Are the Special days/months that can optimise more revenue?
# 
#     - Based on the previous bar plot, we see that from scale 0 to 1 "as Special Day", if the day isn't relative to Mother's day or Valentine's day (when the value = 0), there is a larger possability that the visitor will do a transaction more than if the day is related to special day (when the value = 1). So we conclude, it's not relevant, and the next plot will highlight the specific months.
#     
#     - We can see that the top 5 months that recorded the highest revenue are:
#         - November
#         - October
#         - September
#         - August
#         - July
#         
# ### - 4. What are the main proporations of Revenue factor?
# 
#     - This insight is obtained from the Heatmap "the corelation percentage", so the main proporations are:
#     Page Values - Product Related - Informational. 
#     
# 
# 
# ##### I have highlighted the description associated with these variables at the top of this project.
#     

# In[ ]:





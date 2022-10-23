#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# ![](07.10.2022_08.32.33_REC.png)

# ![](07.10.2022_08.33.01_REC.png)

# ![](07.10.2022_08.33.35_REC.png)

# In[ ]:





# ## Guidelines based on this analysis:
# 
# https://www.sciencedirect.com/science/article/pii/S2352340918315191
#     
# https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand   

# # Questions:
# 
#     - 1. What's the cancellation rate?
#     - 2. What's the highest day/month for the cancellation rate?
#     - 3. Is cancellation rate related to single/married type?
#     - 4. What's the proportion of the cancellation rates?
#     - 5. Which Booking Distribution Channel has the highest cancelation rates and impact on overall business numbers?

# In[ ]:





# # <font color='blue'>1. Data Preprocessing</font>
# 

# In[1]:


# Import the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mode


# In[2]:


# Load the raw data frame "df"

df = pd.read_csv("hotel_bookings.csv")
df


# In[3]:


# Checking the data D-types and overall insights

df.info()


# In[4]:


# Checking the statistical proporation of each column in df

df.describe()


# # <font color='blue'>2. Data Wrangling</font>

# In[5]:


# Missing value counts in the Data Frame

missing_values = df.isnull().sum()/len(df)
missing_values = missing_values[missing_values > 0]
missing_values.sort_values(inplace=True)
missing_values


# In[6]:


# Drop the agent & company as both have the highest missing values > 90& of the total values in the column
# Imputation technique or KNN can't help in predicting those values

df.drop(["agent", "company"], axis = 1, inplace = True)


# In[7]:


# Fille the null values with mode; the most repeative values in each column

df["country"] = df["country"].fillna(df["country"].mode()[0])
df["children"] = df["children"].fillna(df["children"].mode()[0])


# In[8]:


df


# # <font color = 'blue'>3. Explatory Data Analysis</font>
# 
# 

# In[9]:


df.hist(figsize = (18,18))


# In[10]:


plt.figure(figsize = (18,18))

sns.heatmap(data = df.corr(), cmap="YlGnBu", annot=True)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)

plt.show()


# ## **1. What's the cancellation rate?**

# In[11]:


# Calculating the ratios

print("The ratios of non-cancelled reservations is:", (round((df["is_canceled"][df["is_canceled"] == 0]).count()/df["is_canceled"].count(), 2))*100, "%")
print("The ratios of cancelled reservations is:", (round((df["is_canceled"][df["is_canceled"] == 1]).count()/df["is_canceled"].count(), 2))*100, "%")


# In[12]:


######### Essential plot for Tableau

plt.figure(figsize=(12,12))
ax = sns.countplot(y="is_canceled", data=df, palette="Pastel1")
total = len(df['is_canceled'])

for v in ax.patches:
        percentage = '{:.1f}%'.format(100 * v.get_width()/total)
        x = v.get_x() + v.get_width() / 1.5
        y = v.get_y() + v.get_height()/2
        ax.annotate(percentage, (x, y), fontsize = 16, weight='bold')

        
plt.show()


# https://matplotlib.org/stable/gallery/color/named_colors.html
# http://rstudio-pubs-static.s3.amazonaws.com/5312_98fc1aba2d5740dd849a5ab797cc2c8d.html


# ## **2. What's the highest day/month for the cancellation rate?**

# In[13]:


df.groupby("arrival_date_day_of_month")["is_canceled"].value_counts().unstack().plot.bar(figsize=(10,7), 
                                                                                          color =('pink', 'teal'))


# In[14]:


# Highlight the unique values

df["arrival_date_year"].value_counts()


# In[15]:


# Select the observations for years 2016 & 2017

data = df[(df["arrival_date_year"] == 2016) | (df["arrival_date_year"] == 2017)]

# https://stackoverflow.com/questions/67332003/pandas-select-rows-from-a-dataframe-based-on-column-values


# In[16]:


# Plot the highest month for cancellation rates.

data.groupby(["arrival_date_month", "arrival_date_year"])["is_canceled"].value_counts().unstack().plot.bar(
    figsize=(10,7), color =('coral', 'deepskyblue'))

plt.legend(title = "Is Canceled", loc ='upper left')
plt.xlabel("Arrival Month per Year", fontsize=15)
plt.ylabel("Number of Values", fontsize=15)
plt.title("Count the Cancellation Rates for 2016/2017 Year", fontsize=20)
plt.grid(axis="y")
plt.show()


# In[17]:


# Plot all the years/months data "for illustration"

df.groupby(["arrival_date_month", "arrival_date_year"])["is_canceled"].value_counts().unstack().plot.bar(figsize=(10,7), 
                                                                                          color =('pink', 'teal'))


# In[18]:


df.groupby("country")["is_canceled"].count().nlargest(35).sort_values(ascending = False)


# In[19]:


#

l = df.groupby("country")["is_canceled"].value_counts().nlargest(50).sort_values(ascending = True).plot.barh(figsize=(18,18), color =('pink', 'teal'))

l.bar_label(l.containers[0], fontsize = 15, label_type='edge')
plt.tight_layout()

l.spines['top'].set_visible(False)
l.spines['right'].set_visible(False)

plt.xlabel("The Total Number", fontsize = 15)
plt.ylabel("The Cancellation Rate per Country", fontsize = 15)

l.grid(axis="y")


# ## **3. Is cancellation rate related to single/married type?**

# In[20]:


dd = df.groupby("is_canceled")[["adults", "children", "babies"]].value_counts().to_frame(name = "Total_Counts").reset_index()


# In[21]:


data = dd[(dd["is_canceled"] == 1) & ((dd["children"] != 0) | (dd["babies"] != 0))]
data

# Since it's impossible to see 0 adults and 2 children go in a trip or booking a hotel, this considered as 
# system error and I will replace it with the mode "the most repeated value in the column".


# In[22]:


data.info()


# In[23]:


from statistics import mode

data["adults"] = round(data["adults"].replace(0, mode(data["adults"])),0)
data


# In[24]:


data.groupby("Total_Counts")[["adults", "children","babies"]].sum().plot.bar(figsize = (12,12), color =('lightcoral','olivedrab', 'darkviolet'))

plt.xlabel("Total Counts", fontsize = 14)
plt.ylabel("The number of Adults with (Children and/or Babies) who Cancelled", fontsize = 14)
plt.title("The Proporation of Cancelled Rates", fontsize = 18)
plt.legend(["Adults", "Children", "Babies"], fontsize = 12)
plt.grid(axis = "y")
plt.show()


# ## **4. What's the proportion of the cancellation rates?**

# In[27]:


#

fig, axes = plt.subplots(1, 2, figsize=(20,15))

sns.boxplot(x='is_canceled', y="lead_time", data=df, palette="BuPu", orient='v', ax=axes[0])
sns.boxplot(x='is_canceled', y="total_of_special_requests", data=df, palette="BuPu", orient='v', ax=axes[1])
            
fig.suptitle('The Main Characteristics for the Cancellation Rates', fontsize = 20)

axes[0].set_ylabel("Lead Time",fontsize = 20)
axes[0].set_xlabel("Is Canceled",fontsize = 20)

axes[1].set_ylabel("Total Sepcial Requests",fontsize = 20)
axes[1].set_xlabel("Is Canceled",fontsize = 20)

axes[0].grid(axis="y")
axes[1].grid(axis="y")

plt.show()

# https://dev.to/thalesbruno/subplotting-with-matplotlib-and-seaborn-5ei8

#https://stackabuse.com/seaborn-box-plot-tutorial-and-examples/


# # Interpretation:
# 
#     -- The reason I have picked the "Lead time" & "Previous cancellations" variables are both of them recorded high corelation with "Is canceled" variable in Heatmap. With values in respect 0.29 & 0.11.
#     -- I wanted to discover the statistical interactions around this relationships.
#     -- As we can see from the box plot that the positive cancellation average is around 130 days.
#     Since lead time: represents the gap in days between the entering date of the booking into the system and the arrival date. 
#     -- From the previous point we can conclude that the longer it recording the reservation and checkin-in the higher possability to cancel the reservation.
#     -- Unexpected, there is no solid relationship between the possability of cancellation and the number of previous cancellation.

# ## **5. Which Booking Distribution Channel has the highest cancelation rates and impact on overall business numbers?**

# In[27]:


df.groupby("distribution_channel")["is_canceled"].value_counts().unstack().plot.bar(figsize=(10,7), 
                                                                                          color =('lightcoral', 'steelblue'))

plt.grid(axis = "y")


# In[35]:


# Calculate the cancelation ratio for the TA & To variable only

Cancelation_Ratio = round(df["is_canceled"].sum()/len(df["distribution_channel"]), 4)*100
print("The Ratio of Cancelation Rates from TA/TO Channel is:", Cancelation_Ratio)


# ## Interpretation:
# 
#     - Based on the previous bar plot, we found that travel agencies (TA) and travel operators (TO) recorded the highest percentage of cancelation rates, which is close to 40k for a period of time 3 years and ratio equal 37.04%
#     
#     - Recommendation; Highly run detailed analysis related to these two channels to measure the factors related to the high rates, and identify the cancelation is related to specific agencies or not. Moreover, is it related to specific months, or follow seasonal trends or not.

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# ### Kindly note that all the graphs in this notebook or the slide file follow the Agoda Color Guidelines as mentioned in this link:
# 
# ##### https://www.agoda.com/press/agoda-logo-guidelines?cid=1844104

# ## Some things to think about:
# 
# **At which part of the funnel should this message be implemented? On a homepage, search results page, at the property level, or booking form?**
# 
#     Answer: Homepage, results page, and personalized notification on the user's email. As it says it needs to start with an action verb and urgent sense to act. Using my analysis can highlight the percentage the user can save while booking the desired destination.
#     
# **What assumptions are you making about the data?**
# 
#     Answer: Main focus was:
#         - Select the most revenue hotels and resorts in the different cities.
#         - Create a hypothesis regarding time period between booking and the check-in.
#         - Trying to use the model to predict the hotel prices values using linear regression and gradiant boosting regressor, and evaluate the accuracy of models.
#         - Checking the booking days that have high cumulative total revenue for all cities.
#         - Checking the profits of the hotels that the users booked in less than 20 days, since this data is related to the prime months. That means this category pay more on specific hotels and I wanted to check this factor as it will definately add more value when contributing the discounts and offers.
#     
# **What factors play a role on price in this analysis?**
# 
#     Answer: Most important factor in this analysis that positively impact the prices is the hotel rating - Stay duration (The time period between the check-in and check-out from the hotel) - the duration between the booking and check-in date.
#     Then other factors that have strong negative impact are: Hotel chain - Duration group.
#     
#     Kindly note that the strong negative corelation means the more of something attached the less in the other. like the more time prior between the booking and check-in dates the less money spend on booking. and so forth.
#     
# **What other factors may influence the analysis that isn’t available in the dataset? What would improve the analysis?**
# 
#     Answer: Practical metrics insights I was able to obtain if:
#      
#     - The number of occupied rooms in each hotel: so I can generate the ADR "Average Daily Rate" metric for the hotels.
#     - Average length of stay (ALOS) metric to help me personalise my analysis and categorize the users so I can help with "email-campaigns" where to send personalised discounts to the right users and increase the conversion rates.
#     
#     
#     General features could play a vital role in this analysis are:
#         - Hotel Reviews.
#         - How close the hotel to the city center and malls.
#         - Data for 1 year to check the seasonlaity/trends in this analysis.
#         
# **What conclusions can you draw from the data?**
# 
#     Answer: The popularity of the hotel depends on the hotel rating, there are specific hotels make higher than 150k in less than 3 months the users are willing to pay even in the most busy part of the year.
#     Hotels with ratings 4 then 3 are the most visitable compared to hotels with the other ratings. In addition to the other notes I have clarified previously and in the next sections, kindly read the interpretation below each part for detailed insights.
#     
#     The most frequent duration of staying is 1 day. The booking traffic happens all week except Saturday and Sunday. Both days have the least recorded values.
#     
#     
# **What recommendations would you give to the Product Owner?**
# 
# I have attached the main insights in the slides.

# ## <p style="background-color:#00A9DC; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">Data Importing & Processing</p>

# In[1]:


# Import the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from datetime import date
from statistics import mode
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score


# In[2]:


df = pd.read_csv("the_full_data.csv")
df


# In[3]:


# Check whether dataframe is balanced or imbalanced

round(df["city_id"].value_counts()/len(df)*100, 2)


# **As we can see here the dataframe is imbalanced, as we have two cities have the largest amount of data with a ratio close to 66% of the whole dataframe. Further raw data is needed to draw more solid outcomes. However there's still pretty interesting insights are exist, and that's what I will show in the next sections.**
# 

# In[4]:


# Modify the data values in these columns

df["ADR_USD"] = df["ADR_USD"].str.replace(",", "")
df["hotel_id"] = df["hotel_id"].str.replace(",", "")
df["city_id"] = df["city_id"].str.replace(",", "")


# In[5]:


# Modify the D-type in these columns

df["ADR_USD"] = df["ADR_USD"].astype("float64")
df["hotel_id"] = df["hotel_id"].astype("int64")
df["city_id"] = df["city_id"].astype("int64")
df["accommadation_type_name"] = df["accommadation_type_name"].astype("string")
df["chain_hotel"] = df["chain_hotel"].astype("string")
df["booking_date"] = pd.to_datetime(df["booking_date"], dayfirst=True)
df["checkin_date"] = pd.to_datetime(df["checkin_date"], dayfirst=True)
df["checkout_date"] = pd.to_datetime(df["checkout_date"], dayfirst=True)


# In[6]:


# Double-check that all the values are in the right shape

df.info()


# In[7]:


# Checking the dataframe

round(df.describe(),2)


# **As we can see here there are outliers in the "ADR_USD" variable, because the mean = 148.09 USD & the max = 3156.86 USD & the min = 4.26 USD. So diffenatly we will run different tests to investigate those outliers.**

# In[8]:


sns.boxplot(df["ADR_USD"])


# **From the Box plot, we have 1 values above 3000 USD.**

# ## <p style="background-color:#00A9DC; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">Data Wrangling</p>

# 
# **Delete the observation that has an average above 3000 USD**

# In[9]:


df = df[df["ADR_USD"] != 3156.86]


# **Checking the Null values**

# In[10]:



missing_values = df.isnull().sum()/len(df)
missing_values


# In[11]:


# Count the unique values in the "Chain_Hotel" Column

df["chain_hotel"].value_counts()


# In[12]:


# Applying the label encoding technique

le = preprocessing.LabelEncoder()

df['chain_hotel'] = le.fit_transform(df['chain_hotel'])


# **Since the data observations in the column "Chain_Hotel" is Nominal data, I will use Label Encoding to convert the "chain" values & "Non-chain" values into 0 & 1 in respect.**

# In[13]:


# Label Encodig: 1 for hotel non-chain & 0 for hotel chain

df["chain_hotel"].value_counts()


# In[14]:


# Computing the time difference between the booking and checkin dates

def get_difference(date1, date2):
    delta = date2 - date1
    return delta.dt.days

df["Between_Booking_Checkin"] = get_difference(df["booking_date"], df["checkin_date"])
df["Stay_Duration"] = get_difference(df["checkin_date"], df["checkout_date"])


# In[15]:


# Checking the unique values in the new column "Between_Booking_Checkin"

df["Between_Booking_Checkin"].unique()


# **From the data conversion, I can see that there are 3 observation in the dates between the booking date and checking date have -1 values, which isn't correct, so I will investigate them and if they are wrong recording from the system then I will replace them depends on the data distribution.**

# In[16]:


df[df["Between_Booking_Checkin"] == -1]


# In[17]:


# Plotting histogram

df["Between_Booking_Checkin"].hist(color = "#00A9Dc")
plt.title("The Data Distribution")
plt.xlabel("Number of Days between Booking & Check-in")


# **Since the data in this variable is right skewed, we will replace -1 observations with the mode not the mean.**

# In[18]:


import statistics as s
from statistics import mode

print(s.mode(df["Between_Booking_Checkin"]))


# In[19]:


# Replcaing the values with the mode

df["Between_Booking_Checkin"] = df["Between_Booking_Checkin"].replace(-1, 1)


# In[20]:


# Checking the unique values in the new column "Stay_Duration"

df["Stay_Duration"].unique()


# In[21]:


# Plotting the "Stay_Duration" variable

df["Stay_Duration"].hist(color = "#00A9Dc")
plt.title("The Data Distribution")
plt.xlabel("Number of Days between Check-in & Check-out")


# **The unique values in the column "Stay_Duration" seem normal.**

# In[22]:


# Converting the Booking_date into week names 

df["Booking_Date"] = df["booking_date"].dt.day_name()


# In[23]:


# Plotting bar plot

plt.figure(figsize = (14,8))
df.groupby(df["Booking_Date"])["ADR_USD"].sum().plot.bar(color = "#00A9Dc")

plt.title("Most Frequent Day makes Highest Revenue", fontsize = 12)
plt.xlabel("Most Frequent Day in Booking", fontsize = 12)
plt.ylabel("Total Revenue for each Day in Millions", fontsize = 12)
plt.grid(axis = "y")


# 
# ## <p style="background-color:#00A9DC; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">Explatory Data Analysis </p>

# ## 3. Data Visualisation

# In[24]:


# Plotting Histogram

df.hist(figsize = (14,14), color = "#00A9Dc")


# In[25]:


# Plotting Scatter chart


plt.figure(figsize = (14,14))

sns.heatmap(df.corr(), annot = True, cmap = "Blues")


# ## Interpretation:
# 
#     - Heatmap highlights the corelationship between one variable and every other variables in the dataset.
#     - From this heatmap we can touch that the "ADR_USD" variable as a dependent variable that relies on:
#         - Star_Rating column: with the corelation of 0.36
#         - Days period between Booking Date & Checkin Date: with the corelation of 0.11
#                 #### so far it's still positive relationship, which is good.
#         - Chain_Hotel column: with the corelation of -0.23 "Negative relationship" which means if the hotel is chain that makes the price goes up, but if it's non-chain, the price goes down. Simply, when the price goes in a direction, the hotel type is "chain" then we assume it's going to the other direction, and vise versa.

# 
# 
# ## <p style="background-color:#00A9DC; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">Checking the relationship between the prices & the amount of days "between the booking and checkin dates"</p>

# **In this section:**
#     
#     - I wanted to categorize the hotels into groups, to compare the prices of the hotels if the user book in less than 20 days from the checkin date, then if the user book between 20 days prior & less than 40 days from the checkin date, and lastly between the 40 days prior and 60 days from the checkin date.

# In[26]:


def Prices_Increase(arg1, arg2):
    x = print(round(len(df[(df["Between_Booking_Checkin"] > arg1) & (df["Between_Booking_Checkin"] <= arg2)])/len(df)*100, 2), "%")
    return x


# In[27]:


Prices_Increase(0,20)


# In[28]:


Prices_Increase(20,40)


# In[29]:


Prices_Increase(40,60)


# | Duration in Days Between the Booking & Check-in | Save Money Percentage of |
# | :- | -: |
# | Between 0-20 Days | 60.38% |
# | Between 20-40 Days | 18.49% |
# | Between 40-60 Days | 9.71% |

# **To understand this simple chart, let's apply an example:**
# 
#     - From the first row: If Agoda user booked his hotel in less than 20 days prior. He will be saved from prices exposure of 60.38%. 
#     
#     - Unlike if he decided to book the hotel between 20 to 40 days prior to his check-in, then he's actually would be saved from prices exposure of just only 18.49%.
#     
#     The main idea here is:
#     
#     - Booking your hotel with sufficient days before your check-in date, the sufficient time prior the check-in you give yourself, means more money into your pocket.

# In[30]:


data = df[(df["Between_Booking_Checkin"] < 20)]


# In[31]:


data = data.groupby(["hotel_id", "city_id"])["ADR_USD"].sum().to_frame(name = "Profits").reset_index()


# In[32]:


Most_expensive_hotels = data[data["Profits"] > 10000]
Most_expensive_hotels


# In[33]:


Most_expensive_hotels.groupby("hotel_id")["Profits"].sum().plot.bar(figsize = (18,12), color = "#B01E8D")

plt.title("Expensive Hotels Booked in less than 20 Days (Visitor Choice) ", fontsize = 14)
plt.ylabel("Profits in Less than 3 Months", fontsize = 14)
plt.xlabel("Hotel ID", fontsize = 14)


plt.grid(axis = "y")
plt.show()


# ## Interpretation:
# 
#     - As we can see here, there are specific hotels have impressive revenue in less than 3 months, and the fun part is that the users booked those hotels less than 20 days prior in the October, Novemeber and December which is the most busy quarter of every year due to public holidays.
#     
#     - These users don't mind to pay extra for those hotels, so this part need further digging in the funnel. As 

# In[34]:


data.groupby("city_id")["Profits"].sum().plot.bar(figsize = (10,8), color = "#B01E8D")


# In[35]:


def barh_plot(arg):
    plt.figure(figsize = (10,8))
    df.groupby(arg)["ADR_USD"].sum().plot.barh(color = "#00A9Dc")
    plt.grid(axis = "x")
    
barh_plot("accommadation_type_name")


# ## Interpretation:
# 
#     - Still the hotels recorded the highest revenue. Followed by resorts then serviced apartment.

# In[36]:


barh_plot("star_rating")


# ## Interpretation:
# 
#     - Strangly, the most revenue came from hotels with 4 stars then 3 stars. Followed by the 5 stars hotels.

# In[37]:


barh_plot("Stay_Duration")


# ## Interpretation:
# 
#     - The most frequent time period for the staying is just 1 day. The numbers are in millions.

# ## <p style="background-color:#00A9DC; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">Checking the most profitable resorts in the whole cities</p>

# 
# ##### Kindly note that the city with id = 16808 doesn't have any resorts.

# In[38]:


most_profitable_resort = df[df["accommadation_type_name"] == "Resort"]


# In[39]:


def barh_plot_for_resorts(arg):
    dataframe = most_profitable_resort[most_profitable_resort["city_id"] == arg]
    dataframe.groupby("hotel_id")["ADR_USD"].sum().plot.barh(figsize = (12,14), color = "#0DB14B")
    plt.grid(axis = "x")
    
    
barh_plot_for_resorts(9395)


# In[40]:


barh_plot_for_resorts(17193) 


# ## Interpretation:
# 
#     - We can see that the resort with id = 3644 recorded a revenue close to 35k in less than 3 months.
#     - Then the resort with id = 1968534 recorded a revenue close to 15k in less than 3 months.

# In[41]:


barh_plot_for_resorts(5085) 


# **Only one resort in this city, which is quite strange and unlikely, so maybe we need to double-check the data on the system for this city's resorts.**

# In[42]:


barh_plot_for_resorts(8584)


# ## Interpretation:
# 
#     - We can see that the resort with id = 21720 recorded a revenue close to 140k in less than 3 months. It's impressive and definately we need to include this resort in the offers and discounts to attaract more visitors.

# ## <p style="background-color:#00A9DC; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">Checking the most profitable resorts in the whole cities</p>

# In[43]:


most_profitable_hotels = df[df["accommadation_type_name"] == "Hotel"]


# In[44]:


def barh_plot_for_hotels(arg):
    dataframe = most_profitable_hotels[most_profitable_hotels["city_id"] == arg]
    dataframe.groupby("hotel_id")["ADR_USD"].sum().plot.barh(figsize = (12,14), color = "#FDB812")
    plt.grid(axis = "x")
    
    
barh_plot_for_hotels(9395)


# In[45]:


barh_plot_for_hotels(17193) 


# In[46]:


barh_plot_for_hotels(5085)


# In[47]:


barh_plot_for_hotels(16808)


# In[48]:


barh_plot_for_hotels(8584)


# ## Interpretation:      "For the previous bar plots"
# 
#     - For the city 9395 we find that:
#         - Hotel(21272) has revenue close to 160k for less than 3 months.
#         - Hotel(197996) has revenue more than 100k for less than 3 months.
#         
#     - For the city 17193 we find that:
#         - Hotel(1243334) has revenue more than 20k for less than 3 months.
#         - Hotel(1244576) has revenue more than 17.5k for less than 3 months.
#         
#     - For the city 5085 we find that:
#         - Hotel(1251372) has revenue close to 175k for less than 3 months.
#         - Hotel(100188) has revenue more than 100k for less than 3 months.
#         - Hotel(2231812) has revenue close to 100k for less than 3 months.
#         
#     - For the city 16808 we find that:
#         - Hotel(16146) has revenue more than 300k for less than 3 months.
#         - Hotel(1545890) has revenue more than 205k for less than 3 months.
#         - Hotel(219762) has revenue more than 180k for less than 3 months.
#         
#     - For the city 8584 we find that:
#         - Hotel(323744) has revenue more than 94k for less than 3 months.
#         - Hotel(374026) has revenue more than 20k for less than 3 months.

# In[49]:


Hotels_Revenues = df.groupby(["hotel_id", "star_rating", "accommadation_type_name"], sort = True)["ADR_USD"].sum().to_frame(name = "Total_Revenue").reset_index()
Hotels_Revenues


# In[50]:


# set seaborn "whitegrid" theme
sns.set_style("white")

plt.figure(figsize=(30,25))

# use the scatterplot function
sns.scatterplot(data=Hotels_Revenues, y="Total_Revenue", x="star_rating",  size="Total_Revenue", hue="accommadation_type_name", 
                palette="flare", edgecolors="black", alpha=0.5, sizes=(200, 2000))

# Add titles (main and on axis)
plt.xlabel("Star Rating", fontsize = 30)
plt.ylabel("Total Revenue", fontsize = 30)
plt.title("The Best Hotels with the Best Revenue", fontsize = 30)

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

# Locate the legend outside of the plot

plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=27)

# show the graph
plt.show()


# https://www.python-graph-gallery.com/bubble-plot-with-seaborn
# https://seaborn.pydata.org/tutorial/color_palettes.html


# In[51]:


Hotels_Revenues.star_rating.unique()


# In[52]:


# set seaborn "whitegrid" theme
sns.set_style("white")


#mme = [0,1,1.5,2,2.5,3,3.5,4,4.5,5] 

plt.figure(figsize=(80,70))

# use the scatterplot function
ax = sns.scatterplot(data=Hotels_Revenues, x="star_rating", y="Total_Revenue", size="Total_Revenue", hue="accommadation_type_name",
                palette="bright", edgecolors="black", alpha=0.5, sizes=(25,90000), legend = True)

# Add titles (main and on axis)
plt.xlabel("Star Rating for Hotels", fontsize = 60)
plt.ylabel("Revenue Amount", fontsize = 60)
plt.title("The Revenue of each Product per Month", fontsize = 60)

#sns.set_xticklabels(mme)
plt.xticks([0,1,1.5,2,2.5,3,3.5,4,4.5,5], fontsize = 45)
plt.yticks(fontsize = 45)


# Locate the legend outside of the plot

for line in range(0,Hotels_Revenues.shape[0]):
     ax.text(Hotels_Revenues.star_rating.iloc[line], Hotels_Revenues.Total_Revenue.iloc[line], Hotels_Revenues.accommadation_type_name.iloc[line], horizontalalignment='left', size='xx-large', color='black', weight='semibold')

plt.legend(loc='upper left', fontsize=80)

# show the graph
plt.show()


# ## In this section, I have divided the whole dataframe following the table I have created earlier

# In[53]:


#

df["Duration_Group"] = np.where(df["Between_Booking_Checkin"] <= 20, 1, (np.where(((df["Between_Booking_Checkin"] > 20) & (df["Between_Booking_Checkin"] <= 40)), 2, ( np.where(((df["Between_Booking_Checkin"] > 40) & (df["Between_Booking_Checkin"] <= 60)), 3, "")))))


# In[54]:


df


# In[55]:


df.to_csv("New_Updated_df.csv", index = False)


# In[91]:


# Plotting the Pie chart

df["Duration_Group"].value_counts().plot.pie(labels = ["From 0-20", "From 20-40", "From 40-60"], figsize = (10,10), colors = ["#00A9DC", "#B01E8D", "#FDB812"])

plt.title("The Duration between the Booking Date & Check-in Date", fontsize = 14)
plt.ylabel("Duration Scale", fontsize = 14)

plt.show()

# https://www.w3schools.com/python/matplotlib_pie_charts.asp


# In[92]:


# Checking the profits per days and groups

data = df.groupby(["Duration_Group", "Booking_Date"])["ADR_USD"].sum().to_frame(name = "Profits").reset_index()
data


# In[93]:


plt.bar(data["Booking_Date"], data["Profits"], color = "#585856")


# ## <p style="background-color:#00A9DC; font-family:newtimeroman; color:#FFFFFF; font-size:175%; text-align:center; border-radius:10px 10px;">AB Testing</p>

# In[140]:


df["Duration_Group"] =  df["Duration_Group"].astype("int64")


# In[141]:


df.corr()


# ### Hypothesis Statement is:
# 
# | Hypothesis Statement | Identification | Critical Value | Decision |
# | :- | -: | :- | -: |
# | Null Hypothesis (H0) | Duration Group has impact less than 9% on the prices | < 0.05 | Reject H0 |
# | Alternative Hypothesis (H1) | Duration Group crucial impact on the prices deregarding the correlation ratio | > 0.05 | Accept H0 |

# In[137]:


from scipy import stats 

t, p = stats.ttest_ind(df.loc[df['Duration_Group'] == "1", 'ADR_USD'],
                       df.loc[df['Duration_Group'] == "2", 'ADR_USD'], 
                       equal_var=False)
print("p-value = {:.4f}, thus Promotion 1 and 2 are statistically similar".format(p))

t, p = stats.ttest_ind(df.loc[df['Duration_Group'] == "1", 'ADR_USD'],
                       df.loc[df['Duration_Group'] == "3", 'ADR_USD'], 
                       equal_var=False)
print("p-value = {:.4f}, thus Promotion 1 and 3 are statistically similar".format(p))

t, p = stats.ttest_ind(df.loc[df['Duration_Group'] == "2", 'ADR_USD'],
                       df.loc[df['Duration_Group'] == "3", 'ADR_USD'], 
                       equal_var=False)
print("p-value = {:.4f}, thus Promotion 2 and 3 are statistically different".format(p))


# ### ANOVA Test

# In[135]:



# a group of Promotions
PromotionNumber = df["Duration_Group"].unique()


d_data = {Duration:df[df['Duration_Group'] == Duration]['ADR_USD'] for Duration in PromotionNumber}

# apply Anova to 3 groups
F, p = stats.f_oneway(d_data["1"], d_data["2"], d_data["3"])
print("p-value: {}, thus rejecting the null hypothesis".format(p))


# ## Check the linearity 

# In[96]:


from sklearn.linear_model import LinearRegression


# In[97]:


x = df.drop('ADR_USD', axis=1)
y = df['ADR_USD']


# In[98]:


x.drop(["hotel_id", "city_id", "accommadation_type_name", "booking_date", "checkin_date", "checkout_date", "Booking_Date"], axis = 1, inplace = True)


# In[99]:


from sklearn.linear_model import LinearRegression


reg = LinearRegression().fit(x, y)
coef = pd.Series(reg.coef_, index = x.columns)

imp_coef = coef.sort_values()
imp_coef.plot(kind = "barh", color = "#585856")
plt.title("Feature importance using Linear Model")


# # Linear Prediction:

# In[63]:


df.drop(["hotel_id", "city_id", "accommadation_type_name", "booking_date", "checkin_date", "checkout_date", "Booking_Date"], axis = 1, inplace = True)


# In[64]:


df


# In[65]:


df.corr()


# In[ ]:





# In[67]:


#spliting the features into train & test sets

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.loc[:, df.columns != "ADR_USD"], df["ADR_USD"], test_size = 0.2, random_state = 0)


# In[68]:


#checking the shape of the training & test sets
print('Shape of the training patterns:', x_train.shape,y_train.shape)
print('Shape of the testing patterns:', x_test.shape,y_test.shape)


# In[69]:


#defining a linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[70]:


#fitting the linear regression model
regressor.fit(x_train,y_train)


# In[71]:


#checking the regression coefficients of the fitted model
print('The regression coefficients are:', regressor.coef_)


# In[72]:


#checking the intercept
print('The intercept is:', regressor.intercept_)


# In[73]:


#making the prediction on the test data
y_pred = regressor.predict(x_test)

#comparing the predicted profit with actual profit
pd.DataFrame(data={'Predicted Profit': y_pred, 'Actual Profit': y_test})


# In[74]:


#mean squared error (MSE)
from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(y_test, y_pred) 
#y_test: Ground truth (correct) target values.
#y_pred: Estimated target values.


print('Mean Squared Error is:', MSE)


# In[75]:


#Root mean squared error (RMSE)
import math

RMSE = math.sqrt(MSE)
print('Root Mean Squared Error is:', RMSE)


# In[76]:


#R-Squared
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print('R-Squared is:', r2)


# In[77]:


#Adjusted R-Squared

adj = 1-(
        (1-r2)*(x_train.shape[0]-1)/
            (x_train.shape[0]-x_train.shape[1]-1)
        )

print('Adjusted R-Squared is:', adj)


# # Gradient Boosting Regressor

# In[78]:


from sklearn.ensemble import GradientBoostingRegressor

# with default parameters
gbr = GradientBoostingRegressor()


# https://www.datatechnotes.com/2019/06/gradient-boosting-regression-example-in.html


# In[79]:


gbr.fit(x_train, y_train)


# In[80]:


y_preds = gbr.predict(x_test)
mse = mean_squared_error(y_test,y_preds)


# In[81]:


# Generate the MSE value

print("MSE Score is: ", mse)


# In[82]:


# Generate the RMSE value

RMSE = math.sqrt(mse)
print('Root Mean Squared Error is:', RMSE)


# In[83]:


# Generate the accuracy value

r2 = r2_score(y_test, y_preds)

print('R-Squared is:', r2*100)


# In[84]:


# Generate the log-MSE value as I have added more penality on the model

from sklearn.metrics import mean_squared_log_error


r = mean_squared_log_error(y_test, y_preds)
round(r*100,2)

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html


# **As we can see here, in both models the accuracy wasn't great since the raw data is limited and imbalanced. Baseline to generate a solid analysis, at least one year of the data is needed. That's why whenever trying to create a model using the training dataset and then execute this model on the test dataset, we see underfitting.**

# In[ ]:





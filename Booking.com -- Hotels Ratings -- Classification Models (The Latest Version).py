#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # **1. Data Importing**

# In[1]:


# import the required packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
import pycaret.classification as pycc


# In[2]:


#load the raw dataset 

df = pd.read_csv("Hotel Booking.csv")
df


# # 2. Understanding the raw variables & observations

# In[3]:


df.info()


# In[6]:


round(df.describe().T,2)


# # 3. Data Wrangling

# In[7]:


# Checking the null values percentage to eliminate the necessary

missing_values = round(df.isnull().sum()/len(df), 4)*100
missing_values = missing_values[missing_values > 0]
missing_values.sort_values()
missing_values


# In[8]:


#Drop the unnecessary variables

df.drop(["uniq_id", "area","pageurl", "extra_adult_rate", "extra_child_rate", "lat", "long", "address"], axis = 1, inplace = True)


# In[9]:


# Replace the Null values with zeros so I can predict those missing values later on

df["hotel_type"] = df["hotel_type"].fillna(0, inplace = False)


# In[10]:


#Convert the dtype of the variables

df["name"] = df["name"].astype('string')
df["city"] = df["city"].astype('string')
df["amenities"] = df["amenities"].astype('string')
df["hotel_type"] = df["hotel_type"].astype('string')
df["Date"] = pd.to_datetime(df["crawl_timestamp"]).dt.date


# ### Since the hotel Rating values are string not numeric, I had to modify the column a little

# In[11]:


# Modify the hotel rating variable to include the numeric values.

df["hotel_star_rating"] = df["hotel_star_rating"].str.replace(' ', '')

# https://stackoverflow.com/questions/41476150/remove-or-replace-spaces-in-column-names


# In[12]:


# Splitting the hotel rating variable into 3 parts

df[["Name", "sep", "No"]] = df["hotel_star_rating"].str.split("(\d)", n=1, expand=True)


# In[13]:


# Convert the D-type for the Date variable

df = df.astype({"Date": 'datetime64[ns]'})


# In[14]:



df["Rating"] = df["sep"].fillna(0, inplace = False).astype(int)


# In[15]:


# Drop the unncessary variables

df.drop(["sep", "hotel_star_rating", "Name", "No", "crawl_timestamp"], axis = 1, inplace = True)


# ### Replace the NULL values in the whole data frame with zeros to facilitate creating graphs, then I will handle these columns later on using machine learning concepts 

# In[16]:


df["review_count"] = df["review_count"].fillna(0, inplace = False)    ####u can't predict this value as it's a fact
df["average_rating"] = df["average_rating"].fillna(0, inplace = False)   ###### predict!
df["photo_count"] = df["photo_count"].fillna(0, inplace = False) ####u can't predict this value as it's a fact
df["cleanliness"] = df["cleanliness"].fillna(0, inplace = False)
df["facilities"] = df["facilities"].fillna(0, inplace = False)
df["location"] = df["location"].fillna(0, inplace = False)
df["staff"] = df["staff"].fillna(0, inplace = False)
df["wifi"] = df["wifi"].fillna(0, inplace = False)
df["comfort"] = df["comfort"].fillna(0, inplace = False)
df["value_for_money"] = df["value_for_money"].fillna(0, inplace = False)####u can't predict this value as it's a fact


# In[17]:


# Now, check the null values in the data frame

df.isna().sum()


# In[18]:


df["average_rating"] = round(df["average_rating"])

df["average_rating"].unique()


# # 4. Data Visualisation

# In[19]:


# Creating the Histogram to check the data distribution in every variable

df.hist(figsize = (18,18), color = "coral")


# In[20]:


# Creating the Heatmap to see the corelationship between the variables

plt.figure(figsize = (14,14))
sns.heatmap(df.corr(), cmap = 'Blues', annot = True)


# In[21]:


# Creating the Scatter Chart to see the dense of the data

def scatter_chart(arg1, arg2):
    sns.scatterplot(x= df["Rating"], y= arg1, palette = "viridis")
    plt.title(arg2)
    plt.grid(axis = "x")


# In[22]:


scatter_chart(df["review_count"], "The relationship between Rating & Number of Reviews")


# In[23]:


scatter_chart(df["photo_count"],"The relationship between Rating & Number of Photos")


# In[24]:


scatter_chart(df["cleanliness"],"The relationship between Rating & Cleaniness")


# In[25]:


scatter_chart(df["wifi"], "The relationship between Rating & WiFi")


# In[26]:


scatter_chart(df["comfort"], "The relationship between Rating & Comfort")


# In[27]:


# Creating the Bar plot

df.groupby('hotel_type')[["review_count", "Rating"]].mean().plot.bar(figsize=(15,12),  color = ('pink', 'teal'))
plt.xlabel("Hotel Type", fontsize = 18)
plt.ylabel('Count', fontsize = 18)
plt.title('The Properation per Flower Type', fontsize = 18)
plt.grid(axis = "x")
plt.legend(fontsize = 14)


# In[28]:


# Highlight specific values in the Hotel type based on the previous plot to check the Rating Level.

pp = df[(df["hotel_type"] == "Resorts") | (df["hotel_type"] == "Inns") | (df["hotel_type"] == "Hotels") | (df["hotel_type"] == "Hostels")]


# In[29]:


# Creating the Box plot

plt.figure(figsize=(30,20))
ax = sns.boxplot(y="value_for_money", x="Rating", hue="hotel_type", data=pp, palette="Set3")


# Add titles (main and on axis)
plt.xlabel("Rating Scale", fontsize = 30)
plt.ylabel("Value of Money", fontsize = 30)
plt.title("The Top Four Hotel Types", fontsize = 30)

plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=27)
plt.xticks(fontsize = 30)


# In[30]:


# Drop the amenities variable for now to start predicting the missing values

df.drop("amenities", axis = 1, inplace = True)


# In[31]:


# Using the Label Encoding technique

le = preprocessing.LabelEncoder()

df['hotel_type'] = le.fit_transform(df['hotel_type'])


# In[32]:


# Drop the unnecessary variables

df.drop(["Date", "name", "hotel_id", "city"], axis = 1, inplace = True)


# ## 5. Data Prediction for Rating & Average Rating Variables

# In[33]:


#Ratio of mussing data fro Saving accounts & Checking accounts

print("The Ratio of Missing Values in the Number of Reviews is = ", (df["Rating"] == 0).sum()/1000, "* 100")
print("The Ratio of Missing Values in the Average Rating Variable is = ", (df["average_rating"] == 0).sum()/1000, "* 100")


# In[34]:


# Splitting the data frame into training & testing sub-sets

X_test_saving_data = df[df["average_rating"] == 0]
X_test_saving_data


# In[35]:


# Checking the shape

X_test_saving_data.shape


# In[36]:


X_train_saving_data = df.loc[~((df['average_rating'] == 0))]
X_train_saving_data

# https://stackoverflow.com/questions/49841989/python-drop-value-0-row-in-specific-columns


# In[37]:


X_train_saving_data.shape


# In[38]:


y_train_saving_data = X_train_saving_data['average_rating']
y_train_saving_data


# In[39]:


y_train_saving_data.shape


# In[40]:


# Using the KNN for predicting the missing values for the average rating first

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=9)

knn.fit(X_train_saving_data, y_train_saving_data)


# In[41]:


# PRedicting formula

y_predict = knn.predict(X_test_saving_data)


# In[42]:


# Check the outputs

print(y_predict)


# In[43]:


# Double check the shape

y_predict.shape


# In[44]:


# Merge the both sub-sets back together 

X_test_saving_data["average_rating"] = X_test_saving_data.loc[X_test_saving_data["average_rating"] == 0, "average_rating"] = y_predict

#https://stackoverflow.com/questions/61238384/replacing-values-in-pandas-dataframe-column-with-same-row-value-from-another-col


# In[45]:


# Load the outputs

X_test_saving_data


# In[46]:


# Using Concat() function to glow the whole datasets all together

df00 = pd.concat([X_train_saving_data, X_test_saving_data], 0, ignore_index=True)


# https://stackoverflow.com/questions/46269804/concatenating-dataframes-on-a-common-index


# ### Since Rating column can't be zeros as in all websites it's a selection input from 1 to 5, that's why I have to replace the zeros in this dataframe "df"

# In[47]:


X_test_saving_data0 = df[df["Rating"] == 0]
X_test_saving_data0


# In[48]:


X_test_saving_data0.shape


# In[49]:


X_train_saving_data0 = df.loc[~((df['Rating'] == 0))]
X_train_saving_data0


# In[50]:


X_train_saving_data0.shape


# In[51]:


y_train_saving_data0 = X_train_saving_data0['Rating']
y_train_saving_data0


# In[52]:


y_train_saving_data0.shape


# In[53]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
  
knn.fit(X_train_saving_data0, y_train_saving_data0)


# In[54]:


y_pred = knn.predict(X_test_saving_data0)


# In[55]:


print(y_pred)


# In[56]:


y_pred.shape


# In[57]:


X_test_saving_data0["Rating"] = X_test_saving_data0.loc[X_test_saving_data0["Rating"] == 0, "Rating"] = y_pred


# In[58]:


df0 = pd.concat([X_train_saving_data0, X_test_saving_data0], 0, ignore_index=True)


# ## 6. Replace the correct, predicted values with the old ones in the original dataframe

# In[59]:


df["rating"] = df0["Rating"]


# In[60]:


df["Average Rating"] = df00["average_rating"]


# In[61]:


# Original Data frame

df


# In[62]:


# Drop the old variables

df.drop(["average_rating", "Rating"], axis = 1 , inplace = True)


# ## 7. Clustering Analysis "To split the data into groups"

# In[63]:


# Split the values

x = df.iloc[:, :-1].values
y = df.iloc[:,-1].values

#checking the shape of the input and output features

print('Shape of the input features:', x.shape)
print('Shape of the output features:', y.shape)


# In[64]:


#spliting the features into train & test sets

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state=0)


# In[65]:


#checking the shape of the training & test sets
print('Shape of the training patterns:', x_train.shape,y_train.shape)
print('Shape of the testing patterns:', x_test.shape,y_test.shape)


# In[66]:


# Calculating the euclidean distance matrix

from sklearn.cluster import KMeans
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    cs.append(kmeans.inertia_)
    
# plot the euclidean distance matrix

plt.figure(figsize = (20,15))

plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()


# In[67]:


# Building model using kmeans

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5,random_state=0)

kmeans.fit(x)

labels = kmeans.labels_

# check how many of the samples were correctly labeled

correct_labels = sum(y == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))

print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# In[68]:


kmeans = KMeans(5)
kmeans.fit(x)


# In[69]:


identified_clusters = kmeans.fit_predict(x)


# In[70]:


identified_clusters = pd.DataFrame(identified_clusters)
identified_clusters.value_counts()


# In[71]:


identified_clusters.columns = ["Service Level"]


# In[72]:


identified_clusters0 = identified_clusters["Service Level"].replace([0, 1, 2, 3, 4], ['Ususal Service', "Great Service", "High Service", "Very High Service", "Exceptional Service"])


# In[73]:


identified_clusters0.value_counts()


# In[74]:


identified_clusters0 = pd.DataFrame(identified_clusters0)


# In[75]:


identified_clusters0.columns = ["Service Level"]


# In[76]:


identified_clusters0


# In[77]:


df["Service Level"] = identified_clusters0["Service Level"]


# In[78]:


df = df[df["Service Level"] != 'NaN']


# In[79]:


df


# ## 8. Building Classification models using Pycaret

# In[102]:


summary_preprocess = pycc.setup(df, target = 'Service Level')

# https://michael-fuchs-python.netlify.app/2022/01/01/automl-using-pycaret-classification/


# In[103]:


x = pycc.get_config('X')
y = pycc.get_config('y')
trainX = pycc.get_config('X_train')
testX = pycc.get_config('X_test')
trainY = pycc.get_config('y_train')
testY = pycc.get_config('y_test')


# In[104]:


x


# In[105]:


y


# In[106]:


available_models = pycc.models()
available_models


# In[107]:


best_clf = pycc.compare_models()


# In[108]:


best_clf_specific = pycc.compare_models(include = ['knn','lr', 'nb', 'dt', 'svm'])


# In[109]:


evaluation_best_clf = pycc.evaluate_model(best_clf_specific)


# In[ ]:





# ## 9. Word Clouding

# In[110]:


# Import the required libraries

import numpy as np 
import pandas as pd
import re
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt


# In[111]:


# Load the original data to extract specific variables

raw_data = pd.read_csv("Hotel Booking.csv")


# In[112]:


# Create new sub-dataset from the original data frame 

Amenities_Avg_Rating = raw_data[['average_rating', 'amenities']]
Amenities_Avg_Rating


# In[114]:


Amenities_Positive_Rating = Amenities_Avg_Rating[Amenities_Avg_Rating["average_rating"] >= 6]


# In[115]:


Amenities_Positive_Rating["amenities"] = Amenities_Positive_Rating["amenities"].str.replace('|', ' ')
Amenities_Positive_Rating["amenities"] = Amenities_Positive_Rating["amenities"].str.replace('amp', '')
Amenities_Positive_Rating["amenities"] = Amenities_Positive_Rating["amenities"].str.replace('1 ', '')
Amenities_Positive_Rating


# In[116]:


def wordCloud_generator(Amenities_Positive_Rating, title=None):
    wordcloud = WordCloud(width = 800, height = 800,
                          background_color ='white',
                          min_font_size = 11, colormap='RdBu'
                         ).generate(" ".join(Amenities_Positive_Rating.values))                      
    plt.figure(figsize = (18, 18), facecolor = None) 
    plt.imshow(wordcloud, interpolation='bilinear') 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.title(title,fontsize=30)
    plt.show()
    
wordCloud_generator(Amenities_Positive_Rating['amenities'], title="Top Words in Positive Amenities")


# In[ ]:





# In[ ]:


# https://medium.com/analytics-vidhya/predicting-the-ratings-of-reviews-of-a-hotel-using-machine-learning-bd756e6a9b9b


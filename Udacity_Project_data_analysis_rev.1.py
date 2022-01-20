#!/usr/bin/env python
# coding: utf-8

# 
# # Project: Investigate a Dataset (TMDB movie dataset)
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# This data set contains information about 10,000 movies collected from The Movie Database (TMDb), including user ratings and revenue.
# 
# I analysed the Dataset and tried to answer the folowing questions through this notebook:-
# 
# 1-Which year has the highest release of movies?
# 
# 2-Top 10 movies which earn highest/Lowest profit?
# 
# 3-Movie with Highest And Lowest Budget?
# 
# 4-Which movie made the highest revenue and lowest as well?
# 
# 5-Movie with shorest and longest runtime?
# 
# 6-Which movie get the highest or lowest votes (Ratings).
# 
# 7-which year has the highest release of movies?
# 
# 8-Find Top 20 actors with the most appearances in films
# 
# 
# 

# In[129]:


# importing built-in Function for my analysis
import numpy as np
import operator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[45]:


#import data to notebook
df = pd.read_csv('tmdb_movies.csv')
df.head(2)


# In[46]:


#data_info
df.info()


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# 
# ### General Properties

# In[47]:


#data contains 10866 sample films 

# First Cleaning Step is to check for rows duplicates.

# checking for dulpicates rows
sum(df.duplicated())
df.drop_duplicates(inplace=True)
df.info()# 1 item dropped Out


# In[48]:


#the data information shows that missing values are in columns
#['cast','homepage','director','tagline','keywords','overview','genres','production_companies']

Missing_val=['cast','homepage','director','tagline','keywords','overview','genres','production_companies']
Missing_val_res=[]
#counting the total number of rows with at least one missing data.any(axis = 1)
for i in Missing_val:
    Missing_val=df[i].isnull().sum()
    print(Missing_val)


# In[49]:


# Missing data are strings type, unique for each sample cannot be replaced by any statistical calculation 
#such as mean and mode values to preserve the data credibility and consistency.

# my first assumption is to delete columns with Missing data not included in my queries 
#['homepage','tagline','keywords','overview']

# my second assumption is to delete 'revenue','budget' columns as both of them are included adjusted by the dataset owner
#in another columns name 'budget_adj','revenue_adj'
df.drop(['homepage','tagline','keywords','overview','revenue','budget'], axis=1, inplace=True)


# In[50]:


# I eleminated 7 columns out of 21
df.shape


# In[51]:


# Second Cleaning Step is to check for columns with null values and calculates their quantities.
# Data contaings Null values in differeny columns 
# My strategy was to start by elimanting columns with high percentage of NULL values
# to maintain the data sample as large as possible.

# show only cloumns with NaN values for both production and cast columns

null_data = df[df['production_companies'].isnull() & df['cast'].isnull()]
null_data.info()


# In[52]:


# 3rd Assumption is to drop all rows where production company value is null. 
df=df.dropna(subset=['production_companies'])


# In[53]:


# dropn all rows where genres values are null. 
df=df.dropna(subset=['genres'])


# In[54]:


# dropn all rows where cast value are null. 
df=df.dropna(subset=['cast'])


# In[55]:


# dropn all rows where director values are null. 
df=df.dropna(subset=['director'])


# In[56]:


# dropn all rows where imdb_id values are null. 
df=df.dropna(subset=['imdb_id'])


# In[61]:


# adjust names for columns
df = df.rename(columns={'revenue_adj': 'revenue', 'budget_adj': 'budget'})


# In[62]:



df.info()


# In[63]:


df.head(2)


# In[64]:



df.shape


# In[65]:


#show column names
list(df.columns)


# In[66]:


# Last cleaning step is to check for zero values in data.(only for float ot integer dtype columns)
df.astype(bool).sum(axis=0)

# budget has 5021 zero values
# revenue has 4750 zero values


# In[105]:


# 4th assumption is to replace budget zero values by budget column mean value
#---------------------------------------------------------------------------------

mean_bud = df['budget'].mean()
# 5 th assumption is to replace revenue zero values by revenue column mean value
#----------------------------------------------------------------------------------
mean_rev = df['revenue'].mean()
# 6 th assumption is to replace runtime zero values by runtime column mean value
#----------------------------------------------------------------------------------
mean_rtime = df['runtime'].mean()
print(mean_bud,mean_rev,mean_rtime)
df['budget']=df['budget'].replace(0,mean_bud)
df['revenue']=df['revenue'].replace(0,mean_rev)
df['runtime']=df['runtime'].replace(0,mean_rtime)


# In[106]:


# check zero value for the dataframe
df.astype(bool).sum(axis=0)


# In[ ]:


# data is now clean, complete and we can start the investigation.
# Data finally consisit of:-
#9770 rows out of 10865 
#15 column out of 21


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# 
# ### Question # 1: Which year has the highest release of movies?

# In[107]:


# Use this, and more code cells, to explore your data. Don't forget to add
#   Markdown cells to document your observations and findings.
# 
df_high_rel=df.groupby(['release_year']).count().sort_values(['original_title'],ascending=False)['original_title']
# It can be seen from the results that year 2014 was the highest_release year.
df_high_rel.head(10)


# In[ ]:





# ###  Question # 2:
# ###  Top 10 movies which earn Lowest profit? AND TOP 10 movies which earn highest profit?

# In[108]:


# profit=revenue-budget
#we need to create a new column name profit
df['Profit']=df['revenue']-df['budget']
# sort by top 10
print('\n Top_10\n')
print('-'*30)
top_10= df.sort_values(['Profit'],ascending=False)
top_10=top_10[['original_title','Profit']]
print(top_10.head(10))
# sort by Lowest 10
print('\n Lowest_10\n')
print('-'*30)
Lowest_10= df.sort_values(['Profit'],ascending=True)
Lowest_10=Lowest_10[['original_title','Profit']]
#show results
print(Lowest_10.head(10))
#df.info()


# ###  Question # 3:

# ###  Movie with Highest And Lowest Budget? 

# In[109]:


HB= df.sort_values(['budget'],ascending=False)
HB=HB[['original_title','budget']]
LB= df.sort_values(['budget'],ascending=True)
LB=LB[['original_title','budget']]
print('\n LB \n',LB.head(1)),print('\n HB \n',HB.head(1))


# ###  Question # 4:

# ### Which movie made the highest revenue and lowest as well?

# In[111]:


HR= df.sort_values(['revenue'],ascending=False)
HR=HR[['original_title','revenue']]
LR= df.sort_values(['revenue'],ascending=True)
LR=LR[['original_title','revenue']]
print('\n LR \n',LR.head(1)),print('\n HR \n',HR.head(1))


# ###  Question # 5:

# ### Movie with shorest and longest runtime?

# In[112]:



Low_runtime= df.sort_values(['runtime'],ascending=False)
Low_runtime=Low_runtime[['original_title','runtime']]
High_runtime= df.sort_values(['runtime'],ascending=True)
High_runtime=High_runtime[['original_title','runtime']]
print('\n Low_runtime \n',Low_runtime.head(1)),print('\n High_runtime \n',High_runtime.head(1))


# ###  Question # 6:

# ###  Which movie get the highest or lowest votes (Ratings).

# In[113]:


#Which movie get the highest or lowest votes (Ratings).
Low_ratings= df.sort_values(['vote_count'],ascending=True)
Low_ratings=Low_ratings[['original_title','vote_count']]
High_ratings= df.sort_values(['vote_count'],ascending=False)
High_ratings=High_ratings[['original_title','vote_count']]
print('\n Low_ratings \n',Low_ratings.head(1)),print('\n High_ratings \n',High_ratings.head(1))


# ###  Visulaization_part1   
# 
# ### Question # 7:

# ###  which year has the highest release of movies?

# In[159]:


# Visulaization_part1
#Which year has the highest release of movies?
data=df.groupby('release_year').count()['id']

#grouping films according to their release year then counting the total number of movies in each year.
df.groupby('release_year').count()['id'].plot(xticks = np.arange(1960,2016,5))

#figure settings
sns.set(rc={'figure.figsize':(10,5)})
plt.title("Number Of Movies Vs Year",fontsize = 13)
plt.xlabel('Release year',fontsize = 11)
plt.ylabel('Num Of Movies',fontsize = 11)


# In[ ]:





# ###  Visulaization_part_2   
# 
# ### Question # 8:

# ### Find Top 20 actors with the most appearances in films

# In[160]:


# Function to define the xaxis label angel of rotation
def rot_xaxis_plot(i):
    for item in ax.get_xticklabels():
        item.set_rotation(i)
    return ()


# In[162]:




#Creating a numpy array to store how many times each actor casted for movies

actor_dict = {}

actors = df["cast"]
actors = actors.str.split("|")
actors = np.array(actors)
for actorList in actors:
    for actor in actorList:
        actor = actor.lstrip() 
        if actor not in actor_dict:
            actor_dict[actor] = 1
        else:
            actor_dict[actor] += 1
                
# sort actors in descending order
sorted_actor_dict = sorted(actor_dict.items(), key = operator.itemgetter(1), reverse = True)

x_axis = list()
y_axis = list()

for item in sorted_actor_dict[0:20]:
    x_axis.append(item[0])
    y_axis.append(item[1])

# Visualization settings
sns.set(rc={'figure.figsize':(12,10)}, font_scale=1.4)
ax = sns.barplot(x_axis, y_axis, palette="Set1")
ax.set(xlabel='Actor names', ylabel='Number of appearances', title = 'Top 20 actors based on the number of the appearances in movies')

rot_xaxis_plot(90)
    

plt.show()


# ###  Conclusions
# ##### We can conclude from the above analysis that year 2014 was the greatest year of film production was 2014 (from year 1960 upto 2015) .            - Top 3 movie with high profits: Starwar, Avatar & Titanic the lowest 3 films in terms of profits are the warrior way, the Lone Ranger and the Alamo.   -The film with the highest budget was the Warrior way and lowest budget was Fear Clinic.                                                                                                   - The film with the highest revenue was shattered glass and lowest revenue was Avatar.                                                                                                        - The longest film duration was Taken and the shortest one was Scrat's continental crack-up part-2.                                                                                    - From figure No 1 we can conclude that film production increased gradualy starting from 1960 until 2014 reached its peak then decayed in 2015.        - My last finding was the most top 3 actors appeared in movies during this period was Robert Dinero, samuel Jackson and Bruce wills and in fact they deserve being leading Actors.

# ###  Limitations
# ##### This analysis was done considering the movies produced between 1960 and 2015 with significant amount of profit of around 50 million dollar, which is not last updated data until 2021,results may differ accordingly. the dataset has some limitation regading the units of measurment of rutime and currency of the revenue, budget was not clearly stated in dataset and assumed to be US-Dollar and time measurment in minutes,  Dropping some rows with missing values and data imputation with mean values to complete the missing values in budget and revenue columns may offset the results. however, dataset was very rich in information. I beleive there much more work and investigation still need to be conducted to useful results.
# 

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Udacity_Project_data_analysis_rev.1.ipynb'])


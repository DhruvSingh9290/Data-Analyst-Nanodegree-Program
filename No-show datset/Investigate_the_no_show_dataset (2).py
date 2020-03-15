#!/usr/bin/env python
# coding: utf-8

# 
# # Project: Investigate the No-show dataset
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
# The No-show dataset: This dataset collects information from 100k medical appointments in Brazil and is focused on the question of whether or not patients show up for their appointment. A number of characteristics about the patient are included in each row.
# 
# I choosed for this project to analyse the No-show dataset. For me I found it important to investigate this dataset and to know closer why people don't attend this appointment which is an important one. I decided to find the factors that push people to get rid of this appointment. 

# ## Questions I will try to answer

# 1)Is there a gender correlation the No-Show ?
# 
# 2)Is there an age correlation to the No-Show factor ?
# 
# 3)Is there a schorlarship correlation to the No-Show factor ?
# 
# 4)Is there a time correlation ?
# 
# 5)Is there an health correlation ?
# 

# In[107]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
df=pd.read_csv('/Users/bert/Downloads/noshowappointments-kagglev2-may-2016.csv')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > **Tip**: In this section of the report, you will load in the data, check for cleanliness, and then trim and clean your dataset for analysis. Make sure that you document your steps carefully and justify your cleaning decisions.
# 
# ### General Properties

# In[4]:


df.head(3)


# In[179]:


df.shape


# In[6]:


df['Scholarship'].value_counts()


# In[5]:


df.dtypes


# In[10]:


df.shape


# In[14]:


df.hist(figsize=(8,8))


# 
# ### Data Cleaning

# In[108]:


df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])


# In[8]:


df.head(3)


# In[9]:


df.dtypes


# In[18]:


df.drop(df[(df.Age < 0) | (df.Age > 100)].index, inplace = True)


# In[23]:


df['No-show'].value_counts()


# For my fourth question I need the time of the appointment so i'm going to create new columns which are goind to help me.

# In[111]:


df['day']=df['ScheduledDay'].dt.day
df['hour']=df['ScheduledDay'].dt.hour
df['month']=df['ScheduledDay'].dt.month
df['hour'].head(2)


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 

# In[180]:


def plot(string):
    a = pd.crosstab(df[string], df['No-show'])
    a.div(a.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.ylabel('Count %')
    
def Diff_stats(string1, int1, int2):
    df_s0 = df[df[string1]==int1]
    df_s1 = df[df[string1]==int2]
    x=df_s0['No-show'].value_counts(normalize=True).Yes
    y=df_s1['No-show'].value_counts(normalize=True).Yes
    
    print('The pourcentage difference is: ',abs(x-y)*100,' %')
#will be useful for a lot of variables


# In[11]:


columns = ['Gender','Scholarship','Diabetes','Hipertension']
color  = ['r','g','y','b']
y=0

for i in columns:
    plt.subplot(int(str(22)+str((columns.index(i)+1))))
    df[i].value_counts(normalize=True).plot.bar(figsize= (9,9), color=color[y])
    plt.xlabel(columns[y])
    plt.ylabel('% count')
    plt.title(columns[y])
    y+=1


# -65% are female
# -85% up to 90% doesn't have a scholarship experience
# -<10% suffer from diabtes
# -<20% from hipertension

# In[14]:


df['Neighbourhood'].value_counts(normalize=True).sort_values(ascending = False)[:10].plot.bar(figsize=(24,6), fontsize = 15.0, color = 'g')
plt.title('Neighbourhood', fontsize=20)
plt.ylabel('% Count',fontsize=20)


# In[26]:


sns.distplot(df['Age'])


# In[23]:


df['Age'].plot.box(figsize=(8,8))


# ### Is there a gender correlation ?

# In[150]:


plot('Gender')


# There seems to be no correlation at all between the gender and the no Show. Around 80% of both genders didn't came to the appointment.

# ### Is there an Age correlation ?

# In[44]:


a = pd.crosstab(df['Age'], df['No-show'])
a.plot()


# We could make some histogrammes.

# In[61]:


df_N = df[df['No-show']=='Yes']
df_N.plot(kind='hist', y='Age')
plt.xlabel('Age')


# In[62]:


df_N = df[df['No-show']=='No']
df_N.plot(kind='hist', y='Age')
plt.xlabel('Age')


# Middle age people (around 50-60) and very young babies are more likely to not go to the appointment.
# 
# In the 'Yes' population they are more babies and young peoples. In the 'Yes' pop there is freq of 3000 and in the 'no' pop there is a freq of 14000. Babies are not brought to the appointment in the majoratity even its the more important part of the 'yes' pop.

# ### Is there a Scholarship correlation ?

# In[102]:


a = pd.crosstab(df['Scholarship'], df['No-show'])
a.plot.bar()


# In[181]:


plot('Scholarship')


# In[84]:


df_s1 = df[df['Scholarship']==1]
df_s0 = df[df['Scholarship']==0]
a = pd.crosstab(df_s1['Scholarship'], df['No-show'], normalize=True)
a.plot.bar(color=('r','g'))
plt.ylabel('Count %')

b = pd.crosstab(df_s0['Scholarship'], df['No-show'], normalize=True)
b.plot.bar(color=('r','g'))
plt.ylabel('Count %')


# In[89]:


df_s1['No-show'].value_counts(normalize=True)
#the precise values


# In[160]:


df_s0['No-show'].value_counts(normalize=True)


# In[146]:


print(0.237363-0.198057,'%')


# There a lot more people that didn't have a scholarship experience.
# 
# for those who didn't have a scholarship experience:
#  more than 80% didn't came and less than 20% came.
#  
# for those who have a scholarship experience:
#  around 75% didn't came and 25% came.
#  
# Scholarship as an influence on the relsult (on the No-show). A scholarship experience result in a higher chance of coming to the appointment.

# ### Time correlation

# Time is a good thing to study. I am not going to study month or days because I don't find those relevant in this dataset.

# In[139]:


#Here is the time histogramme for people who came
df_Y = df[df['No-show']=='Yes']
plt.hist(df_Y['hour'], color='g')


# In[147]:


#Here is the time histogramme for people who didn't came
df_N = df[df['No-show']=='No']
plt.hist(df_N['hour'],color='y')


# The two histogrammes are exactly the same exept for one hour slot: 6am. People didn't show up at 6am.
# So the time is a factor just for one hour slot. 6am is pretty early.

# ### Handicap, Diabetes and Hipertension correlation ?

# I decided to treat those variables together because I suppose It have the same impact on the No-show.

# In[182]:


plot('Handcap')


# In[176]:


Diff_stats('Handcap',0,4)


# In[183]:


plot('Diabetes')


# In[177]:


Diff_stats('Diabetes',0,1)


# In[184]:


plot('Hipertension')


# In[178]:


Diff_stats('Hipertension',0,1)


# So as expected Hipertension and Handicap evoluted together and react the same. If a person is impacted by hipertension or diabetes he has around 3% less chances of going to the appointment.
# 
# Otherwise If you are impacted by a Handcap you have more chances to go to the appointment. It can get up to 13% if your handcap is class 4 handcap.
# 
# So the 3 variables didn't react in the same way.

# <a id='conclusions'></a>
# ## What can we get out of it 
# 
# We did four little studies on the age, gender, scholarship and time (in hour).
# 
# - Gender :
# There are no correlation at all on the No-show. We can't expect from a gender to go more often to the appointment.
# 
# - Age:
# There is ! In fact middle age peoples (around 50) and babies are more likely to not show up ! 
# 
# - Scholarship:
# Again there is. People with a scholarship experience are 4% more likely to show up. Maybe people with a scholarship background can better understand the importance of this appointment.
# 
# - Hour:
# A little correlation: at 6am a lot more appointment are cancelled. When It is too early peoples don't want to go to the appointment.
# 
# - Health:
# Some desease can push you to not go to your appointment by around 3% like the diabete and the hipertension. But f you suffer from handcap you are way more likely to go to the appointment, this go up with the level of your handcap (max 13% for a level 4).
# So we may conclude on the one hand that heavier illness (when your life is in danger) push people to go to the appointment and on the over hand that people with little desease (no life-threatening danger) are maybe more tired and don't want to go to the appointment: maybe they don't need it as much as others.

# We can conclude that many factors are related to the fact that people don't show up but I expected it to be more relevant. There are a lot of factors wich have a little impact. As far as those factors didn't have so much importance we should have had more criterias.

# # Limitations

# What are the limits of this dataset ?
# 
# - The dimension of the sample. We may lack some important informations, I don't think that peoples decisions can be reduced to 10 variables. We lack some variables but the purpose of this exploration was just to have a vision of the major variables.
# For exemple the wealth columns would have been appreciated. This one could have been correlate with the scholarship experience and the presence to appointment.
# 
# - Not deep enough. To completly understand the dataset we should have looked into the correlations between each variables not just the 'No-show' and one variable at a time. This is caused by a lack of statistical background from me.
# 
# - The shape of the dataset. Even if the shape is around 100000 rows it can't represent the all population. I don't know if this sample is from anyone or if It was produced by a minority of the population.

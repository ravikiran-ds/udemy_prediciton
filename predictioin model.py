# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 09:35:33 2020

@author: Ravi Kiran
"""

import pandas as pd
import matplotlib.pyplot as plt
import codecs
import seaborn as sns

doc = codecs.open('C:/Users/HP/Documents/ML/Projects/ratings prediction Udemy/data.txt','rU','UTF-16')

#importing tab separated data
df=pd.read_csv("C:/Users/HP/Documents/ML/Projects/ratings prediction Udemy/data.txt",sep="\t")
#taking only the required data
#THE ACTUAL DATA CONTAINS 7 FEATURES AND 320 ROWS OF DATA
df=df.iloc[:320,0:7]

#looking at the columns
df.columns

#cleaning the cost feature by removing the rupee symbol and comma
df["Cost"]=df["Cost"].apply(lambda x: x.split("₹")[1])
df["Cost"]=df["Cost"].apply(lambda x: x.replace(",",''))
#converting to int datatype
df["Cost"]=df["Cost"].astype('int')

#cleaning the duration feature
#introducing new features to work with
df["Duration in hr"]=df["Duration"].apply(lambda x: 1 if " total hours"  in x.lower() else 0)
df["Duration in hr"]=df["Duration"].apply(lambda x: 1 if " total hour"  in x.lower() else 0)
df["Duration"]=df.Duration.apply(lambda x:x.replace(" total hours",""))
df["Duration"]=df.Duration.apply(lambda x:x.replace(" total mins",""))
df["Duration"]=df.Duration.apply(lambda x:x.replace(" total hour",""))
#converting to float datatype
df["Duration"]=df["Duration"].astype('float')
#transforming all the variable to minutes
df["Duration in min"]=df["Duration"]
df.loc[df["Duration in hr"]==1,"Duration in min"]=df["Duration in min"]*60 

#cleaning the lectures feature
df["Lectures"]=df.Lectures.apply(lambda x: x.replace(" lectures",''))
#converting to int type
df.Lectures=df.Lectures.astype("int")

#cleaning the student feature
df.Students=df.Students.apply(lambda x:x.replace("(","").replace(",","").replace(")",""))
#converting to in type
df.Students=df.Students.astype("int")

#to see how much revenue the course might have made
df["Revenue"]=df["Students"]*df["Cost"]
#highest earned course is  
rev_high=max(df["Revenue"])
title=df.loc[(df["Revenue"]==max(df["Revenue"])),"Title"].values
print("The most profitable course is {} with ₹{} revenue".format(title[0],rev_high))
#least earning course is
rev_low=min(df["Revenue"])
title=df.loc[(df["Revenue"]==min(df["Revenue"])),"Title"].values
print("The most profitable course is {} with ₹{} revenue".format(title[0],rev_low))

#average revenue is
print("average revenue is {}".format(df["Revenue"].mean()))

#lookina at the datatypes
df.dtypes

#looking for missing values 
df.isnull().sum()

#seeing the different levels
df["Levels"].unique()

df.columns

#seeing the distribution of ratings
min(df["Ratings"])
max(df["Ratings"])
df["Ratings"].hist()
plt.xlabel("Ratings")
plt.ylabel("Frequency")
plt.title("Rating distribution")
plt.show()

#distribution of cost
#most of the courses cost the same
df["Cost"].plot.hist()
plt.xlabel("Cost in Rupee")
plt.ylabel("Frequency")
plt.title("Cost distribution")
plt.show()

#disttribution of duration
#most courses lie between 500 minutes to 2500 minutes ie 8 hrs - 41 hrs
df["Duration in min"].hist()
plt.xlabel("Duration in minutes")
plt.ylabel("Frequency")
plt.title("Duration distribution")
plt.show()

#relationship between student and ratings
min(df.Students)
max(df.Students)
plt.scatter(x=df.Students,y=df.Ratings,color="darkblue")
plt.xlim(100,150000)
plt.show()

#heatmap for correlation 
#THERE IS NO STRONG RELATIONSHIP WITH DEP VARIABLE
sns.heatmap(df.corr(),annot=True, fmt='0.1f')

#shows that approximately 90% courses are rated between 4.0 - 4.8
len(df.loc[(df["Ratings"]>4.0) & (df["Ratings"]<4.8),"Ratings"])/len(df["Ratings"])

#gettig dummies for the levels
#since beginner is less than intermediate and, intermediate is less than expert 
#we will only use LabelEncoder
from sklearn.preprocessing import LabelEncoder
lab_df=LabelEncoder()
df["Levels"]=lab_df.fit_transform(df["Levels"])
#seing the labels
list(lab_df.classes_)
#encoding the ratings
#creating bins to convert continous data to categorical data
df["Ratings_Cat"]=pd.cut(df.Ratings,bins=[0,3.5,4,4.5,5],labels=['Bad','Okay','Good','Excellent'])
rat_enc=LabelEncoder()
df.Ratings_Cat=rat_enc.fit_transform(df.Ratings_Cat)
list(rat_enc.classes_)


#splitting dependent and inndependent variables
x=df.iloc[:,[1,2,4,5,8]].values
y=df.iloc[:,-1].values

#lets say that ratings are classification of a course that it is good or bad

#lets scale the df's
from sklearn.preprocessing import StandardScaler
x_scl=StandardScaler()
x=x_scl.fit_transform(x)

#splitting the date to training and testing sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=101)

#looking at the graph for rationship between student and ratings we see that it is a normal graph
#i will try and implement the svm with rbf kernel
from sklearn.svm import SVC
sv_cls=SVC(kernel='rbf',random_state=101)
sv_cls.fit(x_train,y_train)

#predicting the values for test set
y_pred=sv_cls.predict(x_test)

#comparing the results
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

#SINCE THERE IS NO STRONG REPLATIONSHIP BETWEEN THE DEPENDENT AND INDEPENDENT VARIABLE
#THE MODEL IS NOT ABLE TO CLASSIFY CORRECTLY
#NEED MORE DATA

#the graph is very non uniform
plt.scatter(x=df.Students,y=df.Ratings_Cat)
plt.plot(df.Students,sv_cls.predict(x_scl.transform(df.iloc[:,[1,2,4,5,8]])))
plt.show()













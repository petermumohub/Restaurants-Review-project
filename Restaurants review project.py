#Importing necessary Libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#setting up pandas and reading the dataset
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', 18)
pd.set_option('display.max.columns',8) 
df=pd.read_csv("C:\\Users\ADMIN\\Downloads\\Restaurant-Reviews.csv")
print(df.head())

#Data cleaning
df.drop(columns=['7514','Review'], inplace=True)
df.duplicated().sum() 
df.drop_duplicates(inplace=True)
df.isnull().sum() 
df.dropna(inplace=True)
df["Rating"].value_counts() 
df["Rating"].replace({"Like":4})
df['Rating']=df["Rating"].replace({"Like":"4"}) 
df["Rating"]=df["Rating"].astype(float)
df["Time"]=pd.to_datetime(df["Time"]) 
#Extracting data from Time column
df["Day"]=df["Time"].dt.day
df["Month"]=df["Time"].dt.month
df["Year"]=df["Time"].dt.year
df["Hour"]=df["Time"].dt.hour

#looping through hour column and categorising time into early morning,morning,afternoon,evening and late night 

def time_of_the_day(hour):
    category=None
    if hour >= 1 and hour <= 4:
        category= "Early Morning"
    elif hour >= 5 and hour <= 12:
        category= "Morning"
    elif hour >= 13 and hour <= 15:
        category= "Afternoon"
    elif hour >= 16 and hour <= 21:
        category= "Evening"
    else:
        category= "Late Night"

    return category    


df['time_of_day']=df['Hour'].apply(time_of_the_day)

#creating a column that clasifies a rating of 3 and above as positive otherwise negative
def sentiment(rating):
    if rating>=3:
        return "positive"
    else:
        return "Negative"
    
df["sentiment"]=df["Rating"].apply(sentiment)
df.drop(["Hour","Metadata"], axis=1,inplace=True) #Dropping hour column 
print(df)

#EXPLORATORY DATA ANALYSIS
df.info()
df.describe()
#checking for outliers in the rating column
sns.boxplot(x="Rating",data=df)
plt.show()

#Top 1o restaurants with the highest avarage rating
Top_10_restaurants_rating=df.groupby('Restaurant')['Rating'].mean().reset_index()
Top_10_restaurants_rating=Top_10_restaurants_rating.sort_values('Rating',ascending=False)
Top_ten_restaurants=Top_10_restaurants_rating.head(10)
print(Top_ten_restaurants)
sns.barplot(y="Restaurant",x="Rating",data=Top_ten_restaurants)
plt.ylabel("Restaurant")
plt.xlabel("Rating")
plt.title("Top_ten_restaurants")
plt.show()

#Relationship between time of day and the rating
Average_time_of_day_ratings=df.groupby("time_of_day")["Rating"].mean().reset_index()
print(Average_time_of_day_ratings)
sns.barplot(x="time_of_day",y="Rating",data=Average_time_of_day_ratings)
plt.xlabel("time_of_day")
plt.ylabel("Rating")
plt.title("Average_time_of_day_ratings")
plt.show()

#Ratings over diffrent years
Av_restaurants_ratings_yearly=df.groupby("Year")["Rating"].mean().reset_index()
Av_restaurants_ratings_yearly=Av_restaurants_ratings_yearly.sort_values("Rating",ascending=False)
print(Av_restaurants_ratings_yearly)
sns.barplot(x="Year",y="Rating",data=Av_restaurants_ratings_yearly)
plt.xlabel("Year")
plt.ylabel("Rating")
plt.title("Average rating per year")
plt.show()

#Monthly average ratings
Monthly_average_rating=df.groupby("Month")["Rating"].mean().reset_index()
Monthly_average_rating=Monthly_average_rating.sort_values("Rating",ascending=False)
print(Monthly_average_rating)
sns.barplot(x="Month",y="Rating",data=Monthly_average_rating)
plt.xlabel("Month")
plt.ylabel("Rating")
plt.title("Average rating per month")
plt.show()

#Comparing between total positive and negative comments
sns.countplot(x="sentiment",data=df)
plt.show()

#hypthesis testing
''''We want to look if there is a sigficant diffrence in ratings across months
#H0=No significant difference in the average ratings across months
#H1=There is significant difference in average ratings across months'''
import statsmodels.api as sm
from statsmodels.formula.api import ols
a=ols('Rating ~ Month',data=df).fit()
b=sm.stats.anova_lm(a,type=2)
b
#From the outcome, the p value(0.159) is greater than .05 therefore we fail to reject the null hypothesis 

#MODEL BUILDING
#Transforming sentimnts columns into numbers for easy modelling
def sentiment(sentiment):
    if sentiment=="positive":
        return 1
    else:
        return 0
df["sentimented"]=df["sentiment"].apply(sentiment)

y=df["sentimented"]
x=df["Rating"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=4)
#Reshapping X_train data into a 2D numpy array
X_train_reshaped=np.array(X_train).reshape(-1, 1)
#Reshapping X_test data into a 2D numpy array
X_test_reshaped=np.array(X_test).reshape(-1,1)
#Model training
model=LogisticRegression()
model.fit(X_train_reshaped, y_train)
#Model prediction
model.predict(X_test_reshaped)
#Model accuracy
accuracy = model.score(X_train_reshaped,y_train)
print(f"The accuracy of the model is, ", accuracy ) 



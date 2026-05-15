import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Phase 1: The Detective Work (Setup & Inspection)
df = pd.read_csv('customer_analytics (1).csv')
print(df.head())  #First 5 rows
print(df.info())  #Dataset info
df.shape
print(df.describe())  #Dataset basic statistics
print("The dataset represents 205 customer data with features such as age, annual income, purchase amount, prefered devices, etc")


#Phase 2: The Cleanup (Data Preprocessing)
print(df.isna().sum())   #no. of missing values in each column
print("Number of duplicate rows: ",df.duplicated().sum())
df=df.drop_duplicates()
columns_dropped = ['Education']
df = df.drop('Education',axis=1)
print(df.isna().sum())     #education column dropped
print(df['AnnualIncome'].mean())
df['AnnualIncome'] = df['AnnualIncome'].fillna(df['AnnualIncome'].mean())
df.shape
print("Education and annual income columns contained missing values.\nEducation column was dropped as it did not have much significance in our analysis.\nAnnual income missing values where filled using their mean, ie., 74499.90")
df.to_csv('cleaned_customer_analytics.csv',index=False)  #saving cleaned dataset


#Phase 3: The Deep Dive (Univariate & Bivariate Analysis)
sns.histplot(x=df['Age'])
plt.title("Histogram of customer age")
plt.show() 

sns.histplot(x=df['AnnualIncome'],kde=True)
plt.title("Histogram of customers' annual income\nThe graph shows us that our majority customers earns upto 100000")
plt.show()

sns.countplot(x='PreferredDevice',data=df)
plt.title("The bar plot shows that tablet is the most popular device bought")
plt.show()


sns.scatterplot(x="Age",y="YearsEmployed",data=df)
plt.title("Scatterplot of Age vs YearsEmployed.\nStrong positive correaltion between age and years employed")
plt.xlabel("Age")
plt.ylabel("Years Employed")
plt.show()

sns.boxplot(x=df["MaritalStatus"],y=df["LastPurchaseAmount"])
plt.title("Marital Status vs Purchase Amount boxplot.\nSingles purchase amount is higher on average than married, and no outliers")
plt.show()


corr=df.corr(numeric_only=True)
sns.heatmap(corr, annot=True,cmap='Blues')
plt.title("Heatmap")
plt.tight_layout()

#Phase 4: The Big Picture (Multivariate & Storytelling)
print("Executive Summary:")
print("1. Most bought device is tablet")
print("2. Most of our customers have an annual income of upto 100000")
print("3. Most features provided do not have high correlation at all, except age and years employed, having a strong positive correlation 0f 0.97")
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df = pd.read_csv(url)
print(df.dtypes)
df.drop("id" , axis = 1 , inplace = True)
df.drop("Unnamed: 0" , axis = 1 , inplace = True)
print(df.describe())
Missing_value = df.isnull()
for column in Missing_value.columns.values.tolist():
    print (column)
    print(Missing_value[column].value_counts())
    print("")
bed_Avg=df['bedrooms'].mean(axis=0)
df['bedrooms'].replace(np.nan , bed_Avg , inplace  =  True)
bath_Avg=df['bathrooms'].mean(axis=0)
df['bathrooms'].replace(np.nan , bath_Avg ,  inplace  =  True)
Missing_value = df.isnull()
for column in Missing_value.columns.values.tolist():
    print (column)
    print(Missing_value[column].value_counts())
    print("")
Floor_Counts = df['floors'].value_counts()
Floor_Counts_df = Floor_Counts.to_frame()
Floor_Counts_df.columns = ['number_of_houses']
print(Floor_Counts_df)
plt.figure(figsize = (10 , 6))
sns.boxplot(x='waterfront' , y='price' , data=df)
plt.title('Houses with a waterfront view or without a waterfront view ')
plt.xlabel('Waterfront view (0 = NO , 1=Yes)')
plt.ylabel('Price')
plt.show()
sns.regplot(x='sqft_above' , y='price' , data = df)
plt.title('Feature sqft_above with price')
plt.xlabel('sqft_above')
plt.ylabel('Price')
X=df[['sqft_living']]
Y=df[['price']]
lm=LinearRegression()
lm.fit(X,Y)
Yhat = lm.predict(X)
R_squ= lm.score(X,Y)
print("The predict value of the price is :" ,Yhat)
print("The R Square is:" ,R_squ )
lm1=LinearRegression()
features =df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]]
lm1.fit(features , Y)
Y_hat = lm1.predict(features)
Rsqu= lm1.score(features,Y)
print("The predict value of the price is :" ,Y_hat)
print("The R Square is:" ,Rsqu )
Input = [('Scale' , StandardScaler()) , ('Polynomial' , PolynomialFeatures(include_bias = False)) , ('Model' , LinearRegression())]
pip = Pipeline(Input)
features=features.astype('float')
pip.fit(features , Y)
Ypip =pip.predict(features)
R_squ_pip= r2_score(Y ,Ypip)
print("The R Square is:" ,R_squ_pip )
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)
print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(x_train, y_train)
y_pred = ridge_model.predict(x_test)
R_squ_ridge= r2_score(y_test , y_pred)
print("The R Square is:" ,R_squ_ridge )
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
ridge_model_poly = Ridge(alpha=0.1)
ridge_model_poly.fit(x_train_poly, y_train)
y_pred_poly = ridge_model_poly.predict(x_test_poly)
R_squ_ridge_poly = r2_score(y_test, y_pred_poly)
print("R² for Ridge regression with polynomial features:", R_squ_ridge_poly)




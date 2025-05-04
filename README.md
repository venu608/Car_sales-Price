# Car_sales-Price


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.preprocessing import LabelEncoder


     

df = pd.read_csv(r"F:\projects\SQL Car Price Data Set\car_price_dataset.csv")
             


     

df.shape


     
(10000, 10)

df


     
Brand	Model	Year	Engine_Size	Fuel_Type	Transmission	Mileage	Doors	Owner_Count	Price
0	Kia	Rio	2020	4.2	Diesel	Manual	289944	3	5	8501
1	Chevrolet	Malibu	2012	2.0	Hybrid	Automatic	5356	2	3	12092
2	Mercedes	GLA	2020	4.2	Diesel	Automatic	231440	4	2	11171
3	Audi	Q5	2023	2.0	Electric	Manual	160971	2	1	11780
4	Volkswagen	Golf	2003	2.6	Hybrid	Semi-Automatic	286618	3	3	2867
...	...	...	...	...	...	...	...	...	...	...
9995	Kia	Optima	2004	3.7	Diesel	Semi-Automatic	5794	2	4	8884
9996	Chevrolet	Impala	2002	1.4	Electric	Automatic	168000	2	1	6240
9997	BMW	3 Series	2010	3.0	Petrol	Automatic	86664	5	1	9866
9998	Ford	Explorer	2002	1.4	Hybrid	Automatic	225772	4	1	4084
9999	Volkswagen	Tiguan	2001	2.1	Diesel	Manual	157882	3	3	3342
10000 rows × 10 columns


#  top 10  in data set 
df.head(10)


     
Brand	Model	Year	Engine_Size	Fuel_Type	Transmission	Mileage	Doors	Owner_Count	Price
0	Kia	Rio	2020	4.2	Diesel	Manual	289944	3	5	8501
1	Chevrolet	Malibu	2012	2.0	Hybrid	Automatic	5356	2	3	12092
2	Mercedes	GLA	2020	4.2	Diesel	Automatic	231440	4	2	11171
3	Audi	Q5	2023	2.0	Electric	Manual	160971	2	1	11780
4	Volkswagen	Golf	2003	2.6	Hybrid	Semi-Automatic	286618	3	3	2867
5	Toyota	Camry	2007	2.7	Petrol	Automatic	157889	4	4	7242
6	Honda	Civic	2010	3.4	Electric	Automatic	139584	3	1	11208
7	Kia	Sportage	2001	4.7	Electric	Semi-Automatic	157495	2	2	7950
8	Kia	Sportage	2014	2.6	Hybrid	Manual	98700	3	4	9926
9	Toyota	RAV4	2005	3.1	Petrol	Manual	107724	2	5	6545

#  finding the last rows in data set 
df.tail(10)


     
Brand	Model	Year	Engine_Size	Fuel_Type	Transmission	Mileage	Doors	Owner_Count	Price
9990	Audi	A3	2019	1.8	Electric	Manual	85496	4	3	11890
9991	Mercedes	E-Class	2017	1.1	Diesel	Automatic	179286	5	1	8214
9992	BMW	5 Series	2016	1.2	Hybrid	Automatic	13386	3	5	12332
9993	Honda	Accord	2019	4.7	Hybrid	Semi-Automatic	155874	4	3	12382
9994	Honda	Civic	2016	2.9	Petrol	Manual	255889	2	2	6682
9995	Kia	Optima	2004	3.7	Diesel	Semi-Automatic	5794	2	4	8884
9996	Chevrolet	Impala	2002	1.4	Electric	Automatic	168000	2	1	6240
9997	BMW	3 Series	2010	3.0	Petrol	Automatic	86664	5	1	9866
9998	Ford	Explorer	2002	1.4	Hybrid	Automatic	225772	4	1	4084
9999	Volkswagen	Tiguan	2001	2.1	Diesel	Manual	157882	3	3	3342

#finding the Null values in that data set
df.info()


     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 10 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Brand         10000 non-null  object 
 1   Model         10000 non-null  object 
 2   Year          10000 non-null  int64  
 3   Engine_Size   10000 non-null  float64
 4   Fuel_Type     10000 non-null  object 
 5   Transmission  10000 non-null  object 
 6   Mileage       10000 non-null  int64  
 7   Doors         10000 non-null  int64  
 8   Owner_Count   10000 non-null  int64  
 9   Price         10000 non-null  int64  
dtypes: float64(1), int64(5), object(4)
memory usage: 781.4+ KB

# Satatical finding the min,max,count percentails 
df.describe()


     
Year	Engine_Size	Mileage	Doors	Owner_Count	Price
count	10000.000000	10000.000000	10000.000000	10000.000000	10000.000000	10000.00000
mean	2011.543700	3.000560	149239.111800	3.497100	2.991100	8852.96440
std	6.897699	1.149324	86322.348957	1.110097	1.422682	3112.59681
min	2000.000000	1.000000	25.000000	2.000000	1.000000	2000.00000
25%	2006.000000	2.000000	74649.250000	3.000000	2.000000	6646.00000
50%	2012.000000	3.000000	149587.000000	3.000000	3.000000	8858.50000
75%	2017.000000	4.000000	223577.500000	4.000000	4.000000	11086.50000
max	2023.000000	5.000000	299947.000000	5.000000	5.000000	18301.00000

# Finding the Null Values
df.isna().sum()


     
Brand           0
Model           0
Year            0
Engine_Size     0
Fuel_Type       0
Transmission    0
Mileage         0
Doors           0
Owner_Count     0
Price           0
dtype: int64
Exploratory Data Analysis (EDA)

df.columns


     
Index(['Brand', 'Model', 'Engine_Size', 'Fuel_Type', 'Transmission', 'Mileage',
       'Doors', 'Owner_Count', 'Price', 'Age'],
      dtype='object')

cat_cols = ['Brand', 'Fuel_Type', 'Transmission', 'Doors']
i = 0

while i < len(cat_cols):  # Avoid hardcoding length
    fig = plt.figure(figsize=[12, 5])

    # First subplot
    plt.subplot(1, 2, 1)
    sns.countplot(x=cat_cols[i], data=df)  # Corrected 'df' parameter
    plt.title(f"Distribution of {cat_cols[i]}")

    i += 1  # Increment index

    # Ensure index does not exceed list length
    if i < len(cat_cols):
        # Second subplot
        plt.subplot(1, 2, 2)
        sns.countplot(x=cat_cols[i], data=df)  # Corrected 'df' parameter
        plt.title(f"Distribution of {cat_cols[i]}")

        i += 1  # Increment index again

    plt.show()


     



num_cols = ['Price','Mileage','Engine_Size','Age']
i=0
while i < 4:
    fig = plt.figure(figsize=[13,3])
    #ax1 = fig.add_subplot(121)
    #ax2 = fig.add_subplot(122)
    
    #ax1.title.set_text(num_cols[i])
    plt.subplot(1,2,1)
    sns.boxplot(x=num_cols[i], data=df)
    i += 1
    
    #ax2.title.set_text(num_cols[i])
    plt.subplot(1,2,2)
    sns.boxplot(x=num_cols[i], data=df)
    i += 1
    
    plt.show()


     



import seaborn as sns
import matplotlib.pyplot as plt

# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Compute correlation and plot heatmap
sns.heatmap(numeric_df.corr(), annot=True, cmap="RdBu")
plt.show()


     


df['Price'] = pd.to_numeric(df['Price'], errors='coerce')  # Convert, setting errors to NaN if needed


     

y = df['Price']
X = df.drop('Price',axis=1)


     

from sklearn.model_selection import train_test_split


     

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("x train: ",X_train.shape)
print("x test: ",X_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)


     
x train:  (8000, 9)
x test:  (2000, 9)
y train:  (8000,)
y test:  (2000,)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
car_pred_model(lr,"Linear_regressor.pkl")


     
Train R2-score : 1.0
Test R2-score : 1.0
Train CV scores : [0.99954502 0.99974825 0.99870681 0.99896243 0.99917219]
Train CV mean : 1.0
C:\Users\Dell\AppData\Local\Temp\ipykernel_19660\2418508315.py:34: UserWarning: 

`distplot` is a deprecated function and will be removed in seaborn v0.14.0.

Please adapt your code to use either `displot` (a figure-level function with
similar flexibility) or `kdeplot` (an axes-level function for kernel density plots).

For a guide to updating your code to use the new functions, please see
https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751

  sns.distplot((y_train-y_pred_train),hist = False,ax = ax[0])


 


     

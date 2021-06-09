# Importuojame bibliotekas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from matplotlib import colors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from Tools.scripts.dutree import display
from sklearn import linear_model
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

# Ar šventės turi įtakos pardavimams? Kaip pagerinti pardavimus parduotuvėse, kur jie yra mažiausi?
# Šventės turi įtakos pardavimams. Vasarį jie padidėja dėl Super Bowl, rugpjūtį prieš rugsėjį vykstančią Labor Day ir
# nuo padėkos, kuri yra laprkitį kyla pardavimai iki Kalėdų. Pastebėta, kad 12 departamentų yra prasčiausi pardavimai.
# Reikia analizuoti toliau, kas daro tam įtaką. 

# Duomenys

features = pd.read_csv('features.csv')
sales = pd.read_csv('sales.csv')
stores = pd.read_csv('stores.csv')

#Super Bowl : 12-Feb-10, 11-Feb-11, 10-Feb-12, 8-Feb-13
#Labor Day : 10-Sep-10, 9-Sep-11, 7-Sep-12, 6-Sep-13
#Thanksgiving : 26-Nov-10, 25-Nov-11, 23-Nov-12, 29-Nov-13
#Christmas : 31-Dec-10, 30-Dec-11, 28-Dec-12, 27-Dec-13

features.head()
stores.head()
sales.head()

sales.info()
stores.describe()
features.info()
#
features['Date'] = pd.to_datetime(features['Date'])
sales['Date'] = pd.to_datetime(sales['Date'])
#
features['IsHoliday'] = LabelEncoder().fit_transform(features['IsHoliday'])
sales['IsHoliday'] = LabelEncoder().fit_transform(sales['IsHoliday'])
#
su_features_sales = pd.merge(features,sales, how='inner',on=['Store', 'Date', 'IsHoliday'])
su_galutinis =pd.merge(su_features_sales,stores,how='inner')
print("Po sujungimo : ",su_galutinis.shape[0])

pd.set_option("display.max.columns", None)
print(su_galutinis.describe())

print(su_galutinis.isnull().sum())
msno.bar(su_galutinis)
plt.show()

print('Dublikatu:', su_galutinis[su_galutinis.duplicated()].shape[0])

print(su_galutinis.isnull().mean()*100)

# daugiau nei 64 proc. Null reikšmių.


pd.DataFrame(su_galutinis.dtypes, columns=['Type'])

labels = stores.Type.value_counts().index.tolist()
sizes = stores.Type.value_counts().values.tolist()
explode = (0.02, 0.02, 0)
plt.figure(figsize=(5,5))
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=45,
        textprops={'fontsize': 18},colors=['#9acd32','#ff7f50','#fac205'])
plt.title('Parduotuvių tipai')
plt.show()

# A tipo parduotuvių sudaro beveik 50 proc.

ax = sns.countplot(stores.Type ,facecolor=(0,0,0,0),linewidth=10,
                   edgecolor=sns.color_palette("crest", 3))
for p in ax.patches:
    ax.annotate(f'Parduotuvių skaičius:\n {p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()-4),
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points',fontsize=10)
plt.show()

# A -22, B-17,C- 6

for df in [su_galutinis]:
    df['Week'] = df['Date'].dt.week
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
plt.figure(figsize=(15,3))
su_galutinis[su_galutinis['Year']==2010].groupby('Month').mean()['Weekly_Sales'].plot()
su_galutinis[su_galutinis['Year']==2011].groupby('Month').mean()['Weekly_Sales'].plot()
su_galutinis[su_galutinis['Year']==2012].groupby('Month').mean()['Weekly_Sales'].plot()
plt.title('Vidutiniai savaitiniai pardavimai kiekvienais metais', fontsize=18)
plt.grid()
plt.legend(['2010', '2011', '2012'], loc='best', fontsize=16)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Months', fontsize=16)
plt.show()
#
plt.figure(figsize=(15,3))
su_galutinis[su_galutinis['Type']=='A'].groupby('Month').mean()['Weekly_Sales'].plot()
su_galutinis[su_galutinis['Type']=='B'].groupby('Month').mean()['Weekly_Sales'].plot()
su_galutinis[su_galutinis['Type']=='C'].groupby('Month').mean()['Weekly_Sales'].plot()
plt.title('Vidutiniai pardavimai pagal parduotuvės tipą', fontsize=18)
plt.grid()
plt.legend(['Type A', 'Type B', 'Type C'], loc='best', fontsize=16)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Months', fontsize=16);
plt.show()

# A  tipo parduotuvės parduoda daugiausiai, bet tam turi įtakos, kad A tipo parduotuvių yra daugiausiai.
# C tipo pardavimai yra stabilūs.
# Taip pat pardavavimai pradeda didėti nuo spalio mėnesio. Tai galima susisieti su artėjančiomis Padėkos diena ir Kalėdų atostogomis.


def scatter(dataset, column):
    plt.figure()
    plt.scatter(dataset[column], dataset['Weekly_Sales'])
    plt.ylabel('Weekly_Sales')
    plt.xlabel(column)
su_galutinis.columns

scatter(su_galutinis, 'Fuel_Price')
plt.show()

# Iš gautos diagramos nėra pastebėta, kad kuro kaina darytų didžiulę įtaką savaitiniams pardavimas. Nuo 4.00 jie mažėja.

scatter(su_galutinis, 'Unemployment')
plt.show()

# Daugiausiai pardavimų būna kuomet nedarbingumo lygis yra nuo 6 iki 9.

scatter(su_galutinis, 'Temperature')
plt.show()

# Nuo 10 pardavimai kyla ir ties 60 pradeda žemėti.

scatter(su_galutinis, 'Store')
plt.show()

#Nepastebima jokia tendencija.

weekly_sales = su_galutinis['Weekly_Sales'].groupby(su_galutinis['Month']).mean()
plt.figure(figsize=(25,8))
sns.barplot(weekly_sales.index, weekly_sales.values, palette='dark')
plt.grid()
plt.title('Average Sales - per Dept', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Month', fontsize=16)
plt.show()


def id(str):
    plt.figure(figsize=(20, 5))
    su_galutinis.groupby(str).mean()['Weekly_Sales'].sort_values().plot(kind='bar', color='#dda0dd')
    plt.title('Kiekvienos parduotuvės vidutiniai pardavimai.', fontsize=18)
    plt.ylabel('Sales', fontsize=16)
    plt.xlabel(str, fontsize=16)
    plt.tick_params(axis='x', labelsize=14)

id('Store')
plt.show()

id('Dept')
plt.show()

# Apie 12 departamentų turi mažiausius pardavimus, reikia investiguoti kaip būtų galima pagerinti jų pardavimus.

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.stripplot(y=su_galutinis['Weekly_Sales'],x=su_galutinis['Type'])
plt.show()

# Didžiausi pardavimai užfiksuoti B tipo parduotuvėse.

print(su_galutinis.corr().to_string())
sns.heatmap(su_galutinis.corr(), annot=True, cmap="cubehelix")
plt.show()

# LINIJINE REGRESIJA
X = su_galutinis.Fuel_Price.values.reshape(-1, 1)
y = su_galutinis.Weekly_Sales.values.reshape(-1, 1)
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.20)
lm = linear_model.LinearRegression()
lm.fit(xTrain, yTrain)
yhat=lm.predict(X)
plt.scatter(X,y, c='grey')
plt.plot(X, yhat)
print(lm.coef_)
print(lm.intercept_)

print(lm.score(xTest, yTest))
plt.show()

# Stebint linijinę regresiją su Temperatūros, Kuro kainom ir nedarbingumu matosi, kad nėra tiesioginio ryšio su pardavimais.
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def getmar(n,X,y,feat):
    melmodel=RandomForestRegressor(n_estimators=n,random_state=1)
    melmodel.fit(X,y)
    test_data=pd.read_csv("test.csv")
    test_X=test_data[feat]
    mar=mean_squared_error(y,melmodel.predict(test_X))
    return(mar)

data=pd.read_csv("train.csv")
print(data.describe())
print(data.columns)
y = data.SalePrice
feat = ['MSSubClass','LotArea', 'YearBuilt','OverallQual','OverallCond',
    '1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','FullBath','HalfBath',
    'BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces',
    'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch',
    'ScreenPorch','PoolArea','YrSold']
X=data[feat]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

v=0
mi=getmar(50,X,y,feat)
s=0
for i in range(1,1000):
    print(i)
    v=getmar(i,X,y,feat)
    if v<=mi:
        mi=v
        s=i
print(s)
print(mi)

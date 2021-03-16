import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def getmar(n,train_X,val_X,train_y,val_y):
    melmodel=RandomForestRegressor(n_estimators=n,random_state=1)
    melmodel.fit(train_X,train_y)
    mar=mean_squared_error(val_y,melmodel.predict(val_X))
    return(mar)

data=pd.read_csv("D:/Python/train.csv")
print(data.describe())
print(data.columns)
y = data.SalePrice
feat = ['LotArea', 'YearBuilt']
X=data[feat]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
v=0
mi=getmar(50,train_X,val_X,train_y,val_y)
s=0
for i in range(1,1000):
    v=getmar(i,train_X,val_X,train_y,val_y)
    if v<=mi:
        mi=v
        s=i
print(s)
print(mi)

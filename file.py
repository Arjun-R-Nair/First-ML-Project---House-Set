import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data=pd.read_csv("train.csv")
print(data.describe())
print(data.columns)
y = data.SalePrice
feat = ['MSSubClass',
    'LotArea', 'YearBuilt','OverallQual','OverallCond','1stFlrSF',
    '2ndFlrSF','LowQualFinSF','GrLivArea','FullBath','HalfBath',
    'BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces',
    'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch',
    'ScreenPorch','PoolArea','YrSold']
X= data[feat]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
model=RandomForestRegressor(n_estimators=178,random_state=1)
model.fit(X,y)

test_data=pd.read_csv("test.csv")
test_X=test_data[feat]
test_preds=model.predict(test_X)

output=pd.DataFrame({'Id': test_data.Id,'SalePrice': test_preds})
output.to_csv('submission.csv',index=False)

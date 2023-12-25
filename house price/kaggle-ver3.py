import pandas as pd
import numpy as np
import xgboost
import seaborn as sns
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import f_classif,SelectPercentile

df_train = pd.read_csv('data\house_train.csv/train.csv', sep=',')
df_test = pd.read_csv('data\house_test.csv/test.csv', sep=',')
# df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)
# print(df_train.shape)

# 訓練資料需要 train_X, train_Y / 預測輸出需要 ids(識別每個預測值), test_X
# 在此先抽離出 train_Y 與 ids, 而先將 train_X, test_X 該有的資料合併成 df, 先作特徵工程
train_Y = df_train['SalePrice']

ids = df_test['Id']
df_train = df_train.drop(['Id', 'SalePrice'] , axis=1)
df_test = df_test.drop(['Id'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()

# estimator = xgboost.XGBRegressor(random_state=42,colsample_bytree=1.0,learning_rate=0.1,max_depth = 3,n_estimators = 250,subsample=1.0,min_child_weight=1,alpha=10.0,reg_lambda=0.1)
#只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')

# 削減文字型欄位, 只剩數值型欄位
dfnum = df[num_features]
dfnum = dfnum.fillna(-1)

# 特徵工程

# feture engineering a new feature "TotalFS"
dfnum['TotalSF'] = (dfnum['TotalBsmtSF'] + dfnum['1stFlrSF']  + dfnum['2ndFlrSF'])

dfnum['YrBltAndRemod'] = dfnum['YearBuilt'] + dfnum['YearRemodAdd']

dfnum['Total_sqr_footage'] = (dfnum['BsmtFinSF1'] + dfnum['BsmtFinSF2']  + dfnum['1stFlrSF']  + dfnum['2ndFlrSF'])
                                 

dfnum['Total_Bathrooms'] = (dfnum['FullBath'] 
                               + (0.5 * dfnum['HalfBath']) 
                               + dfnum['BsmtFullBath'] 
                               + (0.5 * dfnum['BsmtHalfBath'])
                              )
                               

dfnum['Total_porch_sf'] = (dfnum['OpenPorchSF'] 
                              + dfnum['3SsnPorch'] 
                              + dfnum['EnclosedPorch'] 
                              + dfnum['ScreenPorch'] 
                              + dfnum['WoodDeckSF']
                             )
dfnum['haspool'] = dfnum['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
dfnum['has2ndfloor'] = dfnum['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
dfnum['hasgarage'] = dfnum['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
dfnum['hasbsmt'] = dfnum['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
dfnum['hasfireplace'] = dfnum['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
MMEncoder = StandardScaler()

dfnum.replace([np.inf, -np.inf], -1, inplace=True)
# 將 NaN 值填補為 -1
dfnum.fillna(-1, inplace=True)

for c in dfnum.columns:
    dfnum[c] = MMEncoder.fit_transform(dfnum[c].values.reshape(-1, 1))
train_num = train_Y.shape[0]

#只取類別值 (object) 型欄位, 存於 object_features 中
object_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'object':
        object_features.append(feature)
print(f'{len(object_features)} Object Features : {object_features}\n')

# 只留類別型欄位
df_obj = df[object_features]
df_obj = df_obj.fillna('None')
train_num = train_Y.shape[0]

# 如果存在缺失值，用全局均值填充缺失值
df_obj.fillna(-1, inplace=True)

# 將類別型欄位轉換為虛擬變數 (One-Hot Encoding)
df_onehot= pd.get_dummies(df_obj, dummy_na=True)  # dummy_na=True 用於處理缺失值

# 合併類別型和目標變量的 DataFrame
df_obj_with_target = pd.concat([df_obj[train_num:], train_Y], axis=1)

# 對每個類別型欄位進行均值編碼
for col in df_obj.columns:
    mean_encoding = df_obj_with_target.groupby(col)['SalePrice'].mean()
    df_obj[col] = df_obj[col].map(mean_encoding)
# 合併數值型和編碼後的類別型的 DataFrame
df_combined = pd.concat([dfnum, df_onehot], axis=1)

# 將 NaN 值填補為 -1
df_combined.fillna(-1, inplace=True)


# 將資料分為訓練集和測試集
train_X = df_combined[:train_num]
test_X = df_combined[train_num:]

xgb = xgboost.XGBRegressor(random_state=42,colsample_bytree=1.0,learning_rate=0.1,max_depth = 3,n_estimators = 300,subsample=1.0)

gdbt = GradientBoostingRegressor(tol=0.1, subsample=0.37, n_estimators=200, max_features=20,
                                 max_depth=6, learning_rate=0.03)
rf = RandomForestRegressor(n_estimators=300, min_samples_split=9, min_samples_leaf=10,
                           max_features='sqrt', max_depth=8, bootstrap=False)
from mlxtend.regressor import StackingRegressor

# 因為 Stacking 需要以模型作為第一層的特徵來源, 因此在 StackingRegressor 中,
# 除了要設本身(第二層)的判定模型 - meta_regressor, 也必須填入第一層的單模作為編碼器 - regressors
# 這裡第二層模型(meta_regressor)的參數, 一樣也需要用 Grid/Random Search, 請參閱講義中的 mlxtrend 網頁
meta_estimator = GradientBoostingRegressor(tol=10, subsample=0.44, n_estimators=100,
                                           max_features='log2', max_depth=4, learning_rate=0.1)
stacking = StackingRegressor(regressors=[xgb, gdbt, rf], meta_regressor=meta_estimator)
stacking .fit(train_X, train_Y)
pred = stacking .predict(test_X)

pred = np.expm1(pred)
sub = pd.DataFrame({'Id': ids, 'SalePrice': pred})
sub.to_csv('prediction20.csv', index=False)

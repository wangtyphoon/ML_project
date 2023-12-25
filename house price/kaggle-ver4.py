import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor,RandomForestRegressor
import numpy as np
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from mlxtend.regressor import StackingRegressor
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns

# 忽略所有警告
warnings.filterwarnings("ignore")
df_train = pd.read_csv('data\house_train.csv/train.csv', sep=',')
df_test = pd.read_csv('data\house_test.csv/test.csv', sep=',')

train_Y = np.log1p(df_train['SalePrice'])
# train_Y = df_train['SalePrice']

ids = df_test['Id']
df_train = df_train.drop(['Id', 'SalePrice'] , axis=1)
df_test = df_test.drop(['Id'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()

train_num = train_Y.shape[0]

def missing_value(df,num):
    # 計算每列的缺失值數量
    missing_values = df.isnull().sum()

    # 計算每列的總數
    total_values = df.shape[0]

    # 計算每列的缺失值比例
    missing_ratio = (missing_values / total_values) * 100

    # 創建一個包含缺失值數量和比例的 DataFrame
    missing_info = pd.DataFrame({
        'Missing Values': missing_values,
        'Missing Ratio (%)': missing_ratio,
        'Data Type': df.dtypes
    })
    # 按照缺失值比例降序排列
    missing_info_sorted = missing_info.sort_values(by='Missing Ratio (%)', ascending=False)

    # 打印排序後的結果
    print(missing_info_sorted[:num])

# 特徵工程-缺失值轉換
df['MSSubClass'] = df['MSSubClass'].astype(str)
df['MSZoning'] = df['MSZoning'].fillna('RL')
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
df['Alley'] = df['Alley'].fillna('None')
df['LotShape'] = df['LotShape'].replace({'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0})
df['LandContour'] = df['LandContour'].replace({'Lvl': 3, 'Bnk': 2, 'HLS': 1, 'Low': 0})
df['Utilities'] = df['Utilities'].fillna('AllPub')
df['LandSlope'] = df['LandSlope'].replace({'Gtl': 3, 'Mod': 2, 'Sev': 1})
df['Exterior1st'] = df['Exterior1st'].fillna('VinylSd')
df['Exterior2nd'] = df['Exterior2nd'].fillna('VinylSd')
df['MasVnrType'] = df['MasVnrType'].fillna('None')
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
df['ExterQual'] = df['ExterQual'].replace({'Ex': 5, 'Gd': 4,'TA': 3, 'Fa': 2,'Po': 1,})
df['ExterCond'] = df['ExterCond'].replace({'Ex': 5, 'Gd': 4,'TA': 3, 'Fa': 2,'Po': 1,})
df['BsmtQual'] = df['BsmtQual'].replace({'Ex': 5, 'Gd': 4,'TA': 3, 'Fa': 2,'Po': 1,})
df['BsmtQual'] = df['BsmtQual'].fillna(0)
df['BsmtCond'] = df['BsmtCond'].replace({'Ex': 5, 'Gd': 4,'TA': 3, 'Fa': 2,'Po': 1,})
df['BsmtCond'] = df['BsmtCond'].fillna(0)
df['BsmtExposure'] = df['BsmtExposure'].replace({'Gd': 3,'Av': 2, 'Mn': 1,'No': 0,})
df['BsmtExposure'] = df['BsmtExposure'].fillna(0)
df['BsmtFinType1'] = df['BsmtFinType1'].replace({'GLQ': 6,'ALQ': 5,'BLQ': 4, 'Rec': 3,  'LwQ': 2,  'Unf': 1,  'NA': 0 })
df['BsmtFinType1'] = df['BsmtFinType1'].fillna(0)
df['BsmtFinType2'] = df['BsmtFinType2'].replace({'GLQ': 6,'ALQ': 5,'BLQ': 4, 'Rec': 3,  'LwQ': 2,  'Unf': 1,  'NA': 0 })
df['BsmtFinType2'] = df['BsmtFinType2'].fillna(0)
df['BsmtFinSF2'] = df['BsmtFinSF2'].fillna(0)
df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(0)
df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(0)
df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)
df['HeatingQC'] = df['HeatingQC'].replace({'Ex': 5, 'Gd': 4,'TA': 3, 'Fa': 2,'Po': 1,})
df['Electrical'] = df['Electrical'].fillna('SBrkr')
df['Electrical'] = df['Electrical'].replace({'SBrkr': 5, 'FuseA': 4,'Mix': 3, 'FuseF': 2,'FuseP': 1,})
df['BsmtFullBath'] = df['BsmtFullBath'].fillna(0.0)
df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(0.0)
df['KitchenQual'] = df['KitchenQual'].replace({'Ex': 5, 'Gd': 4,'TA': 3, 'Fa': 2,'Po': 1,})
df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mean())
df['Functional'] = df['Functional'].replace({'Typ': 6,    'Min1': 5,   'Min2': 4,   'Mod': 3,    'Maj1': 2,  'Maj2': 1,   'Sev': 0,    'Sal': -1}) 
df['Functional'] = df['Functional'].fillna(0)
df['FireplaceQu'] = df['FireplaceQu'].fillna(0)
df['FireplaceQu'] = df['FireplaceQu'].replace({'Ex': 5, 'Gd': 4,'TA': 3, 'Fa': 2,'Po': 1,})
df['GarageType'] = df['GarageType'].fillna('Attchd')
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean())
df['GarageFinish'] = df['GarageFinish'].replace({'Fin': 3,  'RFn': 2,   'Unf': 1})
df['GarageFinish'] = df['GarageFinish'].fillna(0)
df['GarageCars'] = df['GarageCars'].fillna(0)
df['GarageArea'] = df['GarageArea'].fillna(0)
df['GarageQual'] = df['GarageQual'].replace({'Ex': 5, 'Gd': 4,'TA': 3, 'Fa': 2,'Po': 1,})
df['GarageQual'] = df['GarageQual'].fillna(0)
df['GarageCond'] = df['GarageCond'].replace({'Ex': 5, 'Gd': 4,'TA': 3, 'Fa': 2,'Po': 1,})
df['GarageCond'] = df['GarageCond'].fillna(0)
df['PavedDrive'] = df['PavedDrive'].replace({'Y': 2,   'P': 1,  'N': 0  })
df['PoolQC'] = df['PoolQC'].fillna(0)  # 將 NaN 轉換為 0
df['PoolQC'] = df['PoolQC'].replace({'Ex': 3, 'Gd': 2, 'Fa': 1})
df['Fence'] = df['Fence'].fillna(0)
df['Fence'] = df['Fence'].replace({'GdWo': 4,  'MnPrv': 3,'GdPrv': 2,'MnWw': 1})
df['MiscFeature'] = df['MiscFeature'].fillna('None')  # 將 NaN 轉換為 0
df['SaleType'] = df['SaleType'].fillna('Normal')
df['YrSold'] = df['YrSold'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)

# 特徵工程-資料新增
df['OwnMasVnr'] = (df['MasVnrArea'] > 0).astype(int)
df['TFBsmtUnfSF'] = (df['BsmtUnfSF'] > 0).astype(int)
df['TFTotalBsmtSF'] = (df['TotalBsmtSF'] > 0).astype(int)
df['TotalSF'] = (df['TotalBsmtSF'] + df['1stFlrSF']  + df['2ndFlrSF'])
df['TFLowQualFinSF'] = (df['LowQualFinSF'] > 0).astype(int)
df['Total_Bathrooms'] = (df['FullBath']  + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))
df['hasgarage'] = (df['GarageArea']> 0).astype(int)
df['hasWoodDeckSF'] = (df['WoodDeckSF']> 0).astype(int)
df['hasOpenPorchSF'] = (df['OpenPorchSF']> 0).astype(int)
df['hasEnclosedPorch'] = (df['EnclosedPorch']> 0).astype(int)
df['has3SsnPorch'] = (df['3SsnPorch']> 0).astype(int)
df['hasScreenPorch'] = (df['ScreenPorch']> 0).astype(int)
df['hasPoolArea'] = (df['PoolArea']> 0).astype(int)
df['YrBltAndRemod'] = df['YearBuilt'] + df['YearRemodAdd']
df['YrBltAndRemod'] = df['YearBuilt'] - df['YearRemodAdd']
df['Total_sqr_footage'] = (df['BsmtFinSF1'] + df['BsmtFinSF2']  + df['1stFlrSF']  + df['2ndFlrSF'])
df['Total_porch_sf'] = (df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch']  + df['WoodDeckSF'])

#只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')
dfnum = df[num_features]

# # 計算每個數值型特徵與目標變數的相關性
# correlation_with_list = dfnum[:train_num].corrwith(pd.Series(train_Y))

# # 將相關性按照絕對值升序排列並選取最低的九個特徵
# lowest_nine_correlations = correlation_with_list.sort_values(ascending=True).head(9)

# # 獲取要刪除的欄位名稱
# columns_to_drop = lowest_nine_correlations.index

# # 在資料集中刪除這九個欄位
# dfnum = dfnum.drop(columns=columns_to_drop, axis=1)
MMEncoder = StandardScaler()
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

# 將類別型欄位轉換為虛擬變數 (One-Hot Encoding)
df_onehot= pd.get_dummies(df_obj, dummy_na=True)  # dummy_na=True 用於處理缺失值

# 合併數值型和編碼後的類別型的 DataFrame
df_combined = pd.concat([dfnum, df_onehot], axis=1)

# 將資料分為訓練集和測試集
train_X = df_combined[:train_num]
test_X = df_combined[train_num:]
xgb = XGBRegressor(learning_rate=0.1, max_depth=3, n_estimators=250, random_state=42)

# base_models = [
#     ('rf', RandomForestRegressor(n_estimators=100, min_samples_split=5, max_depth=30)),
#     ('gb', GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)),
#     ('svr', SVR(kernel='linear', gamma=0.01, C=0.1)),
#     ('xgb', XGBRegressor(learning_rate=0.1, max_depth=3, n_estimators=250, random_state=42)),
#     ('ridge', Ridge(alpha=10)),
# ]

# Define meta model
# meta_model = XGBRegressor(learning_rate=0.1, max_depth=3, n_estimators=250, random_state=42)

# Create StackingRegressor
# stacking = StackingRegressor(estimators=base_models, final_estimator=meta_model)
# gbft = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
# # Evaluate the stacked model using cross-validation
# print(cross_val_score(xgb, train_X, train_Y, cv=5, scoring='neg_root_mean_squared_error').mean())
# print(cross_val_score(gbft, train_X, train_Y, cv=5, scoring='neg_root_mean_squared_error').mean())

# stacking.fit(train_X, train_Y)
# pred = stacking.predict(test_X)

# pred = np.expm1(pred)
# sub = pd.DataFrame({'Id': ids, 'SalePrice': pred})
# sub.to_csv('predictionstacking.csv', index=False)
# Fit the stacking model
# xgb.fit(train_X, train_Y)

# # Make predictions on the training set
# stacking_pred = xgb.predict(train_X)

# # Plotting the actual vs predicted values
# plt.figure(figsize=(10, 6))
# plt.scatter(train_Y, stacking_pred, alpha=0.5)
# plt.plot([min(train_Y), max(train_Y)], [min(train_Y), max(train_Y)], '--', color='red', linewidth=2)
# plt.title('Actual vs Predicted Values - xgbRegressor')
# plt.xlabel('Actual SalePrice (log-transformed)')
# plt.ylabel('Predicted SalePrice (log-transformed)')
# plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
rf = RandomForestRegressor(n_estimators=300, min_samples_split=9, min_samples_leaf=10,
                           max_features='sqrt', max_depth=8, bootstrap=False)
estimator = XGBRegressor(learning_rate=0.1,max_depth = 3,n_estimators = 300)
gdbt = GradientBoostingRegressor(tol=0.1, subsample=0.37, n_estimators=200, max_features=20,
                                 max_depth=6, learning_rate=0.03)
meta_estimator = GradientBoostingRegressor(tol=1, subsample=0.44, n_estimators=50, max_depth=4, learning_rate=0.1,min_samples_leaf=2,alpha=0.9)
hist = HistGradientBoostingRegressor(max_iter=300, max_depth=1, learning_rate=0.1)
cat = CatBoostRegressor(iterations=300, depth=4, learning_rate=0.05, loss_function='RMSE', subsample=0.3)
lgbm = LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.07, subsample=0.3)
xgb = XGBRegressor(learning_rate=0.1,max_depth = 3,n_estimators = 300)
# print(cross_val_score(estimator, train_X, train_Y, cv=5, scoring='neg_root_mean_squared_error').mean())

# 堆疊泛化套件 mlxtrend, 需要先行安裝(使用 pip 安裝即可)在執行環境下
from mlxtend.regressor import StackingRegressor
from sklearn.model_selection import RandomizedSearchCV

# 因為 Stacking 需要以模型作為第一層的特徵來源, 因此在 StackingRegressor 中,
# 除了要設本身(第二層)的判定模型 - meta_regressor, 也必須填入第一層的單模作為編碼器 - regressors
# 這裡第二層模型(meta_regressor)的參數, 一樣也需要用 Grid/Random Search, 請參閱講義中的 mlxtrend 網頁
meta_estimator = GradientBoostingRegressor(tol=5, subsample=0.44, n_estimators=150,
                                           max_features='log2', max_depth=4, learning_rate=0.05,min_samples_leaf=6,alpha=0.7)
stacking = StackingRegressor(regressors=[xgb, hist, cat], meta_regressor=meta_estimator)
# print(stacking.get_params())
# # Parameter grid for RandomizedSearchCV
# params = {
#     'meta_regressor__alpha' : [0.7, 0.8, 0.9],
#     'meta_regressor__n_estimators': [50, 100, 150],
#     'meta_regressor__max_depth': [3, 4, 5],
#     'meta_regressor__learning_rate': [0.1, 0.05, 0.01],
#     'meta_regressor__tol' : [2, 5, 10],
#     'meta_regressor__min_samples_leaf':[2,4,6]
# }

# # RandomizedSearchCV
# grid = RandomizedSearchCV(estimator=stacking, param_distributions=params, cv=5, n_iter=10, scoring='neg_root_mean_squared_error', random_state=42)

# # Fit the RandomizedSearchCV object to the data
# grid.fit(train_X, train_Y)

# # Print the best parameters and the corresponding performance score
# print("Best Parameters: ", grid.best_params_)
# print("Best Score: ", grid.best_score_)

# # grid.fit(X, y)
print(cross_val_score(stacking, train_X, train_Y, cv=5, scoring='neg_root_mean_squared_error').mean())

stacking .fit(train_X, train_Y)
pred = stacking .predict(test_X)

pred = np.expm1(pred)
sub = pd.DataFrame({'Id': ids, 'SalePrice': pred})
sub.to_csv('prediction20.csv', index=False)

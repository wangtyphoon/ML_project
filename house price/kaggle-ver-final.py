import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
import numpy as np
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingRegressor
import matplotlib.pyplot as plt

# 忽略所有警告
warnings.filterwarnings("ignore")

# 讀取訓練和測試資料
df_train = pd.read_csv('data\house_train.csv/train.csv', sep=',')
df_test = pd.read_csv('data\house_test.csv/test.csv', sep=',')

# 對目標變數 'SalePrice' 做 log 轉換
train_Y = np.log1p(df_train['SalePrice'])

# 從測試集中提取 'Id' 欄位以供後續使用
ids = df_test['Id']

# 從訓練集中刪除 'Id' 和 'SalePrice' 欄位
df_train = df_train.drop(['Id', 'SalePrice'], axis=1)

# 從測試集中刪除 'Id' 欄位
df_test = df_test.drop(['Id'], axis=1)

# 將訓練和測試集連接起來以進行特徵工程
df = pd.concat([df_train, df_test])

# 顯示連接後的 DataFrame 的前幾列
df.head()

train_num = train_Y.shape[0]

# 顯示 DataFrame 中缺失值的相關信息的函數
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

# 特徵工程 - 處理特定欄位的缺失值
# 將欄位轉換為適當的資料型別
# 使用有意義的替代值或填補方法處理特定欄位的缺失值
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

# 處理缺失值後顯示相關信息
missing_value(df, train_num)

# 提取數值特徵並應用標準化
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')
dfnum = df[num_features]

MMEncoder = StandardScaler()
for c in dfnum.columns:
    dfnum[c] = MMEncoder.fit_transform(dfnum[c].values.reshape(-1, 1))

# 提取類別特徵進行 one-hot 編碼
object_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'object':
        object_features.append(feature)
print(f'{len(object_features)} Object Features : {object_features}\n')

# 提取類別特徵並使用 one-hot 編碼轉換為虛擬變數
df_obj = df[object_features]
df_onehot = pd.get_dummies(df_obj, dummy_na=True)

# 合併數值型和編碼後的類別型的 DataFrame
df_combined = pd.concat([dfnum, df_onehot], axis=1)

# 將資料分割回訓練和測試集
train_X = df_combined[:train_num]
test_X = df_combined[train_num:]

# 定義集成模型
gdbt = GradientBoostingRegressor(tol=0.1, subsample=0.37, n_estimators=200, max_features=20,
                                 max_depth=6, learning_rate=0.03)
hist = HistGradientBoostingRegressor(max_iter=500, max_depth=1, learning_rate=0.1)
cat = CatBoostRegressor(iterations=300, depth=4, learning_rate=0.05, loss_function='RMSE', subsample=0.3)
lgbm = LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.07, subsample=0.3)
xgb = XGBRegressor(learning_rate=0.1, max_depth=3, n_estimators=300)

# 創建 VotingRegressor 模型
voting_reg = VotingRegressor([('xgb', xgb), ('gdbt', gdbt), ('cat', cat), ('lgbm', lgbm), ('hist', hist)])
voting_reg.fit(train_X, train_Y)

# 在測試集上進行預測
pred = voting_reg.predict(train_X)

# 將預測結果從對數尺度轉換回原始尺度
pred = np.expm1(pred)

# # 創建一個提交的 DataFrame
# sub = pd.DataFrame({'Id': ids, 'SalePrice': pred})

# # 將提交的 DataFrame 保存為 CSV 檔案
# sub.to_csv('final_house.csv', index=False)

# 使用交叉驗證評估集成模型的性能
print(cross_val_score(voting_reg, train_X, train_Y, cv=5, scoring='neg_root_mean_squared_error').mean())
# 原始数据的目标变量
y_true = np.exp(train_Y) - 1  # 将对数尺度转换回原始尺度

# 创建一个散点图，以观察原始数据与预测结果之间的关系
plt.scatter(y_true, pred)
plt.xlabel('True SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('True vs Predicted SalePrice(manual)')

# 绘制对角线（y=x），表示完美预测的情况
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], linestyle='--', color='red')
plt.grid('--')
plt.savefig("manual.jpg",dpi=400)

plt.show()
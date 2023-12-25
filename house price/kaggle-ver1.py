import pandas as pd
import numpy as np
import xgboost
import seaborn as sns
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

df_train = pd.read_csv('data\house_train.csv/train.csv', sep=',')
df_test = pd.read_csv('data\house_test.csv/test.csv', sep=',')
print(df_train.shape)

# 訓練資料需要 train_X, train_Y / 預測輸出需要 ids(識別每個預測值), test_X
# 在此先抽離出 train_Y 與 ids, 而先將 train_X, test_X 該有的資料合併成 df, 先作特徵工程
# train_Y = np.log1p(df_train['SalePrice'])
train_Y = df_train['SalePrice']

ids = df_test['Id']

df_train = df_train.drop(['Id', 'SalePrice'] , axis=1)
df_test = df_test.drop(['Id'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()

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
dfnum['AgeDiff'] = dfnum['YearBuilt'] - dfnum['YearRemodAdd']
dfnum['BsmtTo1stFlrRatio'] = dfnum['TotalBsmtSF'] / dfnum['1stFlrSF']
dfnum['BathToRoomRatio'] = dfnum['FullBath'] / dfnum['TotRmsAbvGrd']
dfnum['AreaToRoomRatio'] = dfnum['GrLivArea'] / dfnum['TotRmsAbvGrd']
dfnum['BathToGarageRatio'] = dfnum['FullBath'] / dfnum['GarageCars']
dfnum['BedroomToBathRatio'] = dfnum['BedroomAbvGr'] / dfnum['FullBath']
dfnum['BathroomToRoomRatio'] = (dfnum['FullBath'] + 0.5 * dfnum['HalfBath']) / dfnum['TotRmsAbvGrd']
dfnum['AreaToGarageRatio'] = dfnum['GrLivArea'] / dfnum['GarageArea']
dfnum['GarageCarsToAreaRatio'] = dfnum['GarageCars'] / dfnum['GarageArea']
dfnum['BathroomToBedroomRatio'] = (dfnum['FullBath'] + 0.5 * dfnum['HalfBath']) / dfnum['BedroomAbvGr']

MMEncoder = StandardScaler()

# # 将训练数据和目标变量合并
# df_train_corr = pd.concat([dfnum[:len(df_train)], train_Y], axis=1)

# # 计算相关性矩阵
# correlation_matrix = df_train_corr.corr()

# # 取出与目标变量的相关性列
# correlation_with_price = correlation_matrix['SalePrice']
# # 按相关性降序排序
# sorted_correlation = correlation_with_price.abs().sort_values(ascending=False)

# # 打印排序后的相关性
# print(sorted_correlation)
dfnum.replace([np.inf, -np.inf], -1, inplace=True)
# 將 NaN 值填補為 -1
dfnum.fillna(-1, inplace=True)

for c in dfnum.columns:
    dfnum[c] = MMEncoder.fit_transform(dfnum[c].values.reshape(-1, 1))
estimator = xgboost.XGBRegressor(random_state=42,colsample_bytree=1.0,learning_rate=0.1,max_depth = 3,n_estimators = 200,subsample=1.0)


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
df_obj.head()

train_num = train_Y.shape[0]
# 將類別型欄位轉換為虛擬變數 (One-Hot Encoding)
df_objects_encoded = pd.get_dummies(df_obj, dummy_na=True)  # dummy_na=True 用於處理缺失值


# 合併數值型和類別型的 DataFrame
df_combined = pd.concat([dfnum, df_objects_encoded], axis=1)

# 將 NaN 值填補為 -1
df_combined.fillna(-1, inplace=True)

# 將資料分為訓練集和測試集
train_X = df_combined[:train_num]
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

test_X = df_combined[train_num:]
# # 定義要調整的超參數範圍
# param_grid = {
#     'grow_policy': ['depthwise', 'lossguide'],         # 決策樹的數量
#     'learning_rate': [0.05, 0.1, 0.15],      # 學習速率
# }

# # 使用 GridSearchCV 進行超參數搜索
# grid_search = GridSearchCV(estimator, param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(train_X, train_Y)

# # 打印最佳超參數組合和相應的評分
# print("Best Parameters:", grid_search.best_params_)
# print("Best Negative MSE:", grid_search.best_score_)
estimator.fit(train_X, train_Y)
pred = estimator.predict(test_X)

# pred = np.expm1(pred)
sub = pd.DataFrame({'Id': ids, 'SalePrice': pred})
sub.to_csv('prediction.csv', index=False)


# 獲取特徵重要性
feature_importance = estimator.feature_importances_

# 將特徵名稱和相應的重要性組合起來
feature_importance_dict = dict(zip(train_X.columns, feature_importance))

# 將字典按照重要性降序排序
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# 只保留前二十個特徵
top_features = sorted_feature_importance[:20]

# 打印特徵重要性
for feature, importance in top_features:
    print(f"{feature}: {importance}")

# 繪製前二十個特徵重要性的長條圖
plt.figure(figsize=(10, 6))
plt.barh(range(len(top_features)), [importance for _, importance in top_features], align='center')
plt.yticks(range(len(top_features)), [feature for feature, _ in top_features])
plt.xlabel('Feature Importance')
plt.title('Top 20 XGBoost Feature Importance')
plt.show()

#定义绘制学习曲线的函数
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# 绘制学习曲线
plot_learning_curve(estimator, "Learning Curve", train_X, train_Y, ylim=(0.7, 1.01), cv=5, n_jobs=4)

plt.show()
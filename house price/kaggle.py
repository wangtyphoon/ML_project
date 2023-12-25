# 載入基本套件
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

# 載入標籤編碼與最小最大化, 以便做最小的特徵工程
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df_train = pd.read_csv('data\house_train.csv/train.csv', sep=',')
df_test = pd.read_csv('data\house_test.csv/test.csv', sep=',')
print(df_train.shape)

# 訓練資料需要 train_X, train_Y / 預測輸出需要 ids(識別每個預測值), test_X
# 在此先抽離出 train_Y 與 ids, 而先將 train_X, test_X 該有的資料合併成 df, 先作特徵工程
train_Y = np.log1p(df_train['SalePrice'])
ids = df_test['Id']
df_train = df_train.drop(['Id', 'SalePrice'] , axis=1)
df_test = df_test.drop(['Id'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()

# 特徵工程-簡化版 : 全部空值先補-1, 所有類別欄位先做 LabelEncoder, 然後再與數字欄位做 MinMaxScaler
# 這邊使用 LabelEncoder 只是先將類別欄位用統一方式轉成數值以便輸入模型, 當然部分欄位做 One-Hot可能會更好, 只是先使用最簡單版本作為範例
LEncoder = LabelEncoder()
# 除上述之外, 還要把標籤編碼與數值欄位一起做最大最小化, 這麼做雖然有些暴力, 卻可以最簡單的平衡特徵間影響力
MMEncoder = MinMaxScaler()
for c in df.columns:
    if df[c].dtype == 'object': # 如果是文字型 / 類別型欄位, 就先補缺 'None' 後, 再做標籤編碼
        df[c] = df[c].fillna('None')
        df[c] = LEncoder.fit_transform(df[c])
    else: # 其他狀況(本例其他都是數值), 就補缺 -1
        df[c] = df[c].fillna(-1)
    # 最後, 將標籤編碼與數值欄位一起最大最小化, 因為需要是一維陣列, 所以這邊切出來後用 reshape 降維
    df[c] = MMEncoder.fit_transform(df[c].values.reshape(-1, 1))
df.head()

# 將前述轉換完畢資料 df , 重新切成 train_X, test_X, 因為不論何種特徵工程, 都需要對 train / test 做同樣處理
# 常見並簡便的方式就是 - 先將 train / test 接起來, 做完後再拆開, 不然過程當中往往需要將特徵工程部分寫兩次, 麻煩且容易遺漏
# 在較複雜的特徵工程中尤其如此, 若實務上如果碰到 train 與 test 需要分階段進行, 則通常會另外寫成函數處理
train_num = train_Y.shape[0]
train_X = df[:train_num]
test_X = df[train_num:]

# 使用線性迴歸模型 : train_X, train_Y 訓練模型, 並對 test_X 做出預測結果 pred
from sklearn.linear_model import LinearRegression
estimator = LinearRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

# estimator.fit(train_X, train_Y)
# pred = estimator.predict(test_X)

# # 將輸出結果 pred 與前面留下的 ID(ids) 合併, 輸出成檔案
# # 可以下載並點開 house_baseline.csv 查看結果, 以便了解預測結果的輸出格式
# # 本範例所與作業所輸出的 csv 檔, 均可用於本題的 Kaggle 答案上傳, 可以試著上傳來熟悉 Kaggle 的介面操作
# pred = np.expm1(pred)
# sub = pd.DataFrame({'Id': ids, 'SalePrice': pred})
# sub.to_csv('prediction.csv', index=False)
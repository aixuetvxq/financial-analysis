import tushare as ts
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

year = 2019
quarter = 3
basics_df = ts.get_stock_basics("2019-09-30")
profit_df = ts.get_profit_data(year, quarter)
growth_df = ts.get_growth_data(year, quarter)
debtpaying_df = ts.get_debtpaying_data(year, quarter)
cashflow_df = ts.get_cashflow_data(year, quarter)
df1 = pd.merge(basics_df, profit_df, on=['code'])  # 并表
df2 = pd.merge(df1, growth_df, on=['code'])
df3 = pd.merge(df2, debtpaying_df, on=['code'])
df = pd.merge(df3, cashflow_df, on=['code'])

df = df.drop_duplicates(subset=['code'], keep='first')  # 删除code的重复行
df = df.dropna(axis=0, how='any')  # 删除空值行
df = df.loc[:, ~df.columns.duplicated()]  # 删除重复列

col_a1 = df['business_income']
col_a2 = df['totalAssets']
col_a3 = df['net_profits']
col_a4 = df['sheqratio']
roa = col_a1 / col_a2
tassrat = col_a3 / col_a2
dbastrt = 1 - col_a4
df.insert(5, 'roa', roa)  # roa资产回报率
df.insert(6, 'tassrat', tassrat)  # tassrat总资产周转率
df.insert(7, 'dbastrt', dbastrt)  # dbastrt资产负债率

col = ['code', 'name', 'industry', 'roe', 'roa', 'tassrat', 'mbrg', 'targ',
       'currentratio', 'dbastrt', 'cf_sales']
df = pd.DataFrame(df, columns=col)
df = df[~df.industry.str.contains('金融行业')]  # 数据预处理 删除异常值
df = df[~df.name.str.contains('ST', 'PT')]
df = df[df['mbrg'] > 1.5]
df = df[df['dbastrt'] < 1]

winsorize_percent = 0.01  # 尾数处理
df['roe'] = stats.mstats.winsorize(df['roe'], (winsorize_percent, winsorize_percent))
df['roa'] = stats.mstats.winsorize(df['roa'], (winsorize_percent, winsorize_percent))
df['tassrat'] = stats.mstats.winsorize(df['tassrat'], (winsorize_percent, winsorize_percent))
df['mbrg'] = stats.mstats.winsorize(df['mbrg'], (winsorize_percent, winsorize_percent))
df['targ'] = stats.mstats.winsorize(df['targ'], (winsorize_percent, winsorize_percent))
df['currentratio'] = pd.to_numeric(df['currentratio'])
df['currentratio'] = stats.mstats.winsorize(df['currentratio'], (winsorize_percent, winsorize_percent))
df['dbastrt'] = stats.mstats.winsorize(df['dbastrt'], (winsorize_percent, winsorize_percent))
df['cf_sales'] = stats.mstats.winsorize(df['cf_sales'], (winsorize_percent, winsorize_percent))

print('------------------------------统计信息--------------------------------')
print(df.describe().T)  # 查看变量df中各个字段的计数、平均值、标准差、最小值、下四分位数、中位数、上四分位、最大值
print(df.drop(['code', 'name', 'industry'], axis=1).corr())  # 相关系数
print('------------------------------统计信息--------------------------------')

y = df[['roe', 'roa']]
df = df.drop(['code', 'name', 'industry', 'roe', 'roa'], axis=1)
# df = (df-df.min())/(df.max()-df.min())
# print(df)
X_train, X_test, y_train, y_test = train_test_split(df.values, y, test_size=0.1, random_state=0)
# print(X_train, X_test, y_train, y_test)
lr = LinearRegression()
lr.fit(X_train, y_train)
print('------------------------------训练、测试结果--------------------------------')
print(lr.coef_)
print(lr.intercept_)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))
print('------------------------------训练、测试结果--------------------------------')

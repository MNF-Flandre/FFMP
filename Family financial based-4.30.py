#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from datetime import datetime, timedelta
import akshare as ak
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# 指定CSV文件的路径
csv_file_path = r'D:\Course\大三下\家庭理财实务\family financial.csv'

# 使用pandas的read_csv函数导入CSV数据
df = pd.read_csv(csv_file_path)


# In[2]:


df1 = df[~df['stknme'].str.contains('ST|退|B')]
filtered_stkcd = df1.loc[~df['Audittyp'].isin(['标准无保留意见', '无保留意见加事项段']), 'Stkcd'].tolist()
df2 = df1[~df1['Stkcd'].isin(filtered_stkcd)]
df2 = df2[~df2['Stkcd'].str.startswith(( '6'))]
df3 = df2[~df2['nnindcd'].str.contains('[Jj]')]
df4 = df3.dropna()
df5 = df4.groupby('Stkcd').filter(lambda x: x['accper'].diff().nunique() == 1)
df6 = df5.groupby('Stkcd').filter(lambda x: x['accper'].min()<=2019)
df7 = df6.groupby('Stkcd').filter(lambda x: (x['F011201A'] > 0).all() and (x['F011201A'] < 1).all())
df8 = df7.dropna()
df8 = df8.drop('Audittyp', axis=1)
df8['Stkcd'] = df8['Stkcd'].astype(str).str.zfill(6)
df8['nnindcd'] = df8['nnindcd'].str[0]
df8['S'] = df8['Shrcr1'] * 2 - df8['Shrcr2']
df8 = df8.drop(['Shrcr1', 'Shrcr2'], axis=1)
df8 = df8.rename(columns={'F100101B': 'PE', 'F020108': 'EPS', 'F050201B': 'ROA', 'F080601A': 'AGR', 'F011201A': 'Debt', 'F062401B': 'FCFF'})
df8 = df8.reset_index(drop=True)
mean_values = df8.groupby(['nnindcd', 'accper']).median()
new_column_names = ['IFPE', 'IFEPS', 'IFROA', 'IFAGR', 'IFDebt', 'IFS', 'IFFCFF']
for column in new_column_names:
    df8[column] = None
df9 = mean_values.reset_index()

# 对 df8 进行逐行遍历
for idx, row in df8.iterrows():
    # 获取 nnindcd 和 accper 列的值
    nnindcd = row['nnindcd']
    accper = row['accper']

    # 使用 df9 的 loc 方法，找到 nnindcd 和 accper 列与 df8 中相同的行
    df9_row = df9.loc[(df9['nnindcd'] == nnindcd) & (df9['accper'] == accper)]

    # 使用 df8 和 df9 中锁定的两行，比较 F100101B、F020108、F050201B、F080601A、F011201A、Shrcr1、Shrcr2、F062401B 列的值
    if df8.loc[idx, 'PE'] < df9_row['PE'].values[0]:
        df8.loc[idx, 'IFPE'] = 1
    else:
        df8.loc[idx, 'IFPE'] = 0
        
    if df8.loc[idx, 'EPS'] > df9_row['EPS'].values[0]:
        df8.loc[idx, 'IFEPS'] = 1
    else:
        df8.loc[idx, 'IFEPS'] = 0
        
    if df8.loc[idx, 'ROA'] > df9_row['ROA'].values[0]:
        df8.loc[idx, 'IFROA'] = 1
    else:
        df8.loc[idx, 'IFROA'] = 0
        
    if df8.loc[idx, 'AGR'] > df9_row['AGR'].values[0]:
        df8.loc[idx, 'IFAGR'] = 1
    else:
        df8.loc[idx, 'IFAGR'] = 0
        
    if df8.loc[idx, 'Debt'] < df9_row['Debt'].values[0]:
        df8.loc[idx, 'IFDebt'] = 1
    else:
        df8.loc[idx, 'IFDebt'] = 0
        
    if df8.loc[idx, 'FCFF'] < df9_row['FCFF'].values[0]:
        df8.loc[idx, 'IFFCFF'] = 1
    else:
        df8.loc[idx, 'IFFCFF'] = 0
        
    if df8.loc[idx, 'S'] < df9_row['S'].values[0]:
        df8.loc[idx, 'IFS'] = 1
    else:
        df8.loc[idx, 'IFS'] = 0
        
df8['IF'] =(df8['IFPE']+df8['IFEPS']+df8['IFROA']+df8['IFAGR']+df8['IFDebt']+df8[ 'IFS']+df8['IFFCFF'])//4
df9=df8.drop( ['stknme','S','PE',  'EPS',  'ROA', 'AGR', 'Debt', 'FCFF','IFPE', 'IFEPS', 'IFROA', 'IFAGR', 'IFDebt', 'IFS', 'IFFCFF','nnindcd'],axis=1)
df9.sort_values('accper', inplace=True)
# 提取年份
df9['Year'] = pd.to_datetime(df9['accper'], format='%Y').dt.year
# 寻找最后一个工作日或交易日
df9['Trading_Day'] = df9['Year'].apply(lambda year: datetime(year+1, 4, 30) if datetime(year+1, 4, 30).weekday() < 5 else datetime(year+1, 4, 30) + timedelta(days=-(datetime(year+1, 4, 30).weekday() - 4)))
df8['IF2']=(df8['IFPE']+df8['IFEPS']+df8['IFROA']+df8['IFAGR']+df8['IFDebt']+df8[ 'IFS']+df8['IFFCFF'])
df8r=df8.drop(['stknme','S','PE',  'EPS',  'ROA', 'AGR', 'Debt', 'FCFF','IFPE', 'IFEPS', 'IFROA', 'IFAGR', 'IFDebt', 'IFS', 'IFFCFF','nnindcd'],axis = 1)
df9['price'] = None
for idx, row in df9.iterrows():
    code = row['Stkcd']
    target_date = row['Trading_Day'].strftime("%Y-%m-%d")

    stock_hfq_df = ak.stock_zh_a_hist(symbol=code, adjust="hfq")
    filtered_df = stock_hfq_df.loc[stock_hfq_df['日期'] == target_date]
    
    if not filtered_df.empty:
        close_price = filtered_df['收盘'].values[0]
        df9.loc[idx, 'price'] = close_price
    else:
        df9.loc[idx, 'price'] = None
for idx, row in df9.iterrows():
    if pd.isnull(row['price']):
        code = row['Stkcd']
        target_date = pd.to_datetime(row['Trading_Day']).strftime("%Y-%m-%d")

        stock_hfq_df = ak.stock_zh_a_hist(symbol=code, adjust="hfq")
        filtered_df = stock_hfq_df.loc[stock_hfq_df['日期'] <= target_date]
        
        while not filtered_df.empty:
            close_price = filtered_df.iloc[-1]['收盘']
            if pd.notnull(close_price):
                df9.loc[idx, 'price'] = close_price
                break
            filtered_df = filtered_df.iloc[:-1]
df9.to_csv(r'D:\Course\大三下\家庭理财实务\family_financial_clean_data.csv', index=False, encoding='utf-8-sig')
df8.to_csv(r'D:\Course\大三下\家庭理财实务\family_financial_complex_data.csv', index=False, encoding='utf-8-sig')


# In[202]:


df_sorted= pd.read_csv(r'D:\Course\大三下\家庭理财实务\family_financial_complex_data.csv')

df_sorted['Stkcd'] = df_sorted['Stkcd'].astype(str).str.zfill(6)
df_sorted = df_sorted.sort_values(['Stkcd', 'accper'], ascending=[True, True])
df_sd = pd.read_csv(r'D:\Course\大三下\家庭理财实务\family_financial_clean_data.csv').sort_values(['Stkcd', 'accper'], ascending=[True, True])
df_sd.reset_index(inplace=True)
df_sorted['price'] = df_sd['price']
df_sorted


# In[204]:


df_sorted['No_debt']=(df_sorted['IFPE']+df_sorted['IFEPS']+df_sorted['IFROA']+df_sorted['IFAGR']+df_sorted[ 'IFS']+df_sorted['IFFCFF'])
df_sorted['No_eps']=(df_sorted['IFPE']+df_sorted['IFROA']+df_sorted['IFAGR']+df_sorted['IFDebt']+df_sorted[ 'IFS']+df_sorted['IFFCFF'])
df_sorted['No_roa']=(df_sorted['IFPE']+df_sorted['IFEPS']+df_sorted['IFAGR']+df_sorted['IFDebt']+df_sorted[ 'IFS']+df_sorted['IFFCFF'])
df_sorted['No_agr']=(df_sorted['IFPE']+df_sorted['IFEPS']+df_sorted['IFROA']+df_sorted['IFDebt']+df_sorted[ 'IFS']+df_sorted['IFFCFF'])
df_sorted['No_s']=(df_sorted['IFPE']+df_sorted['IFEPS']+df_sorted['IFROA']+df_sorted['IFAGR']+df_sorted['IFDebt']+df_sorted['IFFCFF'])
df_sorted['No_fcff']=(df_sorted['IFPE']+df_sorted['IFEPS']+df_sorted['IFROA']+df_sorted['IFAGR']+df_sorted['IFDebt']+df_sorted[ 'IFS'])
df_sorted['No_pe']=(df_sorted['IFEPS']+df_sorted['IFROA']+df_sorted['IFAGR']+df_sorted['IFDebt']+df_sorted[ 'IFS']+df_sorted['IFFCFF'])
df_sorted['No_fcff_roa']=(df_sorted['IFPE']+df_sorted['IFEPS']+df_sorted['IFAGR']+df_sorted['IFDebt']+df_sorted[ 'IFS'])
df_sorted['No_eps_roa']=(df_sorted['IFPE']+df_sorted['IFAGR']+df_sorted['IFDebt']+df_sorted[ 'IFS']+df_sorted['IFFCFF'])
df_sorted['No_fcff_eps']=(df_sorted['IFPE']+df_sorted['IFROA']+df_sorted['IFAGR']+df_sorted['IFDebt']+df_sorted['IFS'])
df_sorted = df_sorted.drop( ['S','PE',  'EPS',  'ROA', 'AGR', 'Debt', 'FCFF','IFPE', 'IFEPS', 'IFROA', 'IFAGR', 'IFDebt', 'IFS', 'IFFCFF'],axis=1)
df_sorted


# In[205]:


# 按照'Stkcd'分组，并对每个分组中的'accper'进行升序排列
df_sorted = df_sorted.sort_values(['Stkcd', 'accper'], ascending=[True, True])

# 计算'price'的变化率
df_sorted['price_change_rate'] = df_sorted.groupby('Stkcd')['price'].pct_change()

# 将变化率每一项都加1
df_sorted['price_change_rate'] = df_sorted['price_change_rate'] + 1

# 乘以前一期的'IF'列的值
df_sorted['price_change_rate_if'] = df_sorted['price_change_rate'] * df_sorted.groupby('Stkcd')['IF'].shift()


# In[207]:





# In[208]:


#df14=df14.reset_index(drop = False)
df15 = pd.concat([df_sorted.reset_index(drop=True),df8r],axis = 1)
df15 = df15.loc[:, ~df15.columns.duplicated()]
mapping = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}
mapping_l = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0.25, 5: 0.5, 6: 0.75, 7: 1}
mapping_s = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1}
mapping_6 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 0}
mapping_5 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0, 7: 0}
mapping_4 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 0}
mapping_3 = {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0}
mapping_2 = {0: 0, 1: 0, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
mapping_1 = {0: 0, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
mapping_0 = {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}

df15['w'] = df15['IF2'].map(mapping)
df15['w_long'] = df15['IF2'].map(mapping_l)
df15['w_short'] = df15['IF2'].map(mapping_s)
df15['w_6'] = df15['IF2'].map(mapping_6)
df15['w_5'] = df15['IF2'].map(mapping_5)
df15['w_4'] = df15['IF2'].map(mapping_4)
df15['w_3'] = df15['IF2'].map(mapping_3)
df15['w_2'] = df15['IF2'].map(mapping_2)
df15['w_1'] = df15['IF2'].map(mapping_1)
df15['w_0'] = df15['IF2'].map(mapping_0)
# 计算'price'的变化率
df15['price_change_rate'] = df15.groupby('Stkcd')['price'].pct_change()

# 将变化率每一项都加1
df15['price_change_rate'] = df15['price_change_rate'] + 1
# 乘以前一期的'IF'列的值
df15['price_change_rate_w'] = df15['price_change_rate'] * df15.groupby('Stkcd')['w'].shift()
df15['price_change_rate_w_l'] = df15['price_change_rate'] * df15.groupby('Stkcd')['w_long'].shift()
df15['price_change_rate_w_s'] = df15['price_change_rate'] * df15.groupby('Stkcd')['w_short'].shift()
df15['price_change_rate_w_6'] = df15['price_change_rate'] * df15.groupby('Stkcd')['w_6'].shift()
df15['price_change_rate_w_5'] = df15['price_change_rate'] * df15.groupby('Stkcd')['w_5'].shift()
df15['price_change_rate_w_4'] = df15['price_change_rate'] * df15.groupby('Stkcd')['w_4'].shift()
df15['price_change_rate_w_3'] = df15['price_change_rate'] * df15.groupby('Stkcd')['w_3'].shift()
df15['price_change_rate_w_2'] = df15['price_change_rate'] * df15.groupby('Stkcd')['w_2'].shift()
df15['price_change_rate_w_1'] = df15['price_change_rate'] * df15.groupby('Stkcd')['w_1'].shift()
df15['price_change_rate_w_0'] = df15['price_change_rate'] * df15.groupby('Stkcd')['w_0'].shift()
# 打印结果

#df_grouped = df15[df15['price_change_rate_w'] != 0].groupby('accper')['price_change_rate_w'].mean()
#df_grouped_l = df15[df15['price_change_rate_w'] != 0].groupby('accper')['price_change_rate_w_l'].mean()
#df_grouped_s = df15[df15['price_change_rate_w'] != 0].groupby('accper')['price_change_rate_w_s'].mean()
df_grouped = df15.groupby('accper').apply(lambda x: (x['price_change_rate_w'].sum()) / (x['w'].shift().sum()))
df_grouped_l = df15.groupby('accper').apply(lambda x: (x['price_change_rate_w_l'].sum()) / (x['w_long'].shift().sum()))
df_grouped_s = df15.groupby('accper').apply(lambda x: ((x['price_change_rate_w_s']).sum()) / (x['w_short'].shift().sum()))
df_grouped_6 = df15.groupby('accper').apply(lambda x: ((x['price_change_rate_w_6']).sum()) / (x['w_6'].shift().sum()))
df_grouped_5 = df15.groupby('accper').apply(lambda x: ((x['price_change_rate_w_5']).sum()) / (x['w_5'].shift().sum()))
df_grouped_4 = df15.groupby('accper').apply(lambda x: ((x['price_change_rate_w_4']).sum()) / (x['w_4'].shift().sum()))
df_grouped_3 = df15.groupby('accper').apply(lambda x: ((x['price_change_rate_w_3']).sum()) / (x['w_3'].shift().sum()))
df_grouped_2 = df15.groupby('accper').apply(lambda x: ((x['price_change_rate_w_2']).sum()) / (x['w_2'].shift().sum()))
df_grouped_1 = df15.groupby('accper').apply(lambda x: ((x['price_change_rate_w_1']).sum()) / (x['w_1'].shift().sum()))
df_grouped_0 = df15.groupby('accper').apply(lambda x: ((x['price_change_rate_w_0']).sum()) / (x['w_0'].shift().sum()))
df15.to_csv(r'C:\Users\13242\Desktop\what.csv', index=False, encoding='utf-8-sig')

df15


# In[209]:


df12r = pd.DataFrame([2447.31, 2158.66, 4749.89, 3156.75, 3439.75, 3756.88, 3010.65, 3913.21, 5123.49, 4016.24])
df12rr = pd.DataFrame([3367.77,3766.17 ,8469.67 ,5985.59,6211.86 ,5860.98 ,5307.56 ,5353.86 ,6485.71 , 5627.9])
df13 = pd.concat([df12r, df12rr], axis=1)
df13.columns = ['000300','000906']
df13['000300_pct_change'] = df13['000300'] / df13['000300'].shift(1)
df13['000906_pct_change'] = df13['000906'] / df13['000906'].shift(1)
df13 = df13.drop([ '000300','000906'],axis = 1)
df13.index = range(2013, 2023)
dff=pd.concat([df_grouped,df_grouped_l,df_grouped_s,df_grouped_6,df_grouped_5,df_grouped_4,df_grouped_3,df_grouped_2,df_grouped_1,df_grouped_0],axis = 1)
dff.index = range(2013, 2023)
dff = pd.concat([dff,df13],axis = 1)
dff.replace(0, np.nan, inplace=True)
dff.columns = ['All','x','7','6','5','4','3','2','1','0','沪深300', '中证500']
dff = dff.drop(['x'],axis = 1)
df = dff.cumprod()
dff


# In[216]:


# 设置 Seaborn 风格和颜色循环
sns.set(style='ticks', palette='colorblind')
#plt.figure(figsize=(16, 9))
plt.rcParams['font.sans-serif'] = 'SimHei'
x = df.index  # 横坐标为第一列
y1 = df['All']  # 第二列数据
y2 = df['沪深300']  # 第三列数据
y3 = df['中证500']  # 第四列数据

# 创建画布和子图
fig, ax = plt.subplots(figsize=(16, 9))

# 绘制折线图
ax.plot(x, y1, label='All', linewidth=4.0)
ax.plot(x, y2, label='沪深300', linewidth=4.0)

# 添加标识
ax.legend(loc='upper left')
# 创建第二个子图




# 显示图表
plt.show()


# In[217]:


sns.set(style='ticks', palette='colorblind')
#plt.figure(figsize=(16, 9))
plt.rcParams['font.sans-serif'] = 'SimHei'
x = df.index[1:]  # 横坐标为第一列
y1 = dff['7'][1:]  # 第二列数据
y2 = dff['沪深300'][1:]  # 第五列数据
y3 = df['7'][1:]
y4 = df['沪深300'][1:]
# 创建画布和子图
fig, ax = plt.subplots(figsize=(16, 9))

# 绘制折线图
ax.plot(x, y4, label='沪深300', linewidth=4.0)
ax.plot(x, y3, label='7', linewidth=4.0)
plt.grid(True, color='gray', linestyle='--', linewidth=0.5)

# 添加标识
ax.legend(loc='upper left')


plt.show()


# In[213]:


sns.set(style='ticks', palette='colorblind')
# 定义颜色值
colors = ['#e6f4ff', '#bae0ff', '#91caff', '#69b1ff', '#4096ff', '#1677ff', '#0958d9', '#003eb3', '#002c8c', '#001d66']
colors = ['#001d66', '#002c8c', '#003eb3', '#0958d9', '#1677ff', '#4096ff', '#69b1ff', '#91caff', '#bae0ff', '#e6f4ff']

# 创建配色方案
custom_palette = sns.color_palette(colors)

# 设置颜色样式为自定义的配色方案
sns.set_palette(custom_palette)
#plt.figure(figsize=(16, 9))
plt.rcParams['font.sans-serif'] = 'SimHei'
x = df.index[1:]  # 横坐标为第一列
y1 = df['7'][1:]  # 第二列数据
y2 = df['6'][1:]  # 第五列数据
y3 = df['5'][1:]
y4 = df['4'][1:]
y5 = df['3'][1:]
y6 = df['2'][1:]
y7 = df['1'][1:]
y8 = df['0'][1:]

# 创建画布和子图
fig, ax = plt.subplots(figsize=(16, 9))

# 绘制折线图
ax.plot(x, y1, label='7', linewidth=4.0)
ax.plot(x, y2, label='6', linewidth=4.0)
ax.plot(x, y3, label='5', linewidth=4.0)
ax.plot(x, y4, label='4', linewidth=4.0)
ax.plot(x, y5, label='3', linewidth=4.0)
ax.plot(x, y6, label='2', linewidth=4.0)
ax.plot(x, y7, label='1', linewidth=4.0)
ax.plot(x, y8, label='0', linewidth=4.0)

# 添加标识
ax.legend(loc='upper left')


# In[214]:


sns.set(style='ticks', palette='colorblind')
# 定义颜色值
colors = ['#e6f4ff', '#bae0ff', '#91caff', '#69b1ff', '#4096ff', '#1677ff', '#0958d9', '#003eb3', '#002c8c', '#001d66']
colors = ['#001d66', '#91caff', '#bae0ff', '#e6f4ff']
# 创建配色方案
custom_palette = sns.color_palette(colors)

# 设置颜色样式为自定义的配色方案
sns.set_palette(custom_palette)
#plt.figure(figsize=(16, 9))
plt.rcParams['font.sans-serif'] = 'SimHei'
x = df.index[1:]  # 横坐标为第一列
y1 = df['7'][1:]  # 第二列数据
y2 = df['6'][1:]  # 第五列数据
y3 = df['5'][1:]
y4 = df['4'][1:]
y5 = df['3'][1:]
y6 = df['2'][1:]
y7 = df['1'][1:]
y8 = df['0'][1:]

# 创建画布和子图
fig, ax = plt.subplots(figsize=(16, 9))

# 绘制折线图
ax.plot(x, y1, label='7', linewidth=4.0)
'''ax.plot(x, y2, label='6', linewidth=4.0)
ax.plot(x, y3, label='5', linewidth=4.0)
ax.plot(x, y4, label='4', linewidth=4.0)
ax.plot(x, y5, label='3', linewidth=4.0)
ax.plot(x, y6, label='2', linewidth=4.0)
ax.plot(x, y7, label='1', linewidth=4.0)'''
ax.plot(x, y8, label='0', linewidth=4.0)

# 添加标识
ax.legend(loc='upper left')


# In[ ]:





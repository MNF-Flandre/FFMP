#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import datetime
import os
import glob
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import time


# In[ ]:


total_conclusion = pd.DataFrame()
for strategy_index in range(0,8):
    time_start = time.time()
    path = r'D:/Course/大三下/家庭理财实务/stk_price'
    start_str = 'TRD_BwardQuotation'
    end_str = '.csv'
    file_path = os.path.join(path, f'{start_str}*{end_str}')
    csv_files = glob.glob(file_path)
    df = pd.DataFrame()
    for file_path in csv_files:
        df1 = pd.read_csv(file_path , encoding = 'utf-8' )
        df = pd.concat([ df , df1 ] , axis = 0)
    start_date = '2013-04-30'
    end_date = '2022-04-30'
    df['TradingDate'] = pd.to_datetime(df['TradingDate'], format='%Y-%m-%d')
    df = df[(df['TradingDate'] >= start_date) & (df['TradingDate'] <= end_date)]
    df['Symbol'] = df['Symbol'].astype(str).str.zfill(6)

    print('Read price data successfully!')

    df.set_index(['TradingDate','Symbol'],inplace = True)
    df.sort_index(inplace= True)
    df.reset_index(inplace = True)
    df.dropna(inplace = True)
    grouped = df.groupby('Symbol')
    df['netvalue'] = grouped['ClosePrice'].pct_change()
    df['netvalue'] = df['netvalue'].fillna(0)
    #df['netvalue'] = (df['netvalue'] + 1).groupby(df['Symbol']).cumprod()#净值计算法
    df['netvalue'] = df['netvalue'] + 1#变动率计算法
    df.set_index(['Symbol','TradingDate'],inplace = True)
    df1 = pd.read_csv(r'C:\Users\13242\Desktop\what.csv')
    df1 = df1[~(df1['IF2'] != strategy_index)]#修改策略参数

    df1['Stkcd'] = df1['Stkcd'].astype(str).str.zfill(6)

    print('Read strategy data successfully!')

    df1 = df1[['Stkcd','accper','price']]
    df1['accper'] = pd.to_datetime(df1['accper'], format='%Y').dt.strftime('%Y')
    df1.set_index(['accper','Stkcd'] , inplace = True)
    df1.sort_index(inplace = True)
    df1.reset_index(inplace = True)
    stklist = list(df1['Stkcd'].drop_duplicates())
    df.reset_index(inplace = True)
    dfa = df[df['Symbol'].isin(stklist)]
    dfa.reset_index(inplace =True , drop = True)
    df_result = pd.DataFrame(columns=['Date','ret', 'NetValue'])
    df_result['Date'] = df['TradingDate'].drop_duplicates()
    df_result['ret'] = 1
    df_result['NetValue'] = 1
    df_result.reset_index( inplace =True , drop = True)
    df_result['Date'] =  pd.to_datetime(df_result['Date'])
    for idx,row in df_result.iterrows():
        if (row['Date'].month >= 5) or ((row['Date'].month == 4) and row['Date'].day == 30) :
            df_result.loc[idx,'time_if'] = row['Date'].year
        else:
            df_result.loc[idx,'time_if'] = row['Date'].year - 1
    dfr = pd.DataFrame()
    for index, row in df_result.iterrows():
        stk = df1.loc[df1['accper'] == str(int(row['time_if'])), 'Stkcd']
        stk = pd.DataFrame(stk)
        stk.reset_index(inplace = True,drop = True)
        stk = stk.T
        dfr = pd.concat([dfr , stk] , axis = 0)
    dfr.reset_index(inplace = True)
    df_result = pd.concat([df_result , dfr] ,axis = 1)
    df_result.set_index('Date',inplace =True)
    df_result.drop(['NetValue','time_if'],axis = 1 , inplace = True)
    df = dfa.copy()
    df.rename(columns={'TradingDate': 'Date'}, inplace=True)
    df.rename(columns={'netvalue': 'ret'}, inplace=True)
    df.set_index(['Date','Symbol'],inplace = True)
    df.drop(['ClosePrice'],axis = 1 , inplace = True)
    df_stk =  df_result.copy() 
    df_stk.drop(['ret','index'],axis = 1 , inplace = True)
    df_ret = df_stk.copy() 
    for idx, row in df_stk.iterrows():
        for col in df_stk.columns:
            if pd.isna(row[col]) == False:
                try:
                    stk = row[col]
                    ret = df.loc[idx,stk].item()
                except:
                    pass
                df_ret.loc[idx,col] = ret
            else:
                pass
        #break
    mean_values = df_ret.mean(axis=1)
    mean_values.dropna()
    netvalue = mean_values.cumprod()
    netvalue = pd.DataFrame(netvalue)
    netvalue.rename(columns={0: ('netvalue'+str(strategy_index))}, inplace=True)
    total_conclusion = pd.concat([total_conclusion,netvalue] , axis = 1)
    time_end = time.time()
    print('strategy',strategy_index ,'time cost', time_end - time_start, 's')


# In[ ]:


total_conclusion


# In[ ]:


strategy = pd.DataFrame()
strategy['low'] = (total_conclusion['netvalue0']+total_conclusion['netvalue1']+total_conclusion['netvalue2']+total_conclusion['netvalue3'])/4
strategy['high'] = (total_conclusion['netvalue4']+total_conclusion['netvalue5']+total_conclusion['netvalue6']+total_conclusion['netvalue7'])/4
strategy


# In[ ]:


path = r'D:\Course\大三下\家庭理财实务\000300'
start_str = 'IDX_Idxtrd'
end_str = '.csv'
file_path = os.path.join(path, f'{start_str}*{end_str}')
csv_files = glob.glob(file_path)
df_hs300 = pd.DataFrame()
for file_path in csv_files:
    df300 = pd.read_csv(file_path , encoding = 'utf-8' )
    df_hs300 = pd.concat([ df_hs300 , df300 ] , axis = 0)
df_hs300.rename(columns={'Idxtrd01': 'Date'}, inplace=True)
df_hs300['Date'] = pd.to_datetime(df_hs300['Date'])

df_hs300.set_index('Date',inplace = True , drop = True)
df_hs300.sort_index(inplace = True)
df_hs300['000300'] = df_hs300['Idxtrd05']/df_hs300['Idxtrd05'].shift()
df_hs300['000300'] = df_hs300['000300'].cumprod()
df_hs300.drop(['Indexcd','Idxtrd05'], inplace = True , axis = 1)


# In[ ]:


df_hs300


# In[ ]:


total_conclusion1 = pd.merge(total_conclusion , df_hs300 , on = 'Date')
total_conclusion1


# In[ ]:


year_conclusion = pd.DataFrame()
for idx,row in total_conclusion1.iterrows():
    if (idx.month == 5) and (idx.day == 2) and (idx.year == 2013):
        row = pd.DataFrame(row).T
        year_conclusion = pd.concat([year_conclusion , row])
        row_cache = copy.deepcopy(row)
        idx_cache = copy.deepcopy(idx)
        continue
    if (idx.month == 5) and (idx_cache.month == 4) and (idx.year != 2013):
        row_cache = pd.DataFrame(row).T
        year_conclusion = pd.concat([year_conclusion , row_cache])
    elif (idx.month == 4) and (idx.day == 29) and (idx.year == 2022):
        row = pd.DataFrame(row).T
        year_conclusion = pd.concat([year_conclusion , row])
        row_cache = copy.deepcopy(row)
        idx_cache = copy.deepcopy(idx)
        continue
    else:
        pass
    row_cache = copy.deepcopy(row)
    idx_cache = copy.deepcopy(idx)
year_conclusion.fillna(1,inplace = True)
year_conclusion


# In[ ]:


sns.set(style='ticks', palette='colorblind')
colors = [ '#003eb3', '#91caff']
custom_palette = sns.color_palette(colors)
sns.set_palette(custom_palette)
plt.rcParams['font.sans-serif'] = 'SimHei'
x = year_conclusion.index  # 横坐标为第一列
y1 = year_conclusion['netvalue7']  # 第二列数据
y2 = year_conclusion['000300']
# 创建画布和子图
fig, ax = plt.subplots(figsize=(16, 9))

# 
ax.plot(x, y1, label='strategy', linewidth=4.0)
ax.plot(x, y2, label='000300', linewidth=4.0)

ax.legend(loc='upper left')

plt.show()


# In[ ]:


sns.set(style='ticks', palette='colorblind')
colors = [ '#003eb3', '#91caff']
custom_palette = sns.color_palette(colors)
sns.set_palette(custom_palette)
plt.rcParams['font.sans-serif'] = 'SimHei'
x = total_conclusion1.index  # 横坐标为第一列
y1 = total_conclusion1['netvalue7']  # 第二列数据
y2 = total_conclusion1['000300']
# 创建画布和子图
fig, ax = plt.subplots(figsize=(16, 9))

# 
ax.plot(x, y1, label='strategy', linewidth=4.0)
ax.plot(x, y2, label='000300', linewidth=4.0)

ax.legend(loc='upper left')

plt.show()


# In[ ]:


sns.set(style='ticks', palette='colorblind')
# 定义颜色值
colors = ['#e6f4ff', '#bae0ff', '#91caff', '#69b1ff', '#4096ff', '#1677ff', '#0958d9', '#003eb3', '#002c8c', '#001d66']
colors = ['#001d66', '#002c8c', '#003eb3', '#0958d9', '#1677ff', '#4096ff', '#69b1ff', '#91caff', '#bae0ff', '#e6f4ff']

# 创建配色方案
custom_palette = sns.color_palette(colors)

# 设置颜色样式为自定义的配色方案
sns.set_palette(custom_palette)
plt.rcParams['font.sans-serif'] = 'SimHei'
x = total_conclusion.index#[1:]  # 横坐标为第一列
y1 = total_conclusion['netvalue7']#[1:]  # 第二列数据
y2 = total_conclusion['netvalue6']#[1:]  # 第五列数据
y3 = total_conclusion['netvalue5']#[1:]
y4 = total_conclusion['netvalue4']#[1:]
y5 = total_conclusion['netvalue3']#[1:]
y6 = total_conclusion['netvalue2']#[1:]
y7 = total_conclusion['netvalue1']#[1:]
y8 = total_conclusion['netvalue0']#[1:]

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


# In[ ]:


sns.set(style='ticks', palette='colorblind')
# 定义颜色值
colors = ['#e6f4ff', '#bae0ff', '#91caff', '#69b1ff', '#4096ff', '#1677ff', '#0958d9', '#003eb3', '#002c8c', '#001d66']
colors = ['#001d66', '#002c8c', '#003eb3', '#0958d9', '#1677ff', '#4096ff', '#69b1ff', '#91caff', '#bae0ff', '#e6f4ff']

# 创建配色方案
custom_palette = sns.color_palette(colors)

# 设置颜色样式为自定义的配色方案
sns.set_palette(custom_palette)
plt.rcParams['font.sans-serif'] = 'SimHei'
x = year_conclusion.index#[1:]  # 横坐标为第一列
y1 = year_conclusion['netvalue7']#[1:]  # 第二列数据
y2 = year_conclusion['netvalue6']#[1:]  # 第五列数据
y3 = year_conclusion['netvalue5']#[1:]
y4 = year_conclusion['netvalue4']#[1:]
y5 = year_conclusion['netvalue3']#[1:]
y6 = year_conclusion['netvalue2']#[1:]
y7 = year_conclusion['netvalue1']#[1:]
y8 = year_conclusion['netvalue0']#[1:]

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


# In[ ]:


sns.set(style='ticks', palette='colorblind')
# 定义颜色值
colors = ['#e6f4ff', '#bae0ff', '#91caff', '#69b1ff', '#4096ff', '#1677ff', '#0958d9', '#003eb3', '#002c8c', '#001d66']
colors = [ '#003eb3', '#91caff']

# 创建配色方案
custom_palette = sns.color_palette(colors)

# 设置颜色样式为自定义的配色方案
sns.set_palette(custom_palette)
plt.rcParams['font.sans-serif'] = 'SimHei'
x = total_conclusion.index
y1 = total_conclusion['netvalue7']
y8 = total_conclusion['netvalue0']
fig, ax = plt.subplots(figsize=(16, 9))

ax.plot(x, y1, label='7', linewidth=4.0)

ax.plot(x, y8, label='0', linewidth=4.0)

# 添加标识
ax.legend(loc='upper left')


# In[ ]:


sns.set(style='ticks', palette='colorblind')
# 定义颜色值
colors = ['#e6f4ff', '#bae0ff', '#91caff', '#69b1ff', '#4096ff', '#1677ff', '#0958d9', '#003eb3', '#002c8c', '#001d66']
colors = [ '#003eb3', '#91caff']

# 创建配色方案
custom_palette = sns.color_palette(colors)

# 设置颜色样式为自定义的配色方案
sns.set_palette(custom_palette)
plt.rcParams['font.sans-serif'] = 'SimHei'
x = year_conclusion.index
y1 = year_conclusion['netvalue7']
y8 = year_conclusion['netvalue0']
fig, ax = plt.subplots(figsize=(16, 9))

ax.plot(x, y1, label='7', linewidth=4.0)

ax.plot(x, y8, label='0', linewidth=4.0)

# 添加标识
ax.legend(loc='upper left')


# In[ ]:





# In[ ]:





# In[ ]:





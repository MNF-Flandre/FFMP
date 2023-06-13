#!/usr/bin/env python
# coding: utf-8

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




'''
策略名称：
单因子日线交易策略
策略流程：
盘前将中小板成分股中st、停牌、退市的股票过滤得到股票池
盘中：
1、通过极值处理、标准化处理、市值中性化处理
2、因子排序获得股票池
3、动态平衡仓位
'''
import pandas as pd
import numpy as np
import statsmodels.api as sm
import time
import math
from decimal import Decimal


#初始化处理
def initialize(context):
    g.factor = 'roe'
    g.factor_parms_info = {'roe':['profit_ability', 'roe' , False], #净资产收益率,最后布尔值为排序方式
                    'operating_revenue_grow_rate':['growth_ability', 'operating_revenue_grow_rate' , False], #营收增速
                    'np_parent_company_cut_yoy':['growth_ability', 'np_parent_company_cut_yoy' , False], #扣非净利润增速
                    }
    # 初始化此策略
    # 设置我们要操作的股票池, 这里我们只操作一支股票
    # g.security = '600570.SS'
    # g.security = get_index_stocks('000300.SS')
    # set_universe(g.security)
    set_params()        #设置策参数
    set_variables()     #设置中间变量
    is_trade_flag = is_trade()
    if is_trade_flag:
        pass
    else:
        set_backtest()      #设置回测条件

#设置策参数
def set_params():
    g.tc=15  # 调仓频率
    g.yb=63  # 样本长度
    g.N=20   # 持仓数目
    g.NoF=3  # 三因子模型
    g.precent = 0.10
    
#设置中间变量
def set_variables():
    g.t=0               #记录连续回测天数
    g.rf=0.04           #无风险利率
    g.if_trade=False    #当天是否交易

#设置回测条件
def set_backtest():
    set_limit_mode('UNLIMITED')


#盘前处理
def before_trading_start(context, data):
    g.everyStock = 0
    g.rf = 0.04
    g.df = pd.DataFrame()
    g.current_date = context.blotter.current_dt.strftime("%Y%m%d")
    # g.all_stocks = get_index_stocks('000906.XBHS', g.current_date)
    g.all_stocks = get_index_stocks('000300.XBHS', g.current_date)
    if g.t%g.tc==0:
        #每g.tc天，交易一次行
        g.if_trade=True 
        #获取股票的状态ST、停牌、退市
        st_status = get_stock_status(g.all_stocks, 'ST')
        halt_status = get_stock_status(g.all_stocks, 'HALT')
        delisting_status = get_stock_status(g.all_stocks, 'DELISTING')
        #将三种状态的股票剔除当日的股票池
        for stock in g.all_stocks.copy():
            if st_status[stock] or halt_status[stock] or delisting_status[stock]:
                g.all_stocks.remove(stock)  
    g.t+=1


#每天交易时要做的事情
def handle_data(context, data):
    for stock in g.all_stocks.copy():
        if stock[:3] == '688':
            g.all_stocks.remove(stock)
    if g.if_trade==True:
        count = 0
        flag = False
        #获取财务数据尝试5次
        while count < 5:
            #获取因子值前一组的股票池
            if get_stocks(g.all_stocks, str(get_trading_day(-1)), 'roe'):
                log.info('本次获取财务数据成功')
                flag = True
                break
            else:
                count +=1
                time.sleep(60)
        if flag:
            pass
        else:
            return
            log.info('本次获取财务数据不成功，请检查数据，本次换仓不进行')
            
        stock_sort = g.stocks
        #把涨停状态的股票剔除
        up_limit_stock = get_limit_stock(context, stock_sort)['up_limit']
        stock_sort = list(set(stock_sort)-set(up_limit_stock))
        position_last_map = [
                position.sid
                for position in context.portfolio.positions.values()
                if position.amount != 0
            ]
        #持仓中跌停的股票不做卖出
        limit_info = get_limit_stock(context, position_last_map)
        hold_down_limit_stock = limit_info['down_limit']
        log.info('持仓跌停股：%s'%hold_down_limit_stock)   
        position_last_map = [
                position.sid
                for position in context.portfolio.positions.values()
                if position.amount != 0
            ]
        #持仓中除了不处于前g.N且跌停不能卖的股票进行卖出
        sell_stocks = list(set(position_last_map)-set(stock_sort[:g.N])-set(hold_down_limit_stock))
        #对不在换仓列表中且飞跌停股的股票进行卖出操作
        order_stock_sell(context, data, sell_stocks)
        #获取仍在持仓中的股票
        position_last_map = [
                position.sid
                for position in context.portfolio.positions.values()
                if position.amount != 0
            ]
        #获取调仓买入的股票
        buy_stocks = list(set(stock_sort)-set(position_last_map))[:(g.N-len(position_last_map))]
        #仓位动态平衡的股票
        balance_stocks = list(set(buy_stocks+position_last_map)-set(hold_down_limit_stock))
        log.info('balance_stocks%s'%len(balance_stocks))
        g.everyStock = context.portfolio.portfolio_value/g.N
        log.info('g.everyStock%s'%g.everyStock)    
        order_stock_balance(context, data, balance_stocks)   
        order_stock_balance(context, data, balance_stocks)            
    g.if_trade=False


#不在换仓目标中且没有跌停的股票进行清仓操作    
def order_stock_sell(context, data, sell_stocks):
    # 对于不需要持仓的股票，全仓卖出
    for stock in sell_stocks:
        stock_sell = stock
        order_target_value(stock_sell, 0)


#非跌停的换仓目标股进行仓位再平衡
def order_stock_balance(context, data, balance_stocks):
    for stock in balance_stocks:
        order_target_value(stock, g.everyStock)


#获取拟持仓股票池
def get_stocks(stocks, date, factor):
    try:
        factor = g.factor 
        sort_type = g.factor_parms_info[factor][-1]
        df = get_factor_values(stocks, factor, date, g.factor_parms_info)
        df.dropna(inplace=True)
        #3倍标准差去极值
        df = winsorize(df, factor, std=3, have_negative=True)
        #z标准化
        df = standardize(df, factor, ty=2)
        #市值中性化
        market_cap_df = get_fundamentals(stocks, 'valuation', fields='total_value', date=date)
        market_cap_df = market_cap_df[['total_value']]
        df = neutralization(df, factor ,market_cap_df)
        df = df.sort_values(by=factor, ascending = sort_type)
        g.stocks = list(df.head(int(len(df)*g.precent)).index)
        return True
    except:
        return False


#获取因子值
def get_factor_values(stock_list, factor, date, factor_parms_info):
    '''
    获取因子值方法
    入参：
    1、股票池：stock_list
    2、因子名称：factor
    3、计算日期：date
    4、因子数据获取需要维护的信息（因子名称、表名、字段名）
    '''
    year = str(date[:4])
    last_year = str(int(year)-1)
    last_2_year = str(int(last_year)-1)
    
    df1 = get_fundamentals(stock_list, table=factor_parms_info[factor][0], fields=factor_parms_info[factor][1], start_year=last_year, end_year=year, report_types='1')
    df2 = get_fundamentals(stock_list, table=factor_parms_info[factor][0], fields=factor_parms_info[factor][1], start_year=last_year, end_year=year, report_types='2')
    df3 = get_fundamentals(stock_list, table=factor_parms_info[factor][0], fields=factor_parms_info[factor][1], start_year=last_year, end_year=year, report_types='3')
    df4 = get_fundamentals(stock_list, table=factor_parms_info[factor][0], fields=factor_parms_info[factor][1], start_year=last_year, end_year=year, report_types='4')

    df11 = get_fundamentals(stock_list, table=factor_parms_info[factor][0], fields=factor_parms_info[factor][1], start_year=last_2_year, end_year=last_year, report_types='1')
    df22 = get_fundamentals(stock_list, table=factor_parms_info[factor][0], fields=factor_parms_info[factor][1], start_year=last_2_year, end_year=last_year, report_types='2')
    df33 = get_fundamentals(stock_list, table=factor_parms_info[factor][0], fields=factor_parms_info[factor][1], start_year=last_2_year, end_year=last_year, report_types='3')
    df44 = get_fundamentals(stock_list, table=factor_parms_info[factor][0], fields=factor_parms_info[factor][1], start_year=last_2_year, end_year=last_year, report_types='4')
    s = 0
    factor_info = {}
    for stock in stock_list.copy():
        s+=1
        if stock in df1.items:
            data1 = df1[stock]
        else:
            data1 = pd.DataFrame()
        if stock in df2.items:
            data2 = df2[stock]
        else:
            data2 = pd.DataFrame()
        if stock in df3.items:
            data3 = df3[stock]
        else:
            data3 = pd.DataFrame()
        if stock in df4.items:
            data4 = df4[stock]
        else:
            data4 = pd.DataFrame()
        if stock in df11.items:
            data11 = df11[stock]
        else:
            data11 = pd.DataFrame()
        if stock in df22.items:
            data22 = df22[stock]
        else:
            data22 = pd.DataFrame()
        if stock in df33.items:
            data33 = df33[stock]
        else:
            data33 = pd.DataFrame()
        if stock in df44.items:
            data44 = df44[stock]
        else:
            data44 = pd.DataFrame()
        dict1 = {}
        parms_info = {'Q1':data1, 'Q2':data2, 'Q3':data3, 'Q4':data4}
        for i in ['Q1','Q2','Q3','Q4']:
            dict1[i] = {}
            for j in ['publ_date', 'end_date', factor_parms_info[factor][1]]:
                if j != 'end_date':
                    if parms_info[i].empty:
                        dict1[i][j] = ''
                    else:
                        dict1[i][j] = parms_info[i][j][-1]
                else:
                    if parms_info[i].empty:
                        dict1[i][j] = ''
                    else:                        
                        dict1[i][j] = list(parms_info[i].index)[-1]
        dict2 = {}
        parms_info2 = {'Q1':data11, 'Q2':data22, 'Q3':data33, 'Q4':data44}
        for i in ['Q1','Q2','Q3','Q4']:
            dict2[i] = {}
            for j in ['publ_date', 'end_date', factor_parms_info[factor][1]]:
                if j != 'end_date':
                    if parms_info2[i].empty:
                        dict2[i][j] = ''
                    else:
                        dict2[i][j] = parms_info2[i][j][-1]
                else:
                    if parms_info2[i].empty:
                        dict2[i][j] = ''
                    else:                        
                        dict2[i][j] = list(parms_info2[i].index)[-1]
        
        df_1 = pd.DataFrame.from_dict(dict1, orient='index')
        df_2 = pd.DataFrame.from_dict(dict2, orient='index')
        df_3 = pd.concat([df_1,df_2])
        df_3 = df_3[df_3['publ_date']<date]
        max_publ_date = max(df_3['publ_date'].values)
        df_3 = df_3[df_3['publ_date']==max_publ_date]
        df_4 = df_3.sort_values(by='end_date')
        df_4.drop_duplicates(subset=None, keep='last', inplace=True)
        x = df_4[factor_parms_info[factor][1]].tail(1).values[0]
        factor_info[stock] = x
    factor_df = pd.DataFrame.from_dict(factor_info, orient='index')
    factor_df.columns = [factor_parms_info[factor][1]]
    return factor_df
          

#保留小数点两位
def replace(x):
    y = Decimal(x)
    y = float(str(round(x, 2)))  
    return y          
    

#生成昨日持仓股票列表
def position_last_close_init(context):
    g.position_last_map = [
        position.sid
        for position in context.portfolio.positions.values()
        if position.amount != 0
    ]    
    

#日级别回测获取持仓中不能卖出的股票(涨停就不卖出)
def get_limit_stock(context, stock_list):
    out_info = {'up_limit':[], 'down_limit':[]}
    history = get_history(5, '1d', ['close','volume'], stock_list, fq='dypre', include=True)
    history = history.swapaxes("minor_axis", "items") 
    def get_limit_rate(stock):
        rate = 0.1
        if stock[:2] == '68':
            rate = 0.2
        elif stock[0] == '3':
            rate = 0.2
        return rate
    for stock in stock_list:
        #log.info(stock)
        df = history[stock]
        #过滤历史停牌的数据    
        df = df[df['volume']>0] 
        if len(df.index)<2:   
            continue          
        last_close = df['close'].values[:][-2]
        curr_price = df['close'].values[:][-1]
        rate = get_limit_rate(stock)
        up_limit_price = last_close*(1+rate)
        up_limit_price = replace(up_limit_price)
        if curr_price >= up_limit_price:
            out_info['up_limit'].append(stock)
        down_limit_price = last_close*(1-rate)
        down_limit_price = replace(down_limit_price)
        if curr_price <= down_limit_price:
            out_info['down_limit'].append(stock)                    
    return out_info
    

# 去极值函数（3倍标准差去极值）
def winsorize(factor_data, factor, std=3, have_negative=True):
    '''
    去极值函数 
    factor:以股票code为index，因子值为value的Series
    std为几倍的标准差，have_negative 为布尔值，是否包括负值
    输出Series
    '''
    r=factor_data[factor]
    if have_negative == False:
        r = r[r>=0]
    else:
        pass
    #取极值
    edge_up = r.mean()+std*r.std()
    edge_low = r.mean()-std*r.std()
    r[r>edge_up] = edge_up
    r[r<edge_low] = edge_low
    r = pd.DataFrame(r)
    return r


# z－score标准化函数：
def standardize(factor_data, factor, ty=2):
    '''
    s为Series数据
    ty为标准化类型:1 MinMax,2 Standard,3 maxabs 
    '''
    temp=factor_data[factor]
    if int(ty)==1:
        re = (temp - temp.min())/(temp.max() - temp.min())
    elif ty==2:
        re = (temp - temp.mean())/temp.std()
    elif ty==3:
        re = temp/10**np.ceil(np.log10(temp.abs().max()))
    return pd.DataFrame(re)


# 市值中性化函数
def neutralization(data_factor, factor, data_market_cap):
    data_market_cap['total_value2'] = 0
    data_market_cap['total_value2'] = data_market_cap['total_value'].apply(lambda x:math.log(x))
    df = pd.concat([data_factor,data_market_cap],axis=1,join='inner')
    y = df[factor]
    x = df['total_value2']
    result = sm.OLS(y,x).fit()
    result = pd.DataFrame(result.resid)
    result.columns = [g.factor]
    return result
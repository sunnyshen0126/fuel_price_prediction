README
===========================

该文件用于说明代码的作用及运行方法

****
	
|作者|沈偲阳|
|---|---
|学号|120120911057
|课题|燃油价格预测及基于预测的采购策略设计


****
## 目录
* [内容](#内容)
    * 数据获取
    * 数据处理
    * 新闻情感分析
    * 长短期预测
    * 采购策略设计

------
# 1.数据获取

## 1.1 燃油相关数据获取
-->oil_price.xlsx 获取方式：从Wind获取

从Wind中获取原油进口数量、波罗的海干散货指数（BDI）、CTFI综合指数、CCFI综合指数、BDTI原油运输指数、BCTI成品油运输指数、全国主要港口货物吞吐量、WTI轻质原油价格、美元指数、美十年期国债利率等与燃油价格相关的数据。

## 1.2 燃油相关新闻
从燃油相关网站获取

a. get_data/get_news.py get_site() 获取具体的网站地址 --> get_data/website.xlsx

b. get_data/get_news.py get_content() 获取对应网站的标题、日期以及内容  --> get_data/content.xlsx

c. get_data/get_news.py separate_word() 对于新闻的标题进行分类，并选择其中的动词、形容词  --> get_data/separate.xlsx

------
# 2.数据处理

## 2.1 燃油的历史涨跌幅处理

计算新加坡0.5%低硫燃料油历史数据一日、三日、五日、十日、二十日、六十日的对数收益率数据

get_data/process_data.py get_price_return()v--> get_data/Singapore.xlsx

# 2.2 将燃油相关新闻与历史涨跌幅数据对应
由于有些新闻并非在节假日公布，需要将这些新闻与最近的一个交易日相关联。这一步的目的是后续直接用收益率对新闻做标注。

get_data/get_close_date.py  get_close_date() --> get_data/all.xlsx

------
# 3.新闻情感分析

## 3.1 线性代数方法（动词和形容词情感分析）
![image](https://github.com/sunnyshen0126/fuel_price_prediction/blob/main/ScreenShots/1.png)

a. sentiment/linear_sentiment.py get_vocab() 获得新闻标题中出现的词汇并统计其出现的次数（动词和形容词） -->sentiment/vocab.xlsx

b. sentiment/linear_sentiment.py calculate_vocab_contribution() 计算每个词汇对于收益率的贡献（训练集为2019～2021的数据）-->sentiment/v_2021.xlsx

c. sentiment/linear-sentiment.py score1() 

按照上图中的线性代数方法按照分别按照一日、三日、五日、十日、二十日、六十日计算新闻情绪因子，分别记为s1,s2,...,s6。 --> sentiment/all.xlsx

## 3.2 矩阵乘法方法

![image](https://github.com/sunnyshen0126/fuel_price_prediction/blob/main/ScreenShots/2.png)

a. sentiment/matrix_sentiment.py calculate_matrix() 获得新闻标题中每个词汇对于收益率的贡献（训练集为2019～2021的数据）-->sentiment/vocab2.xlsx

b. sentiment/matrix_sentiment.py matrix_score()

按照上图中的矩阵乘法方法按照分别按照一日、三日、五日、十日、二十日、六十日计算新闻情绪因子，分别记为m1,m2,...,m6。 --> sentiment/all.xlsx

## 3.3 BERT情感分析

由于预训练模型的size较大，因此放在以下的百度网盘链接中。

链接:https://pan.baidu.com/s/1xpLF2i72I44YSJbSZuefYw  密码:a494

BERT情感分析可以分为数据处理、训练、测试三个部分

### 3.3.1 数据处理

bert/process_data.py 获得训练集、验证集、测试集及全部数据(train.tsv, dev.tsv, test.tsv, data.tsv), 训练集选取2019～2021的数据，将一条新闻对应的一日、三日、五日、十日、二十日、六十日收益率>0/<=0分别对应正面和负面，存储在data_r1, data_r3, data_r5, data_r10, data_r20, data_r60中，训练的结果放在output_r1, output_r3, output_r5, output_r10, output_r20, output_r60

### 3.3.2 训练
假设数据处理在data_r1中，输出在output_r1中

#### 训练方式1 （在Python Console中运行）
    export DATA_DIR = 'data_r1'
    export BERT_BASE_DIR = 'FinBERT_L-12_H-768_A-12'
    export OUT_DIR = 'output_r1'
    python run_classifier.py --task_name=policy --do_train=True --do_eval=True --do_predict=False --data_dir=$DATA_DIR/ --vocab_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=50 --train_batch_size=16 --learning_rate=2e-5 --num_train_epochs=8.0 --output_dir=$OUT_DIR

#### 训练方式2
    训练：打开run_classifier.py
    data_dir(row24): process_data.py处理后的路径（'data_r1')
    bert_config_file(row29): 'FinBERT_L-12_H-768_A-12/bert_config.json'
    vocab_file(row35): 'FinBERT_L-12_H-768_A-12/vocab.txt'
    output_dir(row39): 输出路径 ('output_r1')
    init_checkpoint(row44): 'FinBERT_L-12_H-768_A-12/bert_model.ckpt'
    do_train(row60): True 
    do_eval(row62): True 
    do_predict(row64): False

------
### 3.3.3 测试
假设数据处理在data_r1中，输出在output_r1中

#### 测试方式1 （在Python Console中运行）
    export DATA_DIR = 'data_r1'
    export BERT_BASE_DIR = 'FinBERT_L-12_H-768_A-12'
    export OUT_DIR = 'output_r1'
    python run_classifier.py --task_name=policy --do_train=False --do_eval=False --do_predict=True --data_dir=$DATA_DIR/ --vocab_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$OUT_DIR --max_seq_length=50 --output_dir=$OUT_DIR

#### 测试方式2
    测试：打开run_classifier.py
    data_dir(row24): process_data.py处理后的路径（'data_r1'）
    bert_config_file(row29): 'FinBERT_L-12_H-768_A-12/bert_config.json'
    vocab_file(row35): FinBERT_L-12_H-768_A-12/vocab.txt
    output_dir(row39): 输出路径，写成output即可 ('output_r1')
    init_checkpoint(row44): 'output_r1'
    do_train(row60): False
    do_eval(row62): True
    do_predict(row64): True

测试得到的结果存储在output_r1/test_results.tsv中

### 3.3.4 将bert因子合并
test_results.tsv中分别有一条新闻正面与负面的情绪，将两者相减得到测试的情绪评分

sentiment/bert_factor.py all_factor_to_allfile() 将一日、三日、五日、十日、二十日、六十日的因子分别记为f1, f2, ..., f6 --> sentiment/all.xlsx

## 3.4 因子处理
由于某些新闻对应不上交易日，因此要将因子对应上最接近的交易日

sentiment/bert_factor.py get_trade_date() --> sentiment/all_factor.xlsx

## 3.5 因子训练集及测试集测试
sentiment/plot.py log_return_threshold(factor, mode) 绘图

factor默认为f1, mode默认为train，画出在不同因子阈值下的绘图结果

------
# 4.长短期燃油价格预测

## 4.1 长期预测
长期预测采用ARIMA和基钦周期两个模型，由于新加坡低硫燃料油历史数据量较少，因此预测WTI轻质原油的价格

### 4.1.1 ARIMA长期预测
a. predict/predict_y_arima.py check_difference()  测试 p, d, q

b. predict/predict_y_arima.py predict_arima(PREDICT_LEN_1=12, last=0, alpha=0.05) --> predict/long_prediction_arima.xlsx

三个因子 PREDICT_LEN_1为预测长度（默认为12，即为预测未来十二个月的价格），last为从过去几个月开始预测（默认为0，即从当前时间段开始预测）,alpha为预测区间的置信度（默认0.05为95%置信度）

### 4.1.2 Kitchin周期长期预测
a. predict/predict_y_kitchin.py predict_kitchin(predict_len=12) --> predict/long_prediction_kitchin.xlsx （predict_len预测长度（默认为12，即为预测未来十二个月的价格））

b. predict/predict_y_kitchin.py predict_year_kitchin(last=0) --> predict/year_prediction_kitchin.xlsx （因子last默认为0，对从last月前开始的后十二个月的每个月做十二个月预测（用于年度采购））
对于未来一年做未来十二个月的预测

## 4.2 短期预测
短期预测采用ARIMA、ARIMAX、VAR、LSTM、BiLSTM五个模型，直接对于Singapore低硫燃料油数据，其他因子选取原油进口数量、波罗的海干散货指数（BDI）、CTFI综合指数、CCFI综合指数、BDTI原油运输指数、BCTI成品油运输指数、全国主要港口货物吞吐量、WTI轻质原油价格、美元指数、美十年期国债利率等与燃油价格相关的数据以及采用BERT训练得到的情绪因子。

### 4.2.1 ARIMA短期预测
a. predict/predict_arima.py get_difference() 插分测试，确定d

b. predict/predict_arima.py get_correlation_plot() 相关性及偏相关性测试，确定p和q的值

c. predict/predict_arima.py one_day_prediction_arima(day=250, last=0) -->predict/ARIMA_day.xlsx 对于day天的未来一日的价格做滚动预测
两个因子，last为从过去的last+day天开始预测，day为day天的滚动预测（默认为last=0, day=250，代表对于过去的250个交易日做未来一日的滚动预测）

d. predict/several_days_prediction_arima(day=5, last=0) --> predict/short_prediction_arima.xlsx 对于last天前的未来day日做预测
两个因子，last为从过去的last天开始预测， day为未来day日的价格预测结果（默认为last=0, day=5，代表对于当前交易日后的未来5个交易日燃油价格做预测）

### 4.2.2 ARIMAX短期预测
a. predict/predict_arimax.py get_difference() 插分测试，确定d

b. predict/predict_arimax.py find_ar_ma() 通过迭代确定最合适的p和q的值

c. predict/predict_arimax.py one_day_prediction_arimax(day=30, last=0) -->predict/ARIMAX_day.xlsx 对于day天的未来一日的价格做滚动预测
两个因子，last为从过去的last+day天开始预测，day为day天的滚动预测（默认为last=0, day=30，代表对于过去的30个交易日做未来一日的滚动预测）

d. predict/several_days_prediction_arimax(day=5, last=0) --> predict/short_prediction_arimax.xlsx 对于last天前的未来day日做预测
两个因子，last为从过去的last天开始预测， day为未来day日的价格预测结果（默认为last=0, day=5，代表对于当前交易日后的未来5个交易日燃油价格做预测）

e. predict/several_days_prediction_arimax(length, start)(requirements: start>=length)--> predict/five_day_prediction_arimax.xlsx 对于start天前后的length个交易日燃油价格做五日预测，主要运用于月度及日度的燃油价格采购
两个因子，对于start天前后的length个交易日燃油价格做五日预测（默认为length=60, start=60，即从过去的60天开始对未来的60个交易日做燃油价格预测，要求start>=length）

### 4.2.3 VAR短期预测
a. predict/predict_var.py check() 插分及协整检验

b. predict/predict_var.py get_lag() 定阶，通过不同模型的AIC、BIC等值实现定阶

c. predict/predict_var.py cusum() 绘制脉冲响应图

d. predict/predict_var.py variance_decom() 绘制方差分解图

e. predict/several_days_prediction_var(day=5, last=0) --> predict/short_prediction_var.xlsx 对于last天前的未来day日做预测
两个因子，last为从过去的last天开始预测， day为未来day日的价格预测结果（默认为last=0, day=5，代表对于当前交易日后的未来5个交易日燃油价格做预测）

f. predict/several_days_prediction_var(length, start)(requirements: start>=length)--> predict/five_day_prediction_var.xlsx 对于start天前后的length个交易日燃油价格做五日预测，主要运用于月度及日度的燃油价格采购
两个因子，对于start天前后的length个交易日燃油价格做五日预测（默认为length=60, start=60，即从过去的60天开始对未来的60个交易日做燃油价格预测，要求start>=length）

### 4.2.4 LSTM短期价格预测
a. predict/several_days_prediction_lstm(day, n_in, n_vars, n_neuron, n_batch, n_epoch, repeats, n_train)
--> short_prediction_lstm.xlsx 对于未来day日价格做预测
因子中day默认为5，即对于未来五个交易日价格进行预测，n_in默认为1，即在构造监督训练集时对数据做一日的偏移（n_in越大历史数据对于预测结果的影响越大，会趋向于接近历史平均价格），n_vars为变量数，默认为14，repeats为训练次数，神经网络训练并不稳定，因此将多次训练的结果取均值作为预测结果，n_train为预测的长度，默认为5日，即对未来五日的未来五天价格进行预测然后进行训练）

b. predict/five_day_prediction_lstm(length, start)  (requirements: start>=length)--> predict/five_day_prediction_lstm.xlsx 对于start天前后的length个交易日燃油价格做五日预测，主要运用于月度及日度的燃油价格采购
两个因子，对于start天前后的length个交易日燃油价格做五日预测（默认为length=30, start=30，即从过去的30天开始对未来的30个交易日做燃油价格预测，要求start>=length）(在five_day_lstm中修改参数，在fit_lstm中修改模型结构）

### 4.2.5 BiLSTM短期价格预测
a. predict/several_days_prediction_bilstm(day, n_in, n_vars, n_neuron, n_batch, n_epoch, repeats, n_train)
--> short_prediction_bilstm.xlsx 对于未来day日价格做预测
因子中day默认为5，即对于未来五个交易日价格进行预测，n_in默认为1，即在构造监督训练集时对数据做一日的偏移（n_in越大历史数据对于预测结果的影响越大，会趋向于接近历史平均价格），n_vars为变量数，默认为14，repeats为训练次数，神经网络训练并不稳定，因此将多次训练的结果取均值作为预测结果，n_train为预测的长度，默认为5日，即对未来五日的未来五天价格进行预测然后进行训练）

b.  predict/five_day_prediction_bilstm(length, start)  (requirements: start>=length)--> predict/five_day_prediction_bilstm.xlsx 对于start天前后的length个交易日燃油价格做五日预测，主要运用于月度及日度的燃油价格采购
两个因子，对于start天前后的length个交易日燃油价格做五日预测（默认为length=30, start=30，即从过去的30天开始对未来的30个交易日做燃油价格预测，要求start>=length）(在five_day_lstm中修改参数，在fit_lstm中修改模型结构）

------
# 5.燃料采购策略设计

## 5.1 月频采购方案设计
![image](https://github.com/sunnyshen0126/fuel_price_prediction/blob/main/ScreenShots/3.png)

a. buy/buy_signal.py get_year_Singapore() 将WTI原油长期预测的收益率复制到新加坡低硫燃料油上，得到燃油价格的预测结果 
b. buy/buy_singal.py year_buy_signal(last=0, storage=5, amount=10000) 得到年度的采购计划（过去一年）--> buy/year_buy_signal.xlsx
三个因子：last代表从当前月的last个月前的12个设计燃油采购策略，默认为0；storage代表每个月的燃油储存成本，默认为5美元/吨；amount为燃油需要的采购量，默认为10000份 注意：需先用4.1.2中的predict_y_kitchin模型对燃油价格进行预测。

## 5.2 未来一月燃油采购方案设计
![image](https://github.com/sunnyshen0126/fuel_price_prediction/blob/main/ScreenShots/4.png)

buy/buy_signal.py month_buy_signal(start='20220301', end='20220331', storage=0.2) 得到未来一月的燃油采购计划 --> buy/month_buy_signal.xlsx

三个因子： start和end分别代表这个月的开始和结束日，长度需要确保为一个月。 storage为0.2代表月内每日的燃油储存成本，默认为每日0.2美元/吨 

row64: 目前读取的为BiLSTM的预测结果，也可选取采用var, arima, arimax, lstm等其他模型预测的结果进行测试。注意：需先用4.2.1至4.2.5中的five_day_prediction_*** 函数对于需要采购的月份进行预测。

## 5.3 未来一周的燃油采购方案设计
![image](https://github.com/sunnyshen0126/fuel_price_prediction/blob/main/ScreenShots/5.png)

buy/buy_signal.py week_buy_signal(start=40) （start为5的倍数） 对于当前预测结果start天前开始的每周进行燃油采购方案设计，start默认为40（默认每周为5日，因此需要确保start为5的倍数） --> buy/week_buy_signal.xlsx

row64: 目前读取的为BiLSTM的预测结果，也可选取采用var, arima, arimax, lstm等其他模型预测的结果进行测试。注意：需先用4.2.1至4.2.5中的five_day_prediction_*** 函数对于需要采购的月份进行预测。



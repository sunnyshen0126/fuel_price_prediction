import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import re
import os
import pandas as pd
import jieba
import jieba.posseg as pseg
import math
import numpy as np

def get_text(url):
    requests.adapters.DEFAULT_RETRIES = 10 # 增加重连次数
    s = requests.session()
    s.keep_alive = False # 关闭多余连接
    c = s.get(url) # 你需要的网址
    return c.text


def filter_tags(str):
    re_cdata = re.compile('//<!\[CDATA\[[^>]*//\]\]>', re.I)  # 匹配CDATA
    re_script = re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I)  # Script
    re_style = re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I)  # style
    re_br = re.compile('<br\s*?/?>')  # 处理换行
    re_h = re.compile('</?\w+[^>]*>')  # HTML标签
    re_comment = re.compile('<!--[^>]*-->')  # HTML注释
    s = re_cdata.sub('', str)  # 去掉CDATA
    s = re_script.sub('', s)  # 去掉SCRIPT
    s = re_style.sub('', s)  # 去掉style
    s = re_br.sub('\n', s)  # 将br转换为换行
    s = re_h.sub('', s)  # 去掉HTML 标签
    s = re_comment.sub('', s)  # 去掉HTML注释
    blank_line = re.compile('\n+')
    s = blank_line.sub('\n', s)
    return s

def get_site():
    # 获取网站地址
    site_list = []
    for i in range(1, 650):
        print(i)
        # 为了取到2019.7.1日之后所有的燃油新闻，这里取200（以后可以取最新的新闻与现有的df结合在一起
        url = 'https://oil.in-en.com/news/intl/list128-'+str(i)+'.html'
        content = get_text(url)
        r_1 = re.findall('<a href="(.*?)" target="_blan', content)
        for i in range(0, len(r_1)):
            if 'https://oil.in-en.com/html/oil' in r_1[i]:
                site_list.append(r_1[i])
    df = pd.DataFrame({'website': site_list})
    df.drop_duplicates(subset=['website'], inplace=True)
    df.to_excel('website.xlsx')
    return df


def get_content():
    # 获取对应网站的标题，日期以及内容
    df = get_site()
    site_list = df['website'].tolist()
    site_list.sort(reverse=True)
    date_list = []
    title_list = []
    content_list = []

    for i in range(0, len(site_list)):
        url = site_list[i]
        content = get_content(url)
        r_1 = re.findall(r'<h1 id="title">(.*?)</h1>', content)
        if len(r_1) >= 1:
            c = re.sub(r'<.*>','',r_1[0])
            title_list.append(c)
        else:
            title_list.append('')
        r_2 = re.findall(r'<div class="content" id="article"><p>(.*?)</p></div>', content)
        if len(r_2) >= 1:
            d = re.sub(r'<.*?>', '', r_2[0])
            if d != '':
                content_list.append(d)
            else:
                content_list.append(r_2[0])
        else:
            content_list.append('')
        r_3 = re.findall(r'日期：(.*?)</b>', content)
        if len(r_3) >= 1:
            date_list.append(r_3[0])
        else:
            date_list.append('')
    for j in range(0, len(content_list)):
        if content_list[j] == '':
            content_list[j] = title_list[j]

    df1 = pd.DataFrame({'date': date_list, 'website': site_list, 'title': title_list, 'content': content_list})

    df1.to_excel('content.xlsx')


def separate_word():
    # 对于标题进行分词
    df = pd.read_excel('content.xlsx')
    date_list = df['date'].tolist()
    title_list = df['title'].tolist()
    content_list = df['content'].tolist()
    separate_list = []

    for i in range(0, len(title_list)):
        title = title_list[i]
        if type(title) == str:
            separate = jieba.lcut(title)
        else:
            separate = ''
        separate_list.append(separate)

    df1 = pd.DataFrame({'date': date_list, 'title': title_list, 'separate': separate_list, 'content': content_list})

    def get_sep(sentence):
        sep = []
        flag = ['ag', 'a', 'ad', 'an', 'vg', 'v', 'vd', 'vn']
        words = pseg.cut(sentence)
        for w in words:
            if w.flag in flag:
                sep.append(w.word)
        return sep

    df1['sep'] = df1.apply(lambda x: get_sep(x.title), axis=1)
    df1.to_excel('separate.xlsx')



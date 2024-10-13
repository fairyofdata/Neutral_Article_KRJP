import streamlit as st
from bs4 import BeautifulSoup
import requests
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service  # Service 클래스 임포트
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import FinanceDataReader as fdr
import mplfinance as mpf
from datetime import datetime, timedelta
import json
import yaml
import streamlit_authenticator as stauth
import numpy as np
import requests as rq
from streamlit_authenticator.utilities.hasher import Hasher
import os.path
import pickle as pkle
from streamlit_js_eval import streamlit_js_eval
from passlib.context import CryptContext
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events
from cvxopt import matrix, solvers
from streamlit_authenticator.utilities import (CredentialsError,
                                               ForgotError,
                                               Hasher,
                                               LoginError,
                                               RegisterError,
                                               ResetError,
                                               UpdateError)
from streamlit_extras.switch_page_button import switch_page
from pymongo import MongoClient
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pypfopt import risk_models, BlackLittermanModel, expected_returns

st.set_page_config(
        page_title="ESG 정보 제공 플랫폼",
        page_icon=":earth_africa:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
# st.title('Kwargs')

# 세션 상태를 초기화 
if 'ndays' not in st.session_state: 
    # 세션 상태에 이미 등록되어 있지 않으면 100일로 초기화 하도록 함
    st.session_state['ndays'] = 100
    
if 'code_index' not in st.session_state:
    # 선택된 종목에 해당하는 정수값을 code_index라는 키로 저장(처음엔 0)
    # 선택된 종목을 세션 상태로 관리
    st.session_state['code_index'] = 0
    
if 'chart_style' not in st.session_state:
    # 차트의 유형은 디폴트로 지정
    st.session_state['chart_style'] = 'default'

if 'volume' not in st.session_state:
    # 거래량 출력 여부는 true 값으로 초기화
    st.session_state['volume'] = True

if 'login_status' not in st.session_state:
    st.session_state['login_status'] = False
    
if 'user_name' not in st.session_state:
    st.session_state['username'] = None

if 'clicked_points' not in st.session_state:
    st.session_state['clicked_points'] = None
    
if 'sliders' not in st.session_state:
    st.session_state['sliders'] = {}

for key in ['environmental', 'social', 'governance']:
    if key not in st.session_state['sliders']:
        st.session_state['sliders'][key] = 0


# 전처리 함수 정의
def preprocess_data(df):
    # 기존 컬럼명을 사용할 수 있도록 유효성을 확인
    if 'environmental' in df.columns and 'social' in df.columns and 'governance' in df.columns:
        # ESG 영역 비중을 백분율로 환산
        df['env_percent'] = df['environmental'] / (df['environmental'] + df['social'] + df['governance'])
        df['soc_percent'] = df['social'] / (df['environmental'] + df['social'] + df['governance'])
        df['gov_percent'] = df['governance'] / (df['environmental'] + df['social'] + df['governance'])

        # 각 영역별 최종 점수 계산 (average_label 필요)
        df['env_score'] = df['average_label'] * df['env_percent']
        df['soc_score'] = df['average_label'] * df['soc_percent']
        df['gov_score'] = df['average_label'] * df['gov_percent']

        # 연도별 가중치 설정
        latest_year = df['Year'].max()
        year_weights = {
            latest_year: 0.5,
            latest_year - 1: 0.25,
            latest_year - 2: 0.125,
            latest_year - 3: 0.0625,
            latest_year - 4: 0.0625
        }

        # 가중치를 반영한 각 영역별 점수 합산
        df['environmental'] = df.apply(lambda x: x['env_score'] * year_weights.get(x['Year'], 0), axis=1)
        df['social'] = df.apply(lambda x: x['soc_score'] * year_weights.get(x['Year'], 0), axis=1)
        df['governance'] = df.apply(lambda x: x['gov_score'] * year_weights.get(x['Year'], 0), axis=1)

        # 동일 기업의 연도별 점수를 합산하여 최종 점수 도출
        final_df = df.groupby(['Company', 'industry', 'ticker']).agg({
            'environmental': 'sum',
            'social': 'sum',
            'governance': 'sum'
        }).reset_index()

        return final_df
    else:
        raise KeyError("The expected columns 'environmental', 'social', and 'governance' are not present in the dataframe.")

# step 1 : load the provided dataset
# file_path = r"C:\esgpage\kwargs\esgpage\userinterface\240820_final_dummy.csv"
# file_path = r"C:\esgpage\kwargs\esgpage\userinterface\240928_dummy_update.csv"
# # dummy = pd.read_csv(file_path, encoding='euc-kr')
# dummy = pd.read_csv(file_path)
# dummy = preprocess_data(dummy)

# 한국거래소 코스피 인덱스에 해당하는 종목 가져오기
@st.cache_data
def getSymbols(market='KOSPI',sort='Marcap'): # 정렬하는 기준을 시장가치(Marcap)으로 함
    df = fdr.StockListing(market)
    # 정렬 설정 (= 시가총액 기준으로는 역정렬)
    ascending = False if sort == 'Marcap' else True
    df.sort_values(by=[sort],ascending=ascending, inplace=True)
    return df[['Code','Name','Market']]

@st.cache_data
def load_stock_data(code, ndays):
    end_date = pd.to_datetime('today')
    start_date = end_date - pd.Timedelta(days=ndays)
    data = fdr.DataReader(code, start_date, end_date)
    return data

# 캔들차트 출력 함수
def plotChart(data): # 외부에서 데이터를 주면 이를 바탕으로 캔들 차트 출력
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    chart_style = st.session_state['chart_style']
    marketcolors = mpf.make_marketcolors(up='red',down='blue') # 양, 음봉 선택
    mpf_style = mpf.make_mpf_style(base_mpf_style=chart_style,marketcolors=marketcolors)

    fig, ax = mpf.plot(
        data=data, # 받아온 데이터
        volume=st.session_state['volume'], # 거래량을 출력 여부에 대한 것
        type='candle', # 차트 타입
        style=mpf_style, # 스타일 객체
        figsize=(10,7),
        fontscale=1.1,
        mav=(5,10,30), # 이동평균선(5, 10, 30일 이동평균을 출력하겠다는 뜻)
        mavcolors=('red','green','blue'), # 각 이동평균선의 색상
        returnfig=True # figure 객체 반환 
    )
    st.pyplot(fig)



def has_changes(sliders):
    return any(sliders[key] != initial_values[key] for key in sliders)

# 크롤링  필요한 함수 정의
def setup_webdriver():
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument('--headless')  # UI 없이 실행하기 위한 headless 모드
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    # 서비스 객체를 사용하여 드라이버 초기화
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(3)
    
    return driver

# 필요한 함수 정의
def setup_webdriver():
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument('--headless')  # UI 없이 실행하기 위한 headless 모드
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    # 서비스 객체를 사용하여 드라이버 초기화
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(3)
    
    return driver

# URL 생성 함수 정의
def makePgNum(num):
    return 1 + 10 * (num - 1)

def makeUrl(search, page):
    page_num = makePgNum(page)
    url = f"https://search.naver.com/search.naver?where=news&sm=tab_pge&query={search}&start={page_num}"
    return url

# 뉴스 크롤링 함수 추가
def crawl_naver_news(search):
    driver = setup_webdriver()

    target_article_count = 10
    collected_article_count = 0
    current_page = 1
    naver_urls = []

    while collected_article_count < target_article_count:
        search_url = makeUrl(search, current_page)
        driver.get(search_url)
        time.sleep(1)  # 대기시간 변경 가능

        a_tags = driver.find_elements(By.CSS_SELECTOR, 'a.info')

        for a_tag in a_tags:
            if collected_article_count >= target_article_count:
                break
            a_tag.click()
            driver.switch_to.window(driver.window_handles[1])
            time.sleep(3)

            url = driver.current_url
            if "news.naver.com" in url:
                naver_urls.append(url)
                collected_article_count += 1

            driver.close()
            driver.switch_to.window(driver.window_handles[0])

        current_page += 1

    driver.quit()
    return fetch_news_contents(naver_urls)

def fetch_news_contents(naver_urls):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"}
    news_list = []

    for url in naver_urls:
        original_html = requests.get(url, headers=headers)
        html = BeautifulSoup(original_html.text, "html.parser")
        title_element = html.select_one("div#ct > div.media_end_head.go_trans > div.media_end_head_title > h2")
        title = title_element.get_text(strip=True) if title_element else "No title found"
        news_list.append((title, url))

    return news_list

# step2-1 : 사용자 입력
def get_user_input():
    with st.form(key='interest_form'):
        # 사용자의 ESG 선호도와 관심 산업군을 입력받는 부분입니다.
        industry_choices = df_new['industry'].unique().tolist()
        selected_industries = st.multiselect('관심 산업군을 선택하세요',industry_choices,key='unique_key_for_industries')
        esg_weights = {}
        for key in ['environmental', 'social', 'governance']:
            st.session_state['sliders'][key] = st.slider(key, 0, 10, st.session_state['sliders'][key], 1)
            esg_weights[key] = st.session_state['sliders'][key]
        submit_button = st.form_submit_button(label='완료')
        
    if submit_button:
        all_sliders_zero = all(value == 0 for value in st.session_state['sliders'].values())
        if not selected_industries or all_sliders_zero:
            st.warning('슬라이더 값을 변경하여 주십시오.')
        else:
            st.write(' ')
            st.write(' ')
            return esg_weights, selected_industries
    return esg_weights, selected_industries

# Step 3: 기업 추천
def recommend_companies(esg_weights, selected_industries, df):
    # 전처리된 데이터에서 사용자의 ESG 선호도 가중치를 반영하여 최종 점수 계산
    df['final_score'] = (
        esg_weights['environmental'] * df['environmental'] +
        esg_weights['social'] * df['social'] +
        esg_weights['governance'] * df['governance']
    )
    # 산업군 필터링
    filtered_df = df[df['industry'].isin(selected_industries)]
    # 상위 10개 기업 선정
    top_companies = filtered_df.sort_values(by='final_score', ascending=False).head(10)
    return top_companies


# 포트폴리오 비중 계산 함수 with Black-Litterman
def calculate_portfolio_weights(df, esg_weights):
    tickers = df['ticker'].tolist()
    price_data = yf.download(tickers, start="2019-01-01", end="2023-01-01")['Adj Close']

    if price_data.isnull().values.any():
        return "일부 데이터가 누락되었습니다. 다른 기업을 선택해 주세요.", None

    # 평균 수익률과 공분산 행렬
    mu_market = expected_returns.capm_return(price_data)  # CAPM을 통한 시장 균형 수익률 계산
    Sigma = risk_models.sample_cov(price_data)  # 샘플 공분산 행렬

    # 사용자 선호도를 반영한 ESG 점수 조정
    df['final_esg_score'] = (
        esg_weights['environmental'] * df['environmental'] +
        esg_weights['social'] * df['social'] +
        esg_weights['governance'] * df['governance']
    )

    # 사용자 ESG 점수를 투자자의 견해로 반영
    # 예: ESG 점수가 높을수록 해당 기업의 기대 수익률이 증가
    P = np.eye(len(tickers))  # P 매트릭스: 각 자산에 대해 의견을 반영
    Q = df['final_esg_score'].values  # Q 벡터: 각 자산에 대한 투자자의 의견 (ESG 점수 반영)

    # Black-Litterman 모델 적용
    bl = BlackLittermanModel(Sigma, pi=mu_market, P=P, Q=Q)
    adjusted_returns = bl.bl_returns()

    # `cvxopt`에서 사용할 행렬 형식으로 변환
    n = len(mu_market)
    P_opt = matrix(Sigma.values)
    q_opt = matrix(-adjusted_returns.values)
    G = matrix(-np.eye(n))
    h = matrix(np.zeros(n))
    A = matrix(1.0, (1, n))
    b = matrix(1.0)

    # 쿼드라틱 프로그래밍 솔버 실행
    sol = solvers.qp(P_opt, q_opt, G, h, A, b)

    # 최적 가중치 추출
    weights = np.array(sol['x']).flatten()

    # 포트폴리오 성과 지표 계산
    expected_return = np.dot(weights, mu_market)
    expected_volatility = np.sqrt(np.dot(weights.T, np.dot(Sigma.values, weights)))
    sharpe_ratio = expected_return / expected_volatility

    # 가중치 정리
    cleaned_weights = dict(zip(tickers, weights))

    return cleaned_weights, (expected_return, expected_volatility, sharpe_ratio)


# with st.sidebar:
#     selected = option_menu("메뉴", ['설문 설명','설문 페이지','ESG 소개', '방법론','최근 뉴스'], 
#         icons=['bi bi-house','bi bi-house','bi bi-globe2','bi bi-map', 'bi bi-newspaper']
#         , menu_icon="cast", default_index=0)

with st.sidebar:
    st.page_link('main_survey_introduce.py', label='홈', icon="🎯")
    st.page_link('pages/survey_page.py', label='설문', icon="📋")
    st.page_link('pages/survey_result.py', label='설문 결과',icon="📊")
    st.page_link('pages/recent_news.py', label='최신 뉴스',icon="🆕")
    st.page_link('pages/esg_introduce.py', label='ESG 소개 / 투자 방법', icon="🧩")

# page=st.session_state

# 초기 페이지
# if selected == '설문 설명':

st.markdown('''<div>
                    <h2 style="font-size:40px; text-align:center;">ESG 평가기관 선호도 설문</h2>
                </div>
                ''',unsafe_allow_html=True)
_,start_page,_ = st.columns([1,2,1])
with start_page:
    st.markdown('''
                <!DOCTYPE html>
                <html lang="ko">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Test Page</title>
                    <style>
                        div[data-testid="stHeadingWithActionElements"]{
                            font-size : 40px;
                        }
                        div[data-testid="stApp"]{
                            background-image: linear-gradient(rgb(178,221,247),rgb(231,246,255))
                        }
                        header[data-testid="stHeader"]{
                            background-color: #b2ddf7;
                            padding-left:80px;
                        }
                        header[data-testid="stHeader"]::after {
                            content: "Kwargs";
                            display: block;
                            font-size: 30px;
                            word-spacing: 30px;
                            font-weight: bold;
                            color: black;
                            padding: 10px;
                        }
                        button[data-testid="baseButton-secondary"]{
                            background-color: #e7f6ff;
                            border-radius: 10px;
                            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                        }
                        div[data-testid="stVerticalBlock"]{
                            text-align : center;
                        }
                        .container {
                            max-width: 800px;
                            margin: auto;
                            padding: 20px;
                            background-color: #e7f6ff;
                            border-radius: 10px;
                            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                        }
                        h1 {
                            text-align: center;
                        }
                        p {
                            font-size: 18px;
                        }
                        .btn-start {
                            display: block;
                            width: 100%;
                            background-color: #4CAF50;
                            color: white;
                            padding: 15px;
                            text-align: center;
                            border: none;
                            border-radius: 5px;
                            font-size: 18px;
                            cursor: pointer;
                            margin-top: 20px;
                        }
                        .btn-start:hover {
                            background-color: #45a049;
                        }
                    </style>
                </head>
                <body>
                <div class="container">
                    <p style="text-align:center; text-color:#0000;">환영합니다!</p>
                    <p>이 설문은 귀하의 <strong>ESG(환경, 사회, 지배구조)</strong> 투자 관점과 가치에 대한 이해를 돕기 위해 마련되었습니다. 
                        귀하의 선호도를 반영하여 보다 개인화된 투자 분석과 포트폴리오 제안을 제공하기 위해, 간단한 질문에 응답해 주세요.
                        귀하의 설문 결과를 반영한 ESG 평가로 보다 신뢰할 수 있는 투자 정보를 제공하며, 사회적 책임과 환경적 가치를 고려한 맞춤형 포트폴리오를 설계합니다.
                    </p>
                    <h2 style="font-size:20px; text-align:center;text-color:#0000;">소요 시간</h2>
                    <p style="text-align:center;text-color:#0000;">약 <strong>5분</strong> 정도 소요됩니다.</p>
                    <p style="text-align:center;text-color:#0000;">여러분의 소중한 의견은 지속 가능한 투자의 중요한 지침이 됩니다. 지금 바로 설문을 시작해 주세요!</p>
                </div>
                </body>
                </html>
                ''',unsafe_allow_html=True)
    st.write('')
    _,start_button,_ = st.columns(3)
    with start_button:
        switch_page = st.button("설문 시작하기")
        if switch_page:
            selected = '설문 페이지'
            st.switch_page('pages/survey_page.py')

# 설문 시작 페이지
# elif selected == '설문 페이지':
#     st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
#     st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)
#     values = {'msci': 0, 'iss': 0, 'sustain': 0, 'sandp': 0, 'esg1': 0}
        
#     df_new = dummy.copy()
   
#     with st.form('usersurvey',clear_on_submit=False):
#     # 설문지 제목
#         st.markdown('<h1 style="font-size:30px;text-align:center;">ESG 관심 테스트</h1>', unsafe_allow_html=True)
#         st.markdown('<p style="font-size:15px;text-align:center;">테스트 결과에 따른 투자 포트폴리오 추천</p>', unsafe_allow_html=True)
#         st.write('')
#         st.markdown('<div class="form-container">', unsafe_allow_html=True)

#         # 질문 1
#         st.markdown('<div class="question" style="font-size:20px; text-align:center;font-weight: bold;">1. 기후 변화에 대한 기업의 대응 전략을 얼마나 중요하게 생각하십니까?</div>', unsafe_allow_html=True)
#         q1 = st.radio('', options=('매우 중요하다', '중요하다', '보통이다', '중요하지 않다', '전혀 중요하지 않다'))
#         st.write('')
        
#         # 질문 2
#         st.markdown('<div class="question" style="font-size:20px;text-align:center;font-weight: bold;">2. 기업의 인권 보호와 노동 조건 개선을 얼마나 중요하게 생각하십니까?</div>', unsafe_allow_html=True)
#         q2 = st.radio('  ', options=('매우 중요하다', '중요하다', '보통이다', '중요하지 않다', '전혀 중요하지 않다'))
#         st.write('')
        
#         # 질문 3
#         st.markdown('<div class="question" style="font-size:20px;text-align:center;font-weight: bold;">3. 기업의 지배구조(경영 투명성, 이사회 독립성 등)를 얼마나 중시합니까?</div>', unsafe_allow_html=True)
#         q3 = st.radio('   ', options=('매우 중요하다', '중요하다', '보통이다', '중요하지 않다', '전혀 중요하지 않다'))
#         st.write('')
        
#         # 질문 4
#         st.markdown('<div class="question" style="font-size:20px;text-align:center;font-weight: bold;">4. 환경, 사회, 지배구조 중 어느 요소가 가장 중요하다고 생각하십니까?</div>', unsafe_allow_html=True)
#         q4 = st.radio('    ', options=('환경(Environment)', '사회(Social)', '지배구조(Government)', '모든 요소가 균형 있게 중요하다'))

#         # 재무 지표에 대한 조정
#         st.markdown('<div class="question" style="font-size:20px; text-align:center;font-weight: bold;">5. 본인의 투자 성향을 선택하여 주십시오.</div>', unsafe_allow_html=True)
#         q5 = st.radio('     ',options=('안정적','공격적','장기적'))

#         _,survey_submitted, _ = st.columns([3,1,3])
#         with survey_submitted:
#             submitted = st.form_submit_button('설문 완료')
#         # if not selected_industries:
#         #     st.markdown('<h1 class="question" style="font-size:23px; text-align:center;">설문에 모두 답변해 주시기 바랍니다.</h1>', unsafe_allow_html=True)

#         if submitted:
#             try:
#                 survey_result = pd.DataFrame(index=['E', 'S', 'G'], columns=['esg1', 'sandp', 'sustain', 'iss', 'msci'])
#                 survey_result.loc[:, :] = 0
#                 if q1 == '매우 중요하다':
#                     survey_result.at['E', 'msci'] = 1
#                 else: 
#                     survey_result.at['E', 'sandp'] += 1
#                     survey_result.at['E', 'sustain'] += 1
#                     survey_result.at['E', 'iss'] += 1
#                     survey_result.at['E', 'esg1'] += 1
                    
#                 if q2 == '매우 중요하다':
#                     survey_result.at['S', 'iss'] += 1
#                 else:
#                     survey_result.at['S', 'msci'] += 1
#                     survey_result.at['S', 'sandp'] += 1
#                     survey_result.at['S', 'sustain'] += 1
#                     survey_result.at['S', 'esg1'] += 1
                    
#                 if q3 == '매우 중요하다':
#                     survey_result.at['G', 'msci'] += 1
#                 else:
#                     survey_result.at['G', 'sandp'] += 1
#                     survey_result.at['G', 'sustain'] += 1
#                     survey_result.at['G', 'iss'] += 1
#                     survey_result.at['G', 'esg1'] += 1
                    
#                 if q4 == '환경(Environment)':
#                     survey_result.at['E', 'msci'] += 1
#                 elif q4 == '사회(Social)':
#                     survey_result.at['S', 'iss'] += 1
#                 elif q4 == '지배구조(Government)':
#                     survey_result.at['G', 'sustain'] += 1
#                 elif q4 == '모든 요소가 균형 있게 중요하다':
#                     survey_result.at['E', 'esg1'] += 1
#                     survey_result.at['S', 'esg1'] += 1
#                     survey_result.at['G', 'esg1'] += 1
                    
#             finally:
#                 page['current_page'] = 'result_page'
                
#                 # 각 기관별 합계 계산 (E, S, G 점수 합산)
#                 total_scores = survey_result.sum(axis=0)
#                 # 각 기관의 가중치 계산 (전체 점수의 비율)
#                 user_feedback_weights = total_scores / total_scores.sum()

#     # if page['current_page'] == 'result_page':
#     if submitted:
#         top_interest_intuition = survey_result.sum().idxmax()
#         col1, col2 = st.columns(2)
#         # ESG 각 섹터에 대한 관심
#         with col1:
#             st.markdown('<h1 style="font-size:15px; text-align:center;">해당 슬라이더의 초기값은 설문지를 바탕으로 도출된 값입니다.</h1>', unsafe_allow_html=True)
#             st.markdown('<h1 style="font-size:20px; text-align:center;">Environmental</h1>', unsafe_allow_html=True)
#             e_value = st.slider(' ', min_value=0, max_value=10, value=survey_result.loc['E'].sum())
            
#             st.markdown('<h1 style="font-size:20px; text-align:center;">Social</h1>', unsafe_allow_html=True)
#             s_value = st.slider('', min_value=0, max_value=10, value=survey_result.loc['S'].sum())
            
#             st.markdown('<h1 style="font-size:20px; text-align:center;">Governance</h1>', unsafe_allow_html=True)
#             g_value = st.slider('  ', min_value=0, max_value=10, value=survey_result.loc['G'].sum())

#         esg_weights = {'environmental': e_value, 'social': s_value, 'governance': g_value}
#         st.write(esg_weights)
#         # industry_choices = df_new['industry'].unique().tolist()
        
#         # 주식 추천 알고리즘 -> 추천 회사 도출
#         # 합산 값 중 가장 큰 값을 가진 열의 이름을 찾음
#         top_interest_intuition = survey_result.sum().idxmax()
    
#                 # 출력 이름 매핑
#         intuition_mapping = {'msci': 'MSCI',
#                 'iss': 'ISS',
#                 'sustain': 'Sustainalytics',
#                 'sandp': 'S&P',
#                 'esg1': 'ESG기준원'
#                 }
#         st.markdown('<div style="font-size:20px;font-weight: bold;">나와 유사한 평가 성향을 가진 기관</div>', unsafe_allow_html=True)
#         st.write('')
#         st.markdown(f"""
#                         <div style="font-size:20px;">
#                             <strong>{intuition_mapping[top_interest_intuition]}</strong>
#                         </div>
#                     """, unsafe_allow_html=True)
        
#         st.write('')
#         # top_companies = recommend_companies(esg_weights,df_new)
    
#         # 포트폴리오 비중 계산
#         industries = df_new['industry'].unique().tolist()
#         processed_df = df_new[df_new['industry'].isin(industries)].copy()
#         # cleaned_weights:각 자산에 할당된 최적의 투자 비율
#         # performance:최적화된 포트폴리오의 성과 지표
#         # 블랙리터만 모델 적용
#         portfolio_weights, portfolio_performance = calculate_portfolio_weights(processed_df, esg_weights)
        
#         st.write('1.')
#         st.write(portfolio_weights)
#         st.write('2.')
#         st.write(portfolio_performance)
#         st.write('3.')
#         st.write(df_new)
#         st.write('4.')
        
        
#         # ticker 열과 portfolio_weights를 매핑하여 새로운 top_companies 데이터프레임 생성
#         top_companies = df_new[df_new['ticker'].isin(portfolio_weights)].copy()

#         # portfolio_weights의 값을 'Weight' 컬럼으로 추가
#         total_weight = sum(portfolio_weights.values())
#         top_companies['Weight'] = top_companies['ticker'].map(portfolio_weights) / total_weight
#         st.write(top_companies)
        
#         col3, col4 = st.columns(2)
#         with col3:
#             top_companies = top_companies.sort_values(by='Weight', ascending=False)
#             st.markdown(f"""
#                     <div>
#                         <h3 style="color:#333; text-align:center; font-size:24px">추천 회사</h3>
#                     </div>
#                 """, unsafe_allow_html=True)
#             fig = px.pie(top_companies, names='Company', values='Weight', color_discrete_sequence=px.colors.qualitative.G10)
#             fig.update_traces(textposition='auto',customdata=top_companies['Weight'] / top_companies['Weight'].sum() * 100,textinfo='percent+label+value',hovertemplate='%{label}: %{customdata:.1f}%',texttemplate='%{label}',)
#             fig.update_layout(font=dict(size=16, color='black'),showlegend=False,
#                 legend=dict(orientation="v",yanchor="middle",y=0.5,xanchor="left",x=1.1),margin=dict(t=40, b=40, l=0, r=0),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)', width=600,height=400,)
#             clicked_points = plotly_events(fig, click_event=True,key="company_click")
            
#         with col4:
#             # 파이차트에 대한 정보
#             st.write(' ')
#             st.write(' ')
#             companies = top_companies['Company'].unique()
#             output = '<div>'
#             order = 1
#             other_percent = 0
#             top_companies = top_companies.sort_values(by='Weight', ascending=False)
#             for i, row in top_companies.iterrows():
#                 weight_percent = row['Weight'] * 100
#                 if weight_percent >= 5:
#                     weight_percent = round(weight_percent, 2)
#                     st.markdown( f'''
#                 <ul style="font-size:17px; letter-spacing: 1.3px;">
#                     <span style="margin-right: 10px;">{order}. {row["Company"]}</span>
#                     <span>{weight_percent}%</span>
#                 </ul>
#                 ''', unsafe_allow_html=True) 
#                     order += 1
#                 else:
#                     other_percent += weight_percent
#             order += 1
#             other_percent = round(other_percent, 2)
#             st.markdown(f'''
#             <ul style="font-size:17px; letter-spacing: 1.3px;">
#                 <span style="margin-right: 10px;">{order}. 기타</span>
#                 <span>{other_percent}%</span>
#             </ul>
#             ''', unsafe_allow_html=True)
            
#         col_1, col_2,col_3 = st.columns([3,3,4])
#         with col_1:
#             if clicked_points:
#                 st.write(clicked_points)
#                 clicked_point = clicked_points[0]
#                 if 'pointNumber' in clicked_point:
#                     company_index = clicked_point['pointNumber']
#                     if company_index < len(top_companies):
#                         company_info = top_companies.iloc[company_index]
#                         clicked_company = company_info['Company']
#                         st.markdown(f"""
#                 <div>
#                 <h3 style="color:#333; text-align:center; font-size:24px">{clicked_company} ({company_info['industry']})</h3>
#                 <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
#                     <div style="flex:1; text-align:center; padding:10px;">
#                         <h4 style="font-weight:bold; font-size:20px;">환경</h4>
#                             <p style="font-size:20px; color:#444;font-weight:bold;text-align:center;">{company_info['environmental']:.2f}</p>
#                     </div>
#                     <div style="flex:1; text-align:center; padding:10px;">
#                         <h4 style="font-weight:bold;font-size:20px;">사회</h4>
#                             <p style="font-size:20px; color:#444;font-weight:bold;text-align:center;">{company_info['social']:.2f}</p>
#                     </div>
#                     <div style="flex:1; text-align:center; padding:10px;">
#                         <h4 style="font-weight:bold;font-size:20px;">지배구조</h4>
#                              <p style="font-size:20px; color:#444;font-weight:bold;text-align:center;">{company_info['governance']:.2f}</p>
#                      </div>
#                 </div>
#                     <div style="text-align:center; margin-top:10px;">
#                     <h4 style="font-size:22px; font-weight:bold;">추천 포트폴리오 비중</h4>
#                         <p style="font-size:22px; font-weight:bold;">{company_info['Weight'] * 100:.2f}%</p>
#                 </div>
#                 </div>
#             """, unsafe_allow_html=True)
#             elif 'company_index' in st.session_state:
#                 company_info = top_companies.iloc[st.session_state['company_index']]
#                 clicked_company = company_info['Company']
#                 st.markdown(f"""
#     <div>
#     <h3 style="color:#333; text-align:center; font-size:24px">{clicked_company} ({company_info['industry']})</h3>
#     <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
#         <div style="flex:1; text-align:center; padding:10px;">
#             <h4 style="font-weight:bold; font-size:20px;">환경</h4>
#                 <p style="font-size:20px; color:#444;font-weight:bold;text-align:center;">{company_info['environmental']:.2f}</p>
#         </div>
#         <div style="flex:1; text-align:center; padding:10px;">
#             <h4 style="font-weight:bold;font-size:20px;">사회</h4>
#                 <p style="font-size:20px; color:#444;font-weight:bold;text-align:center;">{company_info['social']:.2f}</p>
#         </div>
#         <div style="flex:1; text-align:center; padding:10px;">
#             <h4 style="font-weight:bold;font-size:20px;">지배구조</h4>
#                  <p style="font-size:20px; color:#444;font-weight:bold;text-align:center;">{company_info['governance']:.2f}</p>
#          </div>
#     </div>
#         <div style="text-align:center; margin-top:10px;">
#         <h4 style="font-size:22px; font-weight:bold;">추천 포트폴리오 비중</h4>
#             <p style="font-size:22px; font-weight:bold;">{company_info['Weight'] * 100:.2f}%</p>
#     </div>
#     </div>
#     """, unsafe_allow_html=True)
#             else:
#                 st.write('클릭을 진행하여 주십시오.')
    
#             st.write(' ')
#             st.write(' ')
#             st.write(' ')
            
        
#         with col_2:
#             try:
#             # st.subheader(f'{clicked_company} 주가 그래프')
#                 st.markdown(f"""<div>
#                             <h2 style="font-size: 24px; text-align:center;">{clicked_company} 주가 그래프</h2>
#                             </div>
#             """, unsafe_allow_html=True)
                
                
#                 with st.form(key='chartsetting', clear_on_submit=True):
#                 # top_companies에서 Company와 Ticker 열 사용
#                     company_choices = top_companies['Company'].tolist()
#                     ticker_choices = top_companies['ticker'].tolist()
#                     ticker_choices = [ticker.replace('.KS', '') for ticker in ticker_choices]

#                     if st.session_state['code_index'] >= len(company_choices):
#                         st.session_state['code_index'] = 0

#             # 회사명으로 선택, 하지만 실제로는 Ticker 값을 사용
#                     choice = clicked_company
#                     code_index = company_choices.index(choice)
#                     code = ticker_choices[code_index]  # 선택한 회사의 Ticker 값을 가져옴

#             # 슬라이더로 기간 선택
#                     ndays = st.slider(label='기간 (days)', min_value=50, max_value=365, value=st.session_state['ndays'], step=1)
#                     # chart_styles = ['default', 'binance', 'blueskies', 'brasil', 'charles', 'checkers', 'classic', 'yahoo', 'mike', 'nightclouds', 'sas', 'starsandstripes']
#                     # chart_style = st.selectbox(label='차트 스타일', options=chart_styles, index=chart_styles.index(st.session_state['chart_style']))
#                     chart_style = 'default'
#                     volume = st.checkbox('거래량', value=st.session_state['volume'])
        
#             # 폼 제출 버튼
#                     if st.form_submit_button(label='OK'):
#                 # 세션 상태 업데이트
#                         st.session_state['ndays'] = ndays
#                         st.session_state['code_index'] = code_index
#                         st.session_state['chart_style'] = chart_style
#                         st.session_state['volume'] = volume

#                 # 선택된 종목의 주가 데이터 로드
#                         data = load_stock_data(code, ndays)
                
#                 # 주가 차트 시각화 함수 호출
#                         plotChart(data)
#             except:
#                 st.write(' ')
                
#         with col_3:
#             if clicked_points:
#                 st.markdown(f"""<div>
#                             <h2 style="font-size: 24px; text-align:center;">{clicked_company} 워드 클라우드</h2>
#                             </div>
#                 """, unsafe_allow_html=True)
#                 # MongoDB에서 Company 필드의 고유 값들을 불러오기
#                 company_list = collection.distinct('Company')
#                 st.write(company_list)
# # 선택할 수 있는 Company 목록을 Streamlit에서 선택
#                 # clicked_company = st.selectbox('Select a Company:', company_list)
#                 # 유니코드 정규화를 사용해 clicked_company와 company_list 값을 동일한 형식으로 변환
#                 clicked_company_normalized = unicodedata.normalize('NFC', clicked_company)

#                 # 리스트 내의 각 값을 정규화 후 비교
#                 clicked_company = next((company for company in company_list if unicodedata.normalize('NFC', company) == clicked_company_normalized), None)

#                 # MongoDB에서 Company_ID 입력을 받아 해당 데이터를 불러오기
#                 # 선택된 Company에 대한 제목들을 가져오기
#                 titles = collection.find({'Company': clicked_company}, {'_id': 0, 'title': 1})

# # 불러온 데이터 리스트로 저장
#                 title_list = [document['title'] for document in titles if 'title' in document]

# # title_list가 비어 있는지 확인
#                 if not title_list:
#                     st.warning("데이터가 없습니다. 다른 Company를 선택해 주세요.")
#                 else:
#     # 형태소 분석기 설정
#                     okt = Okt()
#                     nouns_adj_verbs = []

#     # 명사, 형용사만 추출
#                     for title in title_list:
#                         tokens = okt.pos(title, stem=True)
#                         for word, pos in tokens:
#                             if pos in ['Noun', 'Adjective']:
#                                 nouns_adj_verbs.append(word)

#     # 빈도수 계산
#                     word_counts = Counter(nouns_adj_verbs)
#                     data = word_counts.most_common(500)
#                     tmp_data = dict(data)

#     # 워드 클라우드 생성 - 폰트 경로 확인 후 설정
#                     try:
#                         wordcloud = WordCloud(
#                     font_path='C:/Windows/Fonts/malgun.ttf',  # Windows 시스템에서 사용할 기본 폰트 설정
#                     background_color='white',
#                     width=800,
#                     height=600
#                         ).generate_from_frequencies(tmp_data)
#                     except OSError:
#                         st.error("폰트 파일을 불러올 수 없습니다. 폰트 경로를 확인하거나 설치해 주세요.")
#                         st.stop()

#     # 워드 클라우드 시각화 및 출력
#                     fig, ax = plt.subplots(figsize=(10, 6))
#                     ax.imshow(wordcloud, interpolation='bilinear')
#                     ax.axis('off')

#     # Streamlit에 워드 클라우드 출력
#                     st.pyplot(fig)

   

        
# elif selected == 'ESG 소개':
#     col1,_,_ = st.columns([1,2,1])
#     with col1:
#         st.subheader('**ESG 소개**')
#         st.image('https://media.istockphoto.com/id/1447057524/ko/%EC%82%AC%EC%A7%84/%ED%99%98%EA%B2%BD-%EB%B0%8F-%EB%B3%B4%EC%A0%84%EC%9D%84-%EC%9C%84%ED%95%9C-%EA%B2%BD%EC%98%81-esg-%EC%A7%80%EC%86%8D-%EA%B0%80%EB%8A%A5%EC%84%B1-%EC%83%9D%ED%83%9C-%EB%B0%8F-%EC%9E%AC%EC%83%9D-%EC%97%90%EB%84%88%EC%A7%80%EC%97%90-%EB%8C%80%ED%95%9C-%EC%9E%90%EC%97%B0%EC%9D%98-%EA%B0%9C%EB%85%90%EC%9C%BC%EB%A1%9C-%EB%85%B9%EC%83%89-%EC%A7%80%EA%B5%AC%EB%B3%B8%EC%9D%84-%EB%93%A4%EA%B3%A0-%EC%9E%88%EC%8A%B5%EB%8B%88%EB%8B%A4.jpg?s=612x612&w=0&k=20&c=ghQnfLcD5dDfGd2_sQ6sLWctG0xI0ouVaISs-WYQzGA=', width=600)
#     st.write("""
#     ESG는 환경(Environment), 사회(Social), 지배구조(Governance)의 약자로, 기업이 지속 가능하고 책임 있는 경영을 위해 고려해야 하는 세 가지 핵심 요소를 의미합니다. ESG는 단순한 윤리적 개념을 넘어, 장기적인 기업의 성공과 지속 가능성을 확보하기 위해 중요한 역할을 합니다.

#         ### 환경 (Environment)
#         환경 요소는 기업이 환경에 미치는 영향을 측정하고 개선하는 데 중점을 둡니다. 이는 기후 변화 대응, 자원 효율성, 오염 방지, 생물 다양성 보전 등의 문제를 포함합니다. 환경 지속 가능성을 강화하는 것은 기업의 평판을 높이고, 법적 리스크를 줄이며, 장기적으로 비용 절감을 가능하게 합니다.

#         ### 사회 (Social)
#         사회 요소는 기업이 사회에 미치는 영향을 평가합니다. 이는 인권 보호, 노동 조건 개선, 지역 사회 기여, 다양성과 포용성 증진 등을 포함합니다. 긍정적인 사회적 영향을 미치는 기업은 직원의 사기와 생산성을 높이고, 고객과 지역 사회의 신뢰를 얻을 수 있습니다.

#         ### 지배구조 (Governance)
#         지배구조 요소는 기업의 경영 방식과 의사 결정 과정을 다룹니다. 이는 투명한 회계 관행, 이사회 구성, 경영진의 윤리적 행동, 주주 권리 보호 등을 포함합니다. 건전한 지배구조는 기업의 안정성과 지속 가능성을 보장하고, 투자자들의 신뢰를 증대시킵니다.

#         ## 왜 ESG가 중요한가요?
#         ### 1. 위험 관리
#         ESG를 고려하는 기업은 환경적, 사회적, 법적 리스크를 더 잘 관리할 수 있습니다. 이는 장기적인 기업의 안정성과 성장을 도모합니다.

#         ### 2. 투자 유치
#         많은 투자자들이 ESG 요인을 고려하여 투자를 결정합니다. ESG를 충실히 이행하는 기업은 더 많은 투자 기회를 얻을 수 있습니다.

#         ### 3. 평판 향상
#         ESG에 대한 책임을 다하는 기업은 고객과 지역 사회로부터 더 높은 신뢰와 긍정적인 평판을 얻습니다. 이는 브랜드 가치를 높이고, 장기적으로 비즈니스 성공에 기여합니다.

#         ### 4. 법적 준수
#         전 세계적으로 ESG 관련 규제가 강화되고 있습니다. ESG 기준을 준수하는 기업은 법적 리스크를 최소화하고, 규제 변경에 유연하게 대응할 수 있습니다.

#         ## 결론
#         ESG는 단순한 트렌드가 아니라, 기업의 지속 가능성과 장기적인 성공을 위한 필수적인 요소입니다. 우리는 ESG 원칙을 바탕으로 책임 있는 경영을 실천하며, 환경 보호, 사회적 기여, 투명한 지배구조를 통해 더 나은 미래를 만들어 나가고자 합니다. 여러분의 지속적인 관심과 지지를 부탁드립니다.
#         """)
    
# elif selected == '방법론':
#     st.write("""
#         안녕하십니까 
#         당사의 주식 추천 사이트에 방문해 주셔서 감사합니다. 저희는 기업의 환경(Environment), 사회(Social), 지배구조(Governance) 측면을 종합적으로 평가하여 사용자에게 최적의 주식을 추천하는 서비스를 제공합니다. 당사의 방법론은 다음과 같은 주요 요소를 포함합니다.

#         ## 1. ESG 스코어 정의 및 평가 기준
#         ESG 스코어는 기업의 지속 가능성과 책임 있는 경영을 측정하는 지표로, 다음과 같은 세 가지 주요 분야를 포함합니다:

#         #### 환경(Environment)
#         기업이 환경 보호를 위해 수행하는 노력과 성과를 평가합니다. 이는 온실가스 배출량, 에너지 효율성, 자원 관리, 재생 가능 에너지 사용 등으로 측정됩니다.

#         #### 사회(Social)
#         기업의 사회적 책임을 평가합니다. 직원 복지, 지역 사회에 대한 기여, 인권 보호, 공급망 관리 등과 같은 요소가 포함됩니다.

#         #### 지배구조(Governance)
#         기업의 관리 및 운영 방식에 대한 투명성과 책임성을 평가합니다. 이사회 구조, 경영진의 윤리, 부패 방지 정책, 주주 권리 보호 등이 고려됩니다.

#         ## 2. 데이터 수집 및 분석
#         저희는 ESG 스코어를 산출하기 위해 신뢰할 수 있는 다양한 데이터 소스를 활용합니다. 주요 데이터 소스에는 기업의 연례 보고서, 지속 가능성 보고서, 뉴스 및 미디어 기사, 그리고 전문 ESG 평가 기관의 리포트가 포함됩니다. 이 데이터를 바탕으로 저희는 다음과 같은 분석 과정을 진행합니다:

#         #### 정량적 분석
#         수치 데이터 및 KPI(핵심 성과 지표)를 기반으로 한 환경적, 사회적, 지배구조적 성과 분석을 수행합니다.

#         #### 정성적 분석
#         기업의 정책 및 이니셔티브, 업계 평판 등을 평가하여 ESG 관련 활동의 질적 측면을 분석합니다.

#         ## 3. ESG 스코어 산출 및 가중치 적용
#         각 기업의 ESG 성과를 기반으로 종합 스코어를 산출하며, 환경, 사회, 지배구조 각 항목에 대해 가중치를 적용하여 전체 ESG 스코어를 계산합니다. 가중치는 산업별, 지역별 특성에 맞추어 조정됩니다. 이 과정에서 기업의 업종과 특성을 반영하여 보다 정확한 평가가 이루어집니다.

#         ## 4. 주식 추천 알고리즘
#         ESG 스코어를 바탕으로 사용자 맞춤형 주식 추천 알고리즘을 운영합니다. 사용자의 투자 목표, 리스크 수용 범위, 관심 산업 등을 고려하여 ESG 점수가 높은 기업을 추천합니다. 알고리즘은 다음과 같은 요소를 반영합니다:

#         #### ESG 스코어
#         높은 ESG 스코어를 가진 기업을 우선 추천합니다.
#         #### 재무 성과
#         기업의 재무 건전성과 성장 잠재력도 함께 고려합니다.
#         #### 시장 동향
#         현재 시장 동향 및 산업별 특성을 반영하여 추천합니다.
    
#         ## 5. 지속적인 모니터링 및 업데이트
#         ESG 관련 정보는 지속적으로 업데이트되며, 기업의 ESG 스코어는 정기적으로 재평가됩니다. 이를 통해 최신 정보를 바탕으로 사용자에게 정확한 추천을 제공하며, 기업의 ESG 성과 변화에 신속하게 대응합니다.

#         ## 6. 투명한 정보 제공
#         저희는 사용자가 신뢰할 수 있는 정보를 제공하기 위해 ESG 스코어 산출 과정과 데이터 출처를 투명하게 공개합니다. 사용자는 각 기업의 ESG 성과에 대한 자세한 정보를 확인할 수 있으며, 이를 바탕으로 보다 나은 투자 결정을 내릴 수 있습니다.
        
#         저희의 ESG 스코어 기반 주식 추천 서비스는 책임 있는 투자와 지속 가능한 성장을 지향합니다. 여러분의 투자 결정에 도움이 되기를 바랍니다.""")

# elif selected == '최근 뉴스':
#     st.write(' ')
#     st.write(' ')
#     st.subheader('최근 경제 뉴스')

#     # 검색어 입력
#     search = st.text_input("검색할 키워드를 입력하세요:")

#     # 버튼 클릭 시 크롤링 시작
#     if st.button("뉴스 검색"):
#         if search:
#             st.write(f"'{search}' 관련 기사를 검색 중입니다...")
#             news_list = crawl_naver_news(search)

#             if news_list:
#                 # st.write(f"수집된 기사 수: {len(news_list)}개")
#                 for title, link in news_list:
#                     st.markdown(f"- [{title}]({link})")
#             else:
#                 st.write("기사를 찾을 수 없습니다.")
#         else:
#             st.write("검색어를 입력해주세요.")
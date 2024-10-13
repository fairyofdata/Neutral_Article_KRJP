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


with st.sidebar:
    st.page_link('main_survey_introduce.py', label='홈', icon="🎯")
    st.page_link('pages/survey_page.py', label='설문', icon="📋")
    st.page_link('pages/survey_result.py', label='설문 결과',icon="📊")
    st.page_link('pages/recent_news.py', label='최신 뉴스',icon="🆕")
    st.page_link('pages/esg_introduce.py', label='ESG 소개 / 투자 방법', icon="🧩")
    
col1,_,_ = st.columns([1,2,1])
with col1:
    st.subheader('**ESG 소개**')
    st.image('https://media.istockphoto.com/id/1447057524/ko/%EC%82%AC%EC%A7%84/%ED%99%98%EA%B2%BD-%EB%B0%8F-%EB%B3%B4%EC%A0%84%EC%9D%84-%EC%9C%84%ED%95%9C-%EA%B2%BD%EC%98%81-esg-%EC%A7%80%EC%86%8D-%EA%B0%80%EB%8A%A5%EC%84%B1-%EC%83%9D%ED%83%9C-%EB%B0%8F-%EC%9E%AC%EC%83%9D-%EC%97%90%EB%84%88%EC%A7%80%EC%97%90-%EB%8C%80%ED%95%9C-%EC%9E%90%EC%97%B0%EC%9D%98-%EA%B0%9C%EB%85%90%EC%9C%BC%EB%A1%9C-%EB%85%B9%EC%83%89-%EC%A7%80%EA%B5%AC%EB%B3%B8%EC%9D%84-%EB%93%A4%EA%B3%A0-%EC%9E%88%EC%8A%B5%EB%8B%88%EB%8B%A4.jpg?s=612x612&w=0&k=20&c=ghQnfLcD5dDfGd2_sQ6sLWctG0xI0ouVaISs-WYQzGA=', width=600)
st.write("""
ESG는 환경(Environment), 사회(Social), 지배구조(Governance)의 약자로, 기업이 지속 가능하고 책임 있는 경영을 위해 고려해야 하는 세 가지 핵심 요소를 의미합니다. ESG는 단순한 윤리적 개념을 넘어, 장기적인 기업의 성공과 지속 가능성을 확보하기 위해 중요한 역할을 합니다.

    ### 환경 (Environment)
    환경 요소는 기업이 환경에 미치는 영향을 측정하고 개선하는 데 중점을 둡니다. 이는 기후 변화 대응, 자원 효율성, 오염 방지, 생물 다양성 보전 등의 문제를 포함합니다. 환경 지속 가능성을 강화하는 것은 기업의 평판을 높이고, 법적 리스크를 줄이며, 장기적으로 비용 절감을 가능하게 합니다.

    ### 사회 (Social)
    사회 요소는 기업이 사회에 미치는 영향을 평가합니다. 이는 인권 보호, 노동 조건 개선, 지역 사회 기여, 다양성과 포용성 증진 등을 포함합니다. 긍정적인 사회적 영향을 미치는 기업은 직원의 사기와 생산성을 높이고, 고객과 지역 사회의 신뢰를 얻을 수 있습니다.

    ### 지배구조 (Governance)
    지배구조 요소는 기업의 경영 방식과 의사 결정 과정을 다룹니다. 이는 투명한 회계 관행, 이사회 구성, 경영진의 윤리적 행동, 주주 권리 보호 등을 포함합니다. 건전한 지배구조는 기업의 안정성과 지속 가능성을 보장하고, 투자자들의 신뢰를 증대시킵니다.

    ## 왜 ESG가 중요한가요?
    ### 1. 위험 관리
    ESG를 고려하는 기업은 환경적, 사회적, 법적 리스크를 더 잘 관리할 수 있습니다. 이는 장기적인 기업의 안정성과 성장을 도모합니다.

    ### 2. 투자 유치
    많은 투자자들이 ESG 요인을 고려하여 투자를 결정합니다. ESG를 충실히 이행하는 기업은 더 많은 투자 기회를 얻을 수 있습니다.

    ### 3. 평판 향상
    ESG에 대한 책임을 다하는 기업은 고객과 지역 사회로부터 더 높은 신뢰와 긍정적인 평판을 얻습니다. 이는 브랜드 가치를 높이고, 장기적으로 비즈니스 성공에 기여합니다.

    ### 4. 법적 준수
    전 세계적으로 ESG 관련 규제가 강화되고 있습니다. ESG 기준을 준수하는 기업은 법적 리스크를 최소화하고, 규제 변경에 유연하게 대응할 수 있습니다.

    ## 결론
    ESG는 단순한 트렌드가 아니라, 기업의 지속 가능성과 장기적인 성공을 위한 필수적인 요소입니다. 우리는 ESG 원칙을 바탕으로 책임 있는 경영을 실천하며, 환경 보호, 사회적 기여, 투명한 지배구조를 통해 더 나은 미래를 만들어 나가고자 합니다. 여러분의 지속적인 관심과 지지를 부탁드립니다.
    """)
st.write('')
st.write("""
    안녕하십니까 
    당사의 주식 추천 사이트에 방문해 주셔서 감사합니다. 저희는 기업의 환경(Environment), 사회(Social), 지배구조(Governance) 측면을 종합적으로 평가하여 사용자에게 최적의 주식을 추천하는 서비스를 제공합니다. 당사의 방법론은 다음과 같은 주요 요소를 포함합니다.

    ## 1. ESG 스코어 정의 및 평가 기준
    ESG 스코어는 기업의 지속 가능성과 책임 있는 경영을 측정하는 지표로, 다음과 같은 세 가지 주요 분야를 포함합니다:

    #### 환경(Environment)
    기업이 환경 보호를 위해 수행하는 노력과 성과를 평가합니다. 이는 온실가스 배출량, 에너지 효율성, 자원 관리, 재생 가능 에너지 사용 등으로 측정됩니다.

    #### 사회(Social)
    기업의 사회적 책임을 평가합니다. 직원 복지, 지역 사회에 대한 기여, 인권 보호, 공급망 관리 등과 같은 요소가 포함됩니다.

    #### 지배구조(Governance)
    기업의 관리 및 운영 방식에 대한 투명성과 책임성을 평가합니다. 이사회 구조, 경영진의 윤리, 부패 방지 정책, 주주 권리 보호 등이 고려됩니다.

    ## 2. 데이터 수집 및 분석
    저희는 ESG 스코어를 산출하기 위해 신뢰할 수 있는 다양한 데이터 소스를 활용합니다. 주요 데이터 소스에는 기업의 연례 보고서, 지속 가능성 보고서, 뉴스 및 미디어 기사, 그리고 전문 ESG 평가 기관의 리포트가 포함됩니다. 이 데이터를 바탕으로 저희는 다음과 같은 분석 과정을 진행합니다:

    #### 정량적 분석
    수치 데이터 및 KPI(핵심 성과 지표)를 기반으로 한 환경적, 사회적, 지배구조적 성과 분석을 수행합니다.

    #### 정성적 분석
    기업의 정책 및 이니셔티브, 업계 평판 등을 평가하여 ESG 관련 활동의 질적 측면을 분석합니다.

    ## 3. ESG 스코어 산출 및 가중치 적용
    각 기업의 ESG 성과를 기반으로 종합 스코어를 산출하며, 환경, 사회, 지배구조 각 항목에 대해 가중치를 적용하여 전체 ESG 스코어를 계산합니다. 가중치는 산업별, 지역별 특성에 맞추어 조정됩니다. 이 과정에서 기업의 업종과 특성을 반영하여 보다 정확한 평가가 이루어집니다.

    ## 4. 주식 추천 알고리즘
    ESG 스코어를 바탕으로 사용자 맞춤형 주식 추천 알고리즘을 운영합니다. 사용자의 투자 목표, 리스크 수용 범위, 관심 산업 등을 고려하여 ESG 점수가 높은 기업을 추천합니다. 알고리즘은 다음과 같은 요소를 반영합니다:

    #### ESG 스코어
    높은 ESG 스코어를 가진 기업을 우선 추천합니다.
    #### 재무 성과
    기업의 재무 건전성과 성장 잠재력도 함께 고려합니다.
    #### 시장 동향
    현재 시장 동향 및 산업별 특성을 반영하여 추천합니다.

    ## 5. 지속적인 모니터링 및 업데이트
    ESG 관련 정보는 지속적으로 업데이트되며, 기업의 ESG 스코어는 정기적으로 재평가됩니다. 이를 통해 최신 정보를 바탕으로 사용자에게 정확한 추천을 제공하며, 기업의 ESG 성과 변화에 신속하게 대응합니다.

    ## 6. 투명한 정보 제공
    저희는 사용자가 신뢰할 수 있는 정보를 제공하기 위해 ESG 스코어 산출 과정과 데이터 출처를 투명하게 공개합니다. 사용자는 각 기업의 ESG 성과에 대한 자세한 정보를 확인할 수 있으며, 이를 바탕으로 보다 나은 투자 결정을 내릴 수 있습니다.
    
    저희의 ESG 스코어 기반 주식 추천 서비스는 책임 있는 투자와 지속 가능한 성장을 지향합니다. 여러분의 투자 결정에 도움이 되기를 바랍니다.""")
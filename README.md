⚠️Alert: This is Ongoing Project

# 한일 관계 뉴스 크롤링 및 중립 기사 생성 서비스📰🤝🤖

> **프로젝트 수행 목적**  
> 이 프로젝트는 한국과 일본 간의 뉴스 기사를 크롤링하여 한일 간 이슈에 대한 중립적인 시각을 제공하는 것을 목표로 합니다. Streamlit을 기반으로 한 사용자 인터페이스(UI)를 통해 기사 크롤링, 군집화, 요약 및 중립 기사 생성의 모든 프로세스를 직관적으로 체험할 수 있습니다.

## 📖 프로젝트 개요

한국의 신문사(**[중앙일보](https://www.joongang.co.kr/), [경향신문](https://www.khan.co.kr/)**)와 일본의 신문사(**[요미우리신문](https://www.yomiuri.co.jp/), [아사히신문](https://www.asahi.com/)**)에서 한국과 일본 관련 주제를 중심으로 기사를 크롤링하여, 
크롤링한 기사들을 동일 주제끼리 군집화하고, OpenAI API를 활용해 다각적인 시각을 반영한 중립 기사를 생성합니다.  
이 서비스는 사용자가 직접 각 단계를 체험할 수 있도록 Streamlit으로 구축되었습니다.

### 주요 기능
- **기사 크롤링**: 한국 및 일본 언론사에서 특정 키워드로 기사 목록을 수집하고, 개별 기사 본문까지 크롤링합니다.
- **군집화**: 수집된 기사를 주제별로 그룹화하여 군집을 형성합니다.
- **요약**: 각 군집에서 상위 주제들을 요약하여 제공하고, 필요 시 OpenAI API를 활용한 요약 기능을 추가합니다.
- **중립 기사 생성**: 군집화된 기사의 핵심 내용들을 바탕으로 중립적인 시각의 기사를 생성하여 사용자에게 제공합니다.

## 🛠️ 기술 스택

- **크롤링**: Selenium, BeautifulSoup, Pandas
- **텍스트 처리 및 군집화**: Multilingual BERT, Scikit-learn
- **중립 기사 생성**: OpenAI API (GPT 모델)
- **인터페이스**: Streamlit
- **언어**: Python 3.8+

## 🚀 설치 및 실행 방법 (TBU)

1. **프로젝트 클론**
   ```bash
   git clone https://github.com/username/news-crawling-neutral-articles.git
   cd news-crawling-neutral-articles
   ```

2. **필수 라이브러리 설치**
   ```bash
   pip install -r requirements.txt
   ```

3. **Chromium 설치 (Colab 환경 시)**
   ```bash
   apt-get update
   apt-get install -y chromium-chromedriver
   ```

4. **Streamlit 앱 실행**
   ```bash
   streamlit run app.py
   ```

   로컬 서버에서 앱이 실행되며, http://localhost:8501 에서 접속하여 프로젝트의 주요 기능을 체험할 수 있습니다.

## 🖥️ 기능별 사용 방법

### 1. **크롤링 시작**
   - 메인 화면의 **크롤링 버튼**을 클릭하여 각 언론사로부터 키워드에 맞는 기사를 수집합니다.
   - 크롤링이 완료되면 기사 목록이 화면에 표시됩니다.
   - 기사 목록에는 기사 제목, 날짜, 링크 등이 포함됩니다.

### 2. **군집화**
   - **군집화 버튼**을 클릭하여 크롤링된 기사들을 주제별로 그룹화합니다.
   - 그룹화된 각 군집의 상위 주제를 요약하여 화면에 표시하며, 필요 시 OpenAI API를 통해 요약문을 생성합니다.

### 3. **중립 기사 생성**
   - **중립 기사 생성 버튼**을 클릭하여 군집화된 기사 내용을 바탕으로 중립적인 시각의 기사를 생성합니다.
   - 생성된 기사는 다각적인 이해를 반영하여 양국 간의 이슈에 대해 객관적인 정보를 제공합니다.

### 4. **결과 저장**
   - 생성된 기사를 PDF 혹은 텍스트 파일로 저장할 수 있는 옵션이 제공됩니다.

## 📂 아키텍처 설명

- **크롤링 모듈**: Selenium과 BeautifulSoup을 사용하여 각 언론사의 웹페이지에서 특정 키워드에 맞는 기사 목록과 링크를 수집합니다. 이후 개별 링크로 접근하여 기사 본문을 크롤링합니다.
- **군집화 모듈**: 다국어 처리가 가능한 BERT 모델을 사용하여 기사의 의미를 파악하고, 주제별로 그룹화합니다.
- **요약 및 중립 기사 생성 모듈**: OpenAI API를 통해 각 군집의 요약문을 생성하고, 중립적인 시각의 기사를 자동으로 작성합니다.
- **사용자 인터페이스 (UI)**: Streamlit을 사용하여 각 기능에 접근할 수 있는 버튼과 결과를 시각적으로 표시합니다.

## 📈 성능 및 품질 테스트

- 다양한 주제로 크롤링하여 군집화 및 중립 기사 생성 결과를 검토했습니다.
- 군집화된 주제별 요약 및 중립 기사 생성의 정확성은 사용자의 피드백을 통해 지속적으로 개선할 수 있습니다.

## 🔍 개선 가능성 및 차후 확장 기능

- **다국어 지원 개선**: 한국어와 일본어 외 다른 언어의 기사까지 처리할 수 있도록 확장 가능.
- **실시간 업데이트 기능**: 특정 시간 간격마다 기사를 자동으로 업데이트하여 최신 기사를 제공하는 기능 추가.
- **AI 모델 고도화**: BERT 외의 다른 NLP 모델을 사용하여 문서의 이해 및 요약 정확도를 높이는 방향으로 개선 가능.

## 💡 프로젝트의 의의

이 프로젝트는 한국과 일본 간의 편향된 언론 보도를 중립적인 시각으로 재구성하여 양국 간의 이해를 증진시키는 것을 목표로 합니다. 다양한 언어와 문화를 다루는 텍스트 데이터 처리, 군집화 및 생성 AI 모델의 활용을 통해 실제 데이터 과학과 NLP 기술을 포트폴리오에 효과적으로 녹여내는 사례로 활용될 수 있습니다.

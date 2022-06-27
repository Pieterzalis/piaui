import pandas as pd
import streamlit as st
from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import datetime,timedelta
import json
import requests
import pickle
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import joblib
from string import digits,punctuation

#import warnings
#warnings.filterwarnings('ignore')

from contextlib import suppress

def dados_texto(data_inicio,data_fim,senha):

    """pega os dados do impresso. Elas são publicadas na última semana do mês anterior"""

    url = "https://piaui.folha.uol.com.br/wp-admin/admin-ajax.php?action=restData"
    payload = json.dumps({
      "auth": senha,
      'qtd':1000,
      'datainicio':data_inicio,
      'datafim':data_fim})

    headers = {'Content-Type': 'application/json'}
    response_impresso = requests.request("GET", url, headers=headers, data=payload).json()

    return response_impresso

def text_clean(raw_html):

    """retira elementos de html do string"""

    cleantext = BeautifulSoup(raw_html, "lxml").text
    return cleantext


def palavras_chaves(materia):

    """considera apenas palavras-chaves, pronomes próprios e tiro = do igualdades"""

    lista_palavras_chaves = []
    nlp_materia = nlp_spacy(materia) ##
    for palavra in nlp_materia:### apenas nomes próprios e substantivos
        if ((palavra.pos_ == 'NOUN')) and len(palavra.text) > 2:
                lista_palavras_chaves.append(palavra.lemma_.lower())
    return ' '.join(lista_palavras_chaves)

nlp_spacy = spacy.load('pt_core_news_sm')
#with suppress(Exception):
      # your code

st.title("Leia mais revista piauí")

form = st.form(key='my_form')
name = form.text_input(label='Insira o link da matéria')
senha = form.text_input(label='Insira a senha')
submit_button = form.form_submit_button(label='Confirma')


#form_2 = st.form(key='form_2')
#senha = form_2.text_input(label='Insira a senha')
#submit_button2 = form_2.form_submit_button(label='Confirme')

html = urlopen(name).read()
soup = BeautifulSoup(html)
strips = list(soup.stripped_strings)

data_fim = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
data_inicio = (datetime.today() - timedelta(days=45)).strftime('%Y-%m-%d')
materias_json = dados_texto(data_inicio,data_fim,senha)

for i in materias_json:
    if i['title'] == strips[0]:
         materia_json = i

texto_comparacao = BeautifulSoup(materia_json['content'], "lxml").text

tfidf = pickle.load(open("tfidf.pickle", "rb"))

df_geral = pd.read_parquet('texto_limpo.parquet')
df_geral = df_geral[~df_geral['texto'].isna()]
df_geral['texto'] = df_geral['texto'].apply(text_clean)

base = tfidf.transform(df_geral['texto'])
punctuation = punctuation+'“‘'+'’”'

df_materia_tfidf = tfidf.transform([palavras_chaves(texto_comparacao)])
df_recomendacoes = pd.DataFrame(cosine_similarity(df_materia_tfidf,base).T,columns=['similaridade'])
df_recomendacoes['titulo'] = df_geral['titulo'].values
df_recomendacoes = df_recomendacoes.merge(df_geral[['titulo','data','urls']])[['titulo','data','urls','similaridade']]
df_recomendacoes.columns = ['título','data','url','sim.']
df_recomendacoes = df_recomendacoes.sort_values(by='sim.',ascending=False).head(20)

lista_hiperlinks = []
for i in df_recomendacoes.index:
    titulo = df_recomendacoes.loc[i,'título']
    link = df_recomendacoes.loc[i,'url']
    hiperlink = f'<a target="_blank" href="{link}">{titulo}</a>'
    lista_hiperlinks.append(hiperlink)

df_recomendacoes['matéria'] = lista_hiperlinks
df_recomendacoes['sim.'] = df_recomendacoes['sim.'].round(2)
df_recomendacoes['data'] = pd.to_datetime(df_recomendacoes['data'],format='%Y-%m-%d').dt.strftime('%Y-%m')

st.write("20 Matérias Mais Similares")
st.write(df_recomendacoes[['matéria','data','sim.']].reset_index(drop=True).to_html(escape=False,index=False), unsafe_allow_html=True)

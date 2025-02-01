# -*- coding: utf-8 -*-
"""Caravelas - 4 - Combinacao de Classificadores.ipynb
USA OS DATASETS COM AS PREDICOES
# Setup
"""

import numpy as np
import os
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.metrics import f1_score, precision_score, recall_score

MODO = 'full'
MODO = 'filtrada'
col_rotulo = 'rotulo_adaptado2'

if col_rotulo=='rotulo_adaptado2':
    prediction_bert = pd.read_csv('../../envRLBERT/treino/historico/bert_prediction_precision_bruta_471_R5_motivo.csv') # rotulo_adaptado2
    prediction_cnn = pd.read_csv('../../envImagemCP/main/historico/prediction_bestprecision_us_R205_motivo.csv') # rotulo_adaptado2
else:    
    prediction_bert = pd.read_csv('../../envRLBERT/treino/historico/bert_prediction_precision_bruta_471_R201_motivo.csv') # rotulo_adaptado1
    prediction_cnn = pd.read_csv('../../envImagemCP/main/historico/prediction_bestprecision_us_R132_motivo.csv') # rotulo_adaptado1

FILE_HISTORICO = 'historico/combinacao_'+col_rotulo+'_'+MODO+'_R3.csv'
FILE_PREDICTION = 'historico/combinacao_'+col_rotulo+'_prediction_'+MODO+'_R3.csv'
#BASE_TESTE = '../../bases/base_teste_bruta_filename.csv'
BASE_TESTE = '../../bases/base_teste_bruta_filename_motivo.csv'
df_test = pd.read_csv(BASE_TESTE, sep=';')
df_test['rotulo'] = df_test[col_rotulo].apply(lambda r: 1 if r=='ACEITA' else 0)

if MODO=='filtrada':
    df_test = df_test[df_test['motivo'].isin(['MIDIA','ATENDE OS CRITÉRIOS'])]
    prediction_bert = prediction_bert[prediction_bert['motivo'].isin(['MIDIA','ATENDE OS CRITÉRIOS'])]
    prediction_cnn = prediction_cnn[prediction_cnn['motivo'].isin(['MIDIA','ATENDE OS CRITÉRIOS'])]

print(len(df_test), len(df_test[df_test['motivo']=='MIDIA']))

y_true = df_test['rotulo']

print(len(prediction_bert), len(prediction_bert[prediction_bert['motivo']=='MIDIA']))

print(len(prediction_cnn), len(prediction_cnn[prediction_cnn['motivo']=='MIDIA']))
pred_bert = list(prediction_bert['proba'])
pred_cnn = list(prediction_cnn['proba'])

# Calcula as proba a priori das classes
priori0 = len(df_test[df_test['rotulo']==0])/len(df_test)
priori1 = len(df_test[df_test['rotulo']==1])/len(df_test)

def calc_produto(ppclass0, ppclass1):
  m0 = priori0 * np.prod(ppclass0)
  m1 = priori1 * np.prod(ppclass1)
  return np.argmax([m0, m1])

def calc_media(ppclass0, ppclass1):
  m0 = (np.sum(ppclass0)/len(ppclass0))
  m1 = (np.sum(ppclass1)/len(ppclass1))
  return np.argmax([m0, m1])

def calc_soma(ppclass0, ppclass1):
  m0 = priori0 + np.sum(ppclass0)
  m1 = priori1 + np.sum(ppclass1)
  return np.argmax([m0, m1])

def calc_maximo(ppclass0, ppclass1):
  m0 = np.max(ppclass0)
  m1 = np.max(ppclass1)
  return np.argmax([m0, m1])

def calc_minimo(ppclass0, ppclass1):
  m0 = np.min(ppclass0)
  m1 = np.min(ppclass1)
  return np.argmax([m0, m1])


y_pred = {'nula':[], 'media':[], 'soma':[],'produto':[], 'maximo':[], 'minimo':[]}
bert = True
cnn = True
rl = False
for i in range(0,len(df_test)):

    ppclass0 = []
    ppclass1 = []

    if bert is True:
#      ppclass1.append(pred_bert[i])
#      ppclass0.append(1-pred_bert[i])
      ppclass1.append(round(pred_bert[i],2))
      ppclass0.append(round(1-pred_bert[i],2))

    #print(ppclass1,ppclass0)
    if rl is True:
      pa = prob_tfidf[i][0] # proba ACEITA
      pr = prob_tfidf[i][1] # REJEITADA
      ppclass1.append(pa)
      ppclass0.append(pr)

    if cnn is True:
#      ppclass1.append(pred_cnn[i])
#      ppclass0.append(1-pred_cnn[i])
      ppclass1.append(round(pred_cnn[i],2))
      ppclass0.append(round(1-pred_cnn[i],2))
    
    #continue
    # FUSAO POR MEDIA
    classe_media = calc_media(ppclass0, ppclass1)
    y_pred['media'].append(classe_media)

    # FUSAO POR SOMA
    classe_soma = calc_soma(ppclass0, ppclass1)
    y_pred['soma'].append(classe_soma)

    # FUSAO POR PRODUTO
    classe_produto = calc_produto(ppclass0, ppclass1)
    y_pred['produto'].append(classe_produto)

    # FUSAO POR MAXIMO
    classe_maximo = calc_maximo(ppclass0, ppclass1)
    y_pred['maximo'].append(classe_maximo)

    classe_minimo = calc_minimo(ppclass0, ppclass1)
    y_pred['minimo'].append(classe_minimo)


fusao = ['media','produto','maximo','minimo','soma']
scores = {'rule':[],'precision':[],'recall':[],'f1':[]}
for f in fusao:
  scores['rule'].append(f)
  scores['f1'].append(round(f1_score(y_true, y_pred[f], pos_label=1, zero_division=0),3))
  scores['precision'].append(round(precision_score(y_true, y_pred[f], pos_label=1, zero_division=0),3))
  scores['recall'].append(round(recall_score(y_true, y_pred[f], pos_label=1, zero_division=0),3))
  df_test['pred_'+f] = y_pred[f]


df = pd.DataFrame(scores)
#df['bert'] = bert
#df['rl'] = rl
#df['cnn'] = cnn
print(df)

df.to_csv(FILE_HISTORICO)

df_test.to_csv(FILE_PREDICTION)


# -*- coding: utf-8 -*-
"""Caravelas - 4 - Combinacao de Classificadores.ipynb

# Setup
"""

import numpy as np
import os
import pandas as pd
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.metrics import f1_score, precision_score, recall_score

IMAGE_SIZE = (224, 224)
PATH = ''
PATH_IMG = '/home/hfrocha/dados/'

BASE_TESTE = '../../bases/base_teste_bruta_filename.csv'
df_test = pd.read_csv(BASE_TESTE, sep=';')
#df_test = df_test.sample(10)
df_test['rotulo'] = df_test['rotulo_adaptado2'].apply(lambda r: 1 if r=='ACEITA' else 0)
len(df_test)


datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_ds = datagen.flow_from_dataframe(df_test, directory = PATH_IMG, x_col = 'filename',
                                      y_col = 'rotulo_adaptado2',
                                      class_mode='binary',
                                      target_size=IMAGE_SIZE,
                                      shuffle=False)


"""
sample = df_test.sample(1)
texto = sample['texto'].item()
label = sample['rotulo'].item()
image = PATH_IMG+sample['filename'].item()
shortcode = sample['shortcode'].item()
print(shortcode, label, image)
"""

model_path = '../../envRLBERT/treino/modelos/tfidf_model_bruta_.pkl'
vectorizer_path = '../../envRLBERT/treino/modelos/tfidf_vetorizer_bruta_.pkl'
vectorizer = pickle.load(open(vectorizer_path,'rb'))
model_tfidf = pickle.load(open(model_path,'rb'))


""" Teste com um exemplar"""
vector = vectorizer.transform([texto])
rl_pred = model_tfidf.predict(vector)[0]
rl_prob = (model_tfidf.predict_proba(vector)[0])
print('Label:', rl_pred)
print('Prob classe positiva', rl_prob[0])
print('Prob classe negativa', rl_prob[1])



MODEL_PATH = '../../envRLBERT/treino/modelos/bestprecisionbruta_471_R5'
model_bert = tf.keras.models.load_model(MODEL_PATH, compile=False)


# Predizendo um exemplar
prediction = model_bert(tf.constant([texto]))
score = float(prediction[0])

print('Label:', (0 if score <=0.5 else 1))
print('Score: ', round(score, 2))
print('Prob classe positiva', round((100 * score), 2))
print('Prob classe negativa', round((100 * (1-score)), 2))


MODEL_PATH = '../../envImagemCP/main/modelos/bestprecision_us_R132'
model_cnn = tf.keras.models.load_model(MODEL_PATH, compile=False)


# Predizendo um exemplar
IMAGE_SIZE = (224, 224)
img = tf.keras.preprocessing.image.load_img(image, target_size=IMAGE_SIZE)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
predictions = model_cnn(img_array, training=False)
score = float(predictions[0])

print('Label:', (0 if score <=0.5 else 1))
print('Score: ', round(score, 2))
print('Prob classe positiva', round((100 * score), 2))
print('Prob classe negativa', round((100 * (1-score)), 2))



# Calcula as proba a priori das classes
#priori0 = len(df_test[df_test['rotulo_adaptado2']=='REJEITADA'])/len(df_test)
#priori1 = len(df_test[df_test['rotulo_adaptado2']=='ACEITA'])/len(df_test)
priori0 = 0.50
priori1 = 0.50

def calc_produto(ppclass0, ppclass1):
  m0 = priori0 * np.prod(ppclass0)
  m1 = priori1 * np.prod(ppclass1)
  return np.argmax([m0, m1])

def calc_media(ppclass0, ppclass1):
  m0 = priori0 + (np.sum(ppclass0)/len(ppclass0))
  m1 = priori1 + (np.sum(ppclass1)/len(ppclass1))
  return np.argmax([m0, m1])

def calc_soma(ppclass0, ppclass1):
  m0 = priori0 + np.sum(ppclass0)
  m1 = priori1 + np.sum(ppclass1)
  return np.argmax([m0, m1])

def calc_maximo(ppclass0, ppclass1):
  m0 = priori0 + np.max(ppclass0)
  m1 = priori1 + np.max(ppclass1)
  return np.argmax([m0, m1])

def calc_minimo(ppclass0, ppclass1):
  m0 = priori0 + np.min(ppclass0)
  m1 = priori1 + np.min(ppclass1)
  return np.argmax([m0, m1])


textos = df_test['texto']
y_true = df_test['rotulo']

# Predizando Lote com BERT
predictions_bert = model_bert.predict(textos, verbose=0)


# Predizendo Lote com TFIDF
vector = vectorizer.transform(textos)
prob_tfidf = model_tfidf.predict_proba(vector)
#rl_pred = model_tfidf.predict(vector)

"""
y_true_string = df_test['rotulo_adaptado2']
f1 = f1_score(y_true_string, rl_pred, pos_label='ACEITA', zero_division=0)
precision = precision_score(y_true_string, rl_pred, pos_label='ACEITA', zero_division=0)
recall = recall_score(y_true_string, rl_pred, pos_label='ACEITA', zero_division=0)
print("METRICAS", precision, recall, f1)
"""

# Predizendo Lote com CNN
predictions_cnn = model_cnn.predict(test_ds, verbose=0)


y_pred = {'nula':[], 'media':[], 'soma':[],'produto':[], 'maximo':[], 'minimo':[]}
bert = True
cnn = True
rl = False
for i in range(0,len(df_test)):

    ppclass0 = []
    ppclass1 = []

    if bert is True:
      ppclass1.append(predictions_bert[i][0])
      ppclass0.append(1-predictions_bert[i][0])
      #y_pred['nula'].append((0 if predictions_bert[i][0] <=0.5 else 1))
      #y_pred['nula'].append(round(predictions_bert[i][0],2))

    if rl is True:
      pa = prob_tfidf[i][0] # proba ACEITA
      pr = prob_tfidf[i][1] # REJEITADA
      ppclass1.append(pa)
      ppclass0.append(pr)
      #y_pred['nula'].append(np.argmax([pr,pa]))

    if cnn is True:
      ppclass1.append(round(predictions_cnn[i][0],2))
      ppclass0.append(round(1-predictions_cnn[i][0],2))
      #y_pred['nula'].append((0 if predictions_cnn[i][0] <=0.5 else 1))
      #y_pred['nula'].append(round(predictions_cnn[i][0],2))
    
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

#print(y_pred['nula'])
"""
f1 = f1_score(y_true, y_pred['nula'], pos_label=1, zero_division=0)
precision = precision_score(y_true, y_pred['nula'], pos_label=1, zero_division=0)
recall = recall_score(y_true, y_pred['nula'], pos_label=1, zero_division=0)
print(precision, recall,f1)
"""

fusao = ['media','produto','maximo','minimo','soma']
scores = {'rule':[],'precision':[],'recall':[],'f1':[]}
for f in fusao:
  scores['rule'].append(f)
  scores['f1'].append(f1_score(y_true, y_pred[f], pos_label=1, zero_division=0))
  scores['precision'].append(precision_score(y_true, y_pred[f], pos_label=1, zero_division=0))
  scores['recall'].append(recall_score(y_true, y_pred[f], pos_label=1, zero_division=0))


df = pd.DataFrame(scores)
df['bert'] = bert
df['rl'] = rl
df['cnn'] = cnn
print(df)

df.to_csv('historico/combinacao.csv')


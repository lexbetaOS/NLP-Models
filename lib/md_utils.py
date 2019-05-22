import numpy as np
import matplotlib.pyplot as plt
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
#model neural network
import tensorflow as tf


#Tokenizacion
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True Elimina puntuaciones

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts,stopSpanish):
    return [[word for word in simple_preprocess(str(doc)) if word not in stopSpanish] for doc in texts]

def make_bigrams(texts,bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts,bigram_mod,trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts,nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

    
#Graficar los verdaderos positivos y los falsos positivos
def graph_error_models(test_y,y_pred):
  num_rows=2
  num_cols=2
  temp=[]
  for i in range(len(test_y)):
      t1=sum([(d==0) for d in y_pred[test_y==i]])
      t2=sum([(d==1) for d in y_pred[test_y==i]])
      t3=sum([(d==2) for d in y_pred[test_y==i]])
      t4=sum([(d==3) for d in y_pred[test_y==i]])
      temp.append([('Consulta',t1),('Cambio de plan',t2),('Creditos',t3),('Excepcion',t4)])
  
  fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 12))
  n=0
  for i in range(num_rows):
      for j in range(num_cols): 
          labels, ys = zip(*temp[n])
          xs = np.arange(len(labels)) 
          width = 1
          axes[i,j].bar(xs, ys, width, align='center')
          t = f'Clase {n}'
          axes[i, j].set(title=t, xticks=xs, yticks=ys)
          n=n+1
  plt.show()

#
def category_to_target(clasificacion):
    clasi=np.array(clasificacion)
    y = np.zeros((len(clasificacion)),dtype=float)
    y1 = np.zeros((len(clasificacion),4),dtype=float)

    y[np.argwhere((clasi=='Consulta')==True)]=0
    y1[np.argwhere((clasi=='Consulta')==True)]=[1,0,0,0]
    y[np.argwhere((clasi=='Cambio de plan')==True)]=1
    y1[np.argwhere((clasi=='Cambio de plan')==True)]=[0,1,0,0]
    y[np.argwhere((clasi=='Creditos')==True)]=2
    y1[np.argwhere((clasi=='Creditos')==True)]=[0,0,1,0]
    y[np.argwhere((clasi=='Excepcion')==True)]=3
    y1[np.argwhere((clasi=='Excepcion')==True)]=[0,0,0,1]

    return y,y1

def corpus_to_input(corpus,dictionary):
    onehot_encoded = list()
    for corpu in corpus:
        letter = [0 for _ in range(len(dictionary))]
        for value in corpu:
            letter[value[0]] = value[1]
        onehot_encoded.append(letter)

    return np.array(onehot_encoded)


#model
def multilayer_perceptron_nn(input_tensor, weights, biases):
    layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    layer_1 = tf.nn.relu(layer_1_addition)
    
    # Hidden layer with RELU activation
    layer_2_multiplication = tf.matmul(layer_1, weights['h2'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
    layer_2 = tf.nn.relu(layer_2_addition)
    
    # Output layer 
    out_layer_multiplication = tf.matmul(layer_2, weights['out'])
    out_layer_addition = out_layer_multiplication + biases['out']
    
    return out_layer_addition
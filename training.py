
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt 
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import plotly.express as px
from plotly.offline import init_notebook_mode
import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import spacy

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Layer, Dense, Dropout, MultiHeadAttention, LayerNormalization, Input, GlobalAveragePooling1D
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer,DataCollatorWithPadding,TFAutoModel,DistilBertConfig,TFDistilBertModel, BertConfig, TFBertModel, TFRobertaModel
from datasets import load_dataset







nltk.download('omw-1.4')
tqdm.pandas()
# spacy_eng = spacy.load("en_core_web_sm")
nltk.download('stopwords')
lemm = WordNetLemmatizer()
init_notebook_mode(connected=True)
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (20,8)
plt.rcParams['font.size'] = 18



data = pd.read_csv('train.csv')

data.dropna(inplace=True)


def text_cleaning(d):
    questions = re.sub('\s+\n+', ' ', d)
    questions = re.sub('[^a-zA-Z0-9]', ' ', questions)
    questions = questions.lower()
    return questions


data['question1_cleaned'] = data['question1'].progress_apply(text_cleaning)
data['question2_cleaned'] = data['question2'].progress_apply(text_cleaning)

data['question1_length'] = data['question1_cleaned'].apply(lambda d: len(d.split()))
data['question2_length'] = data['question2_cleaned'].apply(lambda d: len(d.split()))

questions1 = data['question1_cleaned'].tolist()
questions2 = data['question2_cleaned'].tolist()

model_checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def encode_text(text, tokenizer):
    encoded = tokenizer.batch_encode_plus(
        text,
        add_special_tokens=True,
        max_length=50,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'
    )
    
    input_ids = np.array(encoded['input_ids'], dtype='int32')
    attention_masks = np.array(encoded['attention_mask'], dtype='int32')
    
    return {'input_ids': input_ids, 'attention_masks': attention_masks}


data = data.sample(400000)
train = data.iloc[:int(400000 * 0.80), :]
val = data.iloc[int(400000 * 0.80):, :]

d1_train = encode_text(train['question1_cleaned'].tolist(), tokenizer)
d2_train = encode_text(train['question2_cleaned'].tolist(), tokenizer)
d1_val = encode_text(val['question1_cleaned'].tolist(), tokenizer)
d2_val = encode_text(val['question2_cleaned'].tolist(), tokenizer)

y_train = train['is_duplicate'].values
y_val = val['is_duplicate'].values


class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
        
    def call(self, embedding1, embedding2):
        return tf.math.abs(embedding1 - embedding2)





BATCH_SIZE = 32


transformer_model = TFBertModel.from_pretrained(model_checkpoint)

input_ids1 = Input(shape=(None, ), name='input_ids1', dtype='int32')
input_ids2 = Input(shape=(None, ), name='input_ids2', dtype='int32')
input_masks1 = Input(shape=(None, ), name='attention_masks1', dtype='int32')
input_masks2 = Input(shape=(None, ), name='attention_masks2', dtype='int32')

embedding_layer1 = transformer_model(input_ids1, attention_mask=input_masks1).last_hidden_state
embedding_layer2 = transformer_model(input_ids2, attention_mask=input_masks2).last_hidden_state

embedding1 = GlobalAveragePooling1D()(embedding_layer1)
embedding2 = GlobalAveragePooling1D()(embedding_layer2)
l1_dist = L1Dist()(embedding1, embedding2)

dense = Dense(512, activation='relu')(l1_dist)
output = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=[input_ids1, input_masks1, input_ids2, input_masks2], outputs = output)
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001),
             metrics=['accuracy'])




for layer in model.layers[:5]:
    layer.trainable = False



early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                           patience=3, verbose=1,
                                           factor=0.3, min_lr=0.0000001)

history = model.fit((np.asarray(d1_train['input_ids']), np.asarray(d1_train['attention_masks']), np.asarray(d2_train['input_ids']),np.asarray(d2_train['attention_masks'])), 
                    y_train, batch_size=BATCH_SIZE, epochs=5,  
                    validation_data=((np.asarray(d1_val['input_ids']),np.asarray(d1_val['attention_masks']),np.asarray(d2_val['input_ids']),np.asarray(d2_val['attention_masks'])), y_val),
                    callbacks=[early_stopping, learning_rate_reduction])

























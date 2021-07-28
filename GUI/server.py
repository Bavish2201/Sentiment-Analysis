import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from flask import Flask, render_template, request, url_for
from preprocessing import encode_examples
import tensorflow as tf
from transformers import DistilBertConfig, TFDistilBertForSequenceClassification, DistilBertTokenizer
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer

from transformers import TFBertForSequenceClassification, BertTokenizer

def convert_example_to_feature(review):
  return tokenizer.encode_plus(review, 
                add_special_tokens = True, # add [CLS], [SEP]
                max_length = 30, # max length of the text that can go to BERT
                truncation=True,
                pad_to_max_length  = True, # add [PAD] tokens
                return_attention_mask = True, # add attention mask to not focus on pad tokens
              )


def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
  return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label

def encode_examples(ds, limit=-1):
  # prepare list, so that we can build up final TensorFlow dataset from slices.
  input_ids_list = []
  token_type_ids_list = []
  attention_mask_list = []
  label_list = []
    
  for index, row in ds.iterrows():
    review = row[0]
    label = row[1]
    bert_input = convert_example_to_feature(review)
  
    input_ids_list.append(bert_input['input_ids'])
    token_type_ids_list.append(bert_input['token_type_ids'])
    attention_mask_list.append(bert_input['attention_mask'])
    label_list.append([label - 1])

  return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)

def predict_fn(x):
  input_ids_list = []
  attention_mask_list = []
  token_type_ids_list = []
  label_list = []
  for review in x:
    bert_input = convert_example_to_feature(review)
  
    input_ids_list.append(bert_input['input_ids'])
    attention_mask_list.append(bert_input['attention_mask'])
    token_type_ids_list.append(bert_input['token_type_ids'])
    label_list.append([1])

  x = tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict).batch(8)
  probs = model.predict(x).logits
  probs = np.array(tf.nn.softmax(probs))
  return probs

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sentiment_analysis', methods=['POST'])
def sentiment_analysis():
    review = request.form['review']
    review_encoded = encode_examples(pd.DataFrame({'review': [review], 'label': [1]})).batch(1)
    logits = model.predict(review_encoded).logits
    probs = np.array(tf.nn.softmax(logits))[0]
    rating = np.argmax(probs) + 1

    explainer = LimeTextExplainer(class_names=[1, 2, 3, 4, 5])
    exp = explainer.explain_instance(review, predict_fn, num_features=6, labels=[0, 1, 2, 3, 4], num_samples=500)
    return render_template('index.html', review_text = review, rating = rating, probs = probs, exp = exp.as_html())


if __name__ == '__main__':

    # model = tf.keras.models.load_model('model/model')
    model = TFBertForSequenceClassification.from_pretrained("bert/model")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    app.run()
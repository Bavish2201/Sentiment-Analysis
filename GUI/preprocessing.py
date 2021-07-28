from transformers import DistilBertConfig, TFDistilBertForSequenceClassification, DistilBertTokenizer

def convert_example_to_feature(review):
  return tokenizer.encode_plus(review, 
                add_special_tokens = True, # add [CLS], [SEP]
                max_length = 30, # max length of the text that can go to BERT
                truncation=True,
                pad_to_max_length  = True, # add [PAD] tokens
                return_attention_mask = True, # add attention mask to not focus on pad tokens
              )


def map_example_to_dict(input_ids, attention_masks, label):
  return {
      "input_ids": input_ids,
      # "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label

def encode_examples(ds, limit=-1):
  # prepare list, so that we can build up final TensorFlow dataset from slices.
  input_ids_list = []
  # token_type_ids_list = []
  attention_mask_list = []
  label_list = []
    
  for index, row in ds.iterrows():
    review = row[0]
    label = row[1]
    bert_input = convert_example_to_feature(review)
  
    input_ids_list.append(bert_input['input_ids'])
    # token_type_ids_list.append(bert_input['token_type_ids'])
    attention_mask_list.append(bert_input['attention_mask'])
    label_list.append([label - 1])

  return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, label_list)).map(map_example_to_dict)

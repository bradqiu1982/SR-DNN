
import io
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds

import collections
import pathlib
import re
import string


from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras import utils

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import tensorflow_text as tf_text

import numpy
import sys

import tensorflow_hub as hub

import shutil
from official.nlp import optimization



def Train():
	dataset, info = tfds.load( 'imdb_reviews/subwords8k',  with_info=True, as_supervised=True)

	train_dataset, test_dataset = dataset['train'], dataset['test']

	encoder = info.features['text'].encoder


	BUFFER_SIZE = 10000
	BATCH_SIZE = 64

	train_dataset = train_dataset.shuffle(BUFFER_SIZE)
	train_dataset = train_dataset.padded_batch(BATCH_SIZE)
	test_dataset = test_dataset.padded_batch(BATCH_SIZE)

	model = tf.keras.Sequential([
	    tf.keras.layers.Embedding(encoder.vocab_size, 64, mask_zero=True),
	    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
	    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
	    tf.keras.layers.Dense(64, activation='relu'),
	    tf.keras.layers.Dropout(0.3),
	    tf.keras.layers.Dense(1)
	])

	model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])

	history = model.fit(train_dataset, epochs=10, validation_data=test_dataset, validation_steps=30)

	test_loss, test_acc = model.evaluate(test_dataset)

	model.save('./txtmodel.h5')


#Train()


def gentextdataset():
	# data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'
	# dataset_dir = utils.get_file(
	# origin=data_url,
	# untar=True,
	# cache_dir='stack_overflow',
	# cache_subdir='')

	# dataset_dir = pathlib.Path(dataset_dir).parent
	# print(dataset_dir)

	DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
	FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']
	for name in FILE_NAMES:
		text_dir = utils.get_file(name, origin=DIRECTORY_URL + name)
		parent_dir = pathlib.Path(text_dir).parent
		list(parent_dir.iterdir())


#gentextdataset()


def replaceescapchars(text, label):
	# text = tf.expand_dims(text, -1)
	return tf.strings.regex_replace(tf.strings.regex_replace(tf.strings.regex_replace(text,"[[:punct:]]"," "),"[[:digit:]]",""),"[[:^ascii:]]",""),label

AUTOTUNE = tf.data.AUTOTUNE
def configure_dataset(dataset):
	return dataset.cache().prefetch(buffer_size=AUTOTUNE)

def classstackovertxt():
	batch_size = 32
	seed = 42
	train_dir = 'D:/PlanningForCast/condaenv/stackover-txt/train'
	
	raw_train_ds = preprocessing.text_dataset_from_directory(
	train_dir,
	batch_size=batch_size,
	validation_split=0.2,
	subset='training',
	seed=seed)

	class_names = raw_train_ds.class_names
	raw_train_ds = raw_train_ds.map(replaceescapchars)

	raw_val_ds = preprocessing.text_dataset_from_directory(
	train_dir,
	batch_size=batch_size,
	validation_split=0.2,
	subset='validation',
	seed=seed)
	raw_val_ds = raw_val_ds.map(replaceescapchars)

	test_dir = 'D:/PlanningForCast/condaenv/stackover-txt/test'
	raw_test_ds = preprocessing.text_dataset_from_directory(test_dir, batch_size=batch_size)
	raw_test_ds = raw_test_ds.map(replaceescapchars)


	tokenizer = tf_text.UnicodeScriptTokenizer()

	VOCAB_SIZE = 50000
	MAX_SEQUENCE_LENGTH = 800

	binary_vectorize_layer = TextVectorization( max_tokens=VOCAB_SIZE,split=tokenizer.tokenize, output_mode='binary')
	int_vectorize_layer = TextVectorization( max_tokens=VOCAB_SIZE,split=tokenizer.tokenize, output_mode='int', output_sequence_length=MAX_SEQUENCE_LENGTH)

	train_text = raw_train_ds.map(lambda text, labels: text)
	binary_vectorize_layer.adapt(train_text)
	int_vectorize_layer.adapt(train_text)


	def binary_vectorize_text(text, label):
		text = tf.expand_dims(text, -1)
		return binary_vectorize_layer(text), label

	def int_vectorize_text(text, label):
		text = tf.expand_dims(text, -1)
		return int_vectorize_layer(text), label


	# print(len(binary_vectorize_layer.get_vocabulary()))
	# print(len(int_vectorize_layer.get_vocabulary()))


	# print(len(class_names))
	# text_batch, label_batch = next(iter(raw_train_ds))
	# first_question, first_label = text_batch[0], label_batch[0]
	# print("Question", first_question)
	# print("Label", first_label)	

	# numpy.set_printoptions(threshold=sys.maxsize)
	# print("'binary' vectorized question:",  binary_vectorize_text(first_question, first_label)[0])

	# encoded_example,lab = int_vectorize_text(first_question, first_label)
	# print("'int' vectorized question:", encoded_example)
	# vocab = numpy.array(int_vectorize_layer.get_vocabulary())
	# print("Round-trip: ", " ".join(vocab[encoded_example]))

	# print("1 ---> ", binary_vectorize_layer.get_vocabulary()[1])
	# print("2 ---> ", binary_vectorize_layer.get_vocabulary()[2])
	# print("3 ---> ", binary_vectorize_layer.get_vocabulary()[2])

	# print("Vocabulary size: {}".format(len(int_vectorize_layer.get_vocabulary())))


	binary_train_ds = raw_train_ds.map(binary_vectorize_text)
	binary_val_ds = raw_val_ds.map(binary_vectorize_text)
	binary_test_ds = raw_test_ds.map(binary_vectorize_text)

	int_train_ds = raw_train_ds.map(int_vectorize_text)
	int_val_ds = raw_val_ds.map(int_vectorize_text)
	int_test_ds = raw_test_ds.map(int_vectorize_text)



	binary_train_ds = configure_dataset(binary_train_ds)
	binary_val_ds = configure_dataset(binary_val_ds)
	binary_test_ds = configure_dataset(binary_test_ds)

	int_train_ds = configure_dataset(int_train_ds)
	int_val_ds = configure_dataset(int_val_ds)
	int_test_ds = configure_dataset(int_test_ds)


	binary_model = tf.keras.Sequential([layers.Dense(4)])
	binary_model.compile( loss=losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
	history = binary_model.fit( binary_train_ds, validation_data=binary_val_ds, epochs=10)

	# int_model = tf.keras.Sequential([
	# layers.Embedding(VOCAB_SIZE+1, 64, mask_zero=True),
	# tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
	# tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
	# tf.keras.layers.Dense(64, activation='relu'),
	# layers.Dropout(0.5),
	# layers.Dense(len(class_names))
	# ])

	# int_model.compile(
	# loss=losses.SparseCategoricalCrossentropy(from_logits=True),
	# optimizer=tf.keras.optimizers.Adam(1e-4),
	# metrics=['accuracy'])
	# history = int_model.fit(int_train_ds, validation_data=int_val_ds, epochs=8)

	binary_loss, binary_accuracy = binary_model.evaluate(binary_test_ds)
	# int_loss, int_accuracy = int_model.evaluate(int_test_ds)

	print("Binary model accuracy: {:2.2%}".format(binary_accuracy))
	# print("Int model accuracy: {:2.2%}".format(int_accuracy))

	export_model = tf.keras.Sequential(
		[binary_vectorize_layer
		, binary_model
		,layers.Activation('sigmoid')])

	export_model.compile(
		loss=losses.SparseCategoricalCrossentropy(from_logits=False),
		optimizer='adam',
		metrics=['accuracy'])

	loss, accuracy = export_model.evaluate(raw_test_ds)
	print("Accuracy: {:2.2%}".format(accuracy))
	
	def get_string_labels(predicted_scores_batch):
		predicted_int_labels = tf.argmax(predicted_scores_batch, axis=1)
		predicted_labels = tf.gather(class_names, predicted_int_labels)
		return predicted_labels

	inputs = [ "how do I extract keys from a dict into a list?",  # python
				"debug public static void main(string[] args) {...}",  # java
			]
	predicted_scores = export_model.predict(inputs)
	predicted_labels = get_string_labels(predicted_scores)
	for input, label in zip(inputs, predicted_labels):
		print("Question: ", input)
		print("Predicted label: ", label.numpy())



#classstackovertxt()


def getstacktrainds():
	batch_size = 32
	seed = 42
	train_dir = 'D:/PlanningForCast/condaenv/stackover-txt/train'
	
	raw_train_ds = preprocessing.text_dataset_from_directory(
	train_dir,
	batch_size=batch_size,
	validation_split=0.2,
	subset='training',
	seed=seed)

	class_names = raw_train_ds.class_names
	raw_train_ds = raw_train_ds.map(replaceescapchars)
	raw_train_ds = configure_dataset(raw_train_ds)

	raw_val_ds = preprocessing.text_dataset_from_directory(
	train_dir,
	batch_size=batch_size,
	validation_split=0.2,
	subset='validation',
	seed=seed)
	raw_val_ds = raw_val_ds.map(replaceescapchars)
	raw_val_ds = configure_dataset(raw_val_ds)

	return raw_train_ds,raw_val_ds

def getstacktestds():
	batch_size = 32
	seed = 42
	test_dir = 'D:/PlanningForCast/condaenv/stackover-txt/test'
	raw_test_ds = preprocessing.text_dataset_from_directory(test_dir, batch_size=batch_size)
	raw_test_ds = raw_test_ds.map(replaceescapchars)
	raw_test_ds = configure_dataset(raw_test_ds)

	return raw_test_ds


def classbyhub():

	raw_train_ds,raw_val_ds = getstacktrainds()

	# int_model = tf.keras.models.Sequential()
	# # int_model.add(hub.KerasLayer('./nnlm-en-dim128_2',input_shape=[], dtype=tf.string , trainable=True))
	# int_model.add(hub.KerasLayer('./universal-sentence-encoder_4',input_shape=[], dtype=tf.string , trainable=True))
	# int_model.add(keras.layers.Dense(16, activation='relu'))
	# int_model.add(layers.Dense(4))
	# int_model.summary()
	
	# int_model.compile( loss=losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
	# int_model.fit(raw_train_ds, validation_data=raw_val_ds, epochs=2)

	# int_model.save('./hub_textclassfy.h5')


	raw_test_ds = getstacktestds()
	export_model = tf.keras.models.load_model('./hub_textclassfy.h5', custom_objects={'KerasLayer': hub.KerasLayer})
	loss, accuracy = export_model.evaluate(raw_test_ds)
	print("Accuracy: {:2.2%}".format(accuracy))

#classbyhub()

def getbert():
	bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8' 
	#bert_model_name = 'experts_pubmed' 

	map_name_to_handle = {
	'bert_en_uncased_L-12_H-768_A-12':'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
	'bert_en_cased_L-12_H-768_A-12':'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
	'bert_multi_cased_L-12_H-768_A-12':'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
	'small_bert/bert_en_uncased_L-2_H-128_A-2':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
	'small_bert/bert_en_uncased_L-2_H-256_A-4':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
	'small_bert/bert_en_uncased_L-2_H-512_A-8':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
	'small_bert/bert_en_uncased_L-2_H-768_A-12':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
	'small_bert/bert_en_uncased_L-4_H-128_A-2':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
	'small_bert/bert_en_uncased_L-4_H-256_A-4':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
	'small_bert/bert_en_uncased_L-4_H-512_A-8':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
	'small_bert/bert_en_uncased_L-4_H-768_A-12':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
	'small_bert/bert_en_uncased_L-6_H-128_A-2':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
	'small_bert/bert_en_uncased_L-6_H-256_A-4':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
	'small_bert/bert_en_uncased_L-6_H-512_A-8':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
	'small_bert/bert_en_uncased_L-6_H-768_A-12':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
	'small_bert/bert_en_uncased_L-8_H-128_A-2':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
	'small_bert/bert_en_uncased_L-8_H-256_A-4':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
	'small_bert/bert_en_uncased_L-8_H-512_A-8':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
	'small_bert/bert_en_uncased_L-8_H-768_A-12':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
	'small_bert/bert_en_uncased_L-10_H-128_A-2':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
	'small_bert/bert_en_uncased_L-10_H-256_A-4':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
	'small_bert/bert_en_uncased_L-10_H-512_A-8':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
	'small_bert/bert_en_uncased_L-10_H-768_A-12':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
	'small_bert/bert_en_uncased_L-12_H-128_A-2':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
	'small_bert/bert_en_uncased_L-12_H-256_A-4':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
	'small_bert/bert_en_uncased_L-12_H-512_A-8':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
	'small_bert/bert_en_uncased_L-12_H-768_A-12':'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
	'albert_en_base':'https://tfhub.dev/tensorflow/albert_en_base/2',
	'electra_small':'https://tfhub.dev/google/electra_small/2',
	'electra_base':'https://tfhub.dev/google/electra_base/2',
	'experts_pubmed':'https://tfhub.dev/google/experts/bert/pubmed/2',
	'experts_wiki_books':'https://tfhub.dev/google/experts/bert/wiki_books/2',
	'talking-heads_base':'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
	}

	map_model_to_preprocess = {
	'bert_en_uncased_L-12_H-768_A-12':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'bert_en_cased_L-12_H-768_A-12':'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
	'small_bert/bert_en_uncased_L-2_H-128_A-2':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-2_H-256_A-4':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-2_H-512_A-8':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-2_H-768_A-12':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-4_H-128_A-2':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-4_H-256_A-4':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-4_H-512_A-8':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-4_H-768_A-12':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-6_H-128_A-2':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-6_H-256_A-4':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-6_H-512_A-8':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-6_H-768_A-12':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-8_H-128_A-2':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-8_H-256_A-4':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-8_H-512_A-8':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-8_H-768_A-12':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-10_H-128_A-2':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-10_H-256_A-4':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-10_H-512_A-8':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-10_H-768_A-12':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-12_H-128_A-2':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-12_H-256_A-4':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-12_H-512_A-8':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-12_H-768_A-12':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'bert_multi_cased_L-12_H-768_A-12':'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
	'albert_en_base':'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
	'electra_small':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'electra_base':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'experts_pubmed':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'experts_wiki_books':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'talking-heads_base':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	}

	tfhub_handle_encoder = map_name_to_handle[bert_model_name]
	tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]
	return tfhub_handle_encoder,tfhub_handle_preprocess

def trainbert():
	raw_train_ds,raw_val_ds = getstacktrainds()

	tfhub_handle_encoder,tfhub_handle_preprocess = getbert()

	text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
	preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
	encoder_inputs = preprocessing_layer(text_input)

	encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
	outputs = encoder(encoder_inputs)
	net = outputs['pooled_output']
	net = tf.keras.layers.Dropout(0.3)(net)
	net = tf.keras.layers.Dense(4, activation='softmax', name='classifier')(net)
	model = tf.keras.Model(text_input, net)
	model.summary()

	epochs = 5
	steps_per_epoch = tf.data.experimental.cardinality(raw_train_ds).numpy()
	num_train_steps = steps_per_epoch * epochs
	num_warmup_steps = int(0.1*num_train_steps)

	init_lr = 3e-5
	optimizer = optimization.create_optimizer(init_lr=init_lr,
	num_train_steps=num_train_steps,
	num_warmup_steps=num_warmup_steps,
	optimizer_type='adamw')

	model.compile( loss=losses.SparseCategoricalCrossentropy(from_logits=False), optimizer=optimizer, metrics=['accuracy'])
	model.fit(raw_train_ds, validation_data=raw_val_ds, epochs=epochs)


trainbert()


def getVocTable(escapds):

	tokenizer = tf_text.UnicodeScriptTokenizer()
	def tokenize(text, unused_label):
		lower_case = tf_text.case_fold_utf8(text)
		return tokenizer.tokenize(lower_case)

	tokenized_ds = escapds.map(tokenize)

	# for text_batch in tokenized_ds.take(5):
	# 	print("Tokens: ", text_batch.numpy())

	tokenized_ds = configure_dataset(tokenized_ds)

	vocab_dict = collections.defaultdict(lambda: 0)
	for toks in tokenized_ds.as_numpy_iterator():
		for tok in toks:
			vocab_dict[tok] += 1

	vocab = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
	vocab = [token for token, count in vocab]
	vocab_size = len(vocab)
	# print("Vocab size: ", vocab_size)
	# print("First five vocab entries:", vocab[:5])
	keys = vocab

	values = range(2, len(vocab) + 2)
	init = tf.lookup.KeyValueTensorInitializer( keys, values, key_dtype=tf.string, value_dtype=tf.int64)
	num_oov_buckets = 1
	vocab_table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets)
	return vocab_table,vocab_size+2,vocab

def classhomlet():
	FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']
	
	def labeler(example, index):
		return example, tf.cast(index, tf.int64)

	labeled_data_sets = []
	for i, file_name in enumerate(FILE_NAMES):
		lines_dataset = tf.data.TextLineDataset('D:/PlanningForCast/condaenv/homlet/'+file_name)
		labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
		labeled_data_sets.append(labeled_dataset)

	BUFFER_SIZE = 50000
	BATCH_SIZE = 64
	VALIDATION_SIZE = 5000
	MAX_SEQUENCE_LENGTH = 800

	all_labeled_data = labeled_data_sets[0]
	for labeled_dataset in labeled_data_sets[1:]:
		all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
	all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)

	# for text, label in all_labeled_data.take(10):
	# 	print("Sentence: ", text.numpy())
	# 	print("Label:", label.numpy())

	
	escapds = all_labeled_data.map(replaceescapchars)
	tokenizer = tf_text.UnicodeScriptTokenizer()
	vocab_table,vocab_size,vocab = getVocTable(escapds)

	def preprocess_text(text, label):
		standardized = tf_text.case_fold_utf8(text)
		tokenized = tokenizer.tokenize(standardized)
		vectorized = vocab_table.lookup(tokenized)
		return vectorized, label

	# example_text, example_label = next(iter(escapds))
	# print("Sentence: ", example_text.numpy())
	# vectorized_text, example_label = preprocess_text(example_text, example_label)
	# print("Vectorized sentence: ", vectorized_text.numpy())

	all_encoded_data = escapds.map(preprocess_text)

	train_data = all_encoded_data.skip(VALIDATION_SIZE).shuffle(BUFFER_SIZE)
	validation_data = all_encoded_data.take(VALIDATION_SIZE)

	train_data = train_data.padded_batch(BATCH_SIZE)
	validation_data = validation_data.padded_batch(BATCH_SIZE)

	# sample_text, sample_labels = next(iter(validation_data))
	# print("Text batch shape: ", sample_text.shape)
	# print("Label batch shape: ", sample_labels.shape)
	# print("First text example: ", sample_text[0])
	# print("First label example: ", sample_labels[0])

	train_data = configure_dataset(train_data)
	validation_data = configure_dataset(validation_data)


	# int_model = tf.keras.Sequential([
	# layers.Embedding(vocab_size, 64, mask_zero=True),
	# tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
	# tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
	# tf.keras.layers.Dense(64, activation='relu'),
	# layers.Dropout(0.5),
	# layers.Dense(3)
	# ])

	int_model.compile(
	loss=losses.SparseCategoricalCrossentropy(from_logits=True),
	optimizer=tf.keras.optimizers.Adam(1e-4),
	metrics=['accuracy'])

	history = int_model.fit(train_data, validation_data=validation_data, epochs=3)

	# preprocess_layer = TextVectorization(
	# max_tokens=vocab_size,
	# standardize=tf_text.case_fold_utf8,
	# split=tokenizer.tokenize,
	# output_mode='int',
	# output_sequence_length=MAX_SEQUENCE_LENGTH)
	# preprocess_layer.set_vocabulary(vocab)

	# export_model = tf.keras.Sequential( [preprocess_layer, int_model, layers.Activation('sigmoid')])

	# export_model.compile(
	# loss=losses.SparseCategoricalCrossentropy(from_logits=False),
	# optimizer='adam',
	# metrics=['accuracy'])

	# # test_ds = escapds.take(VALIDATION_SIZE).batch(BATCH_SIZE)
	# # test_ds = configure_dataset(test_ds)
	# # loss, accuracy = export_model.evaluate(test_ds)
	# # print("Loss: ", loss)
	# # print("Accuracy: {:2.2%}".format(accuracy))

	# inputs = [
	# "Join'd to th' Ionians with their flowing robes,",  # Label: 1
	# "the allies, and his armour flashed about him so that he seemed to all",  # Label: 2
	# "And with loud clangor of his arms he fell.",  # Label: 0
	# ]

	# predicted_scores = export_model.predict(inputs)
	# predicted_labels = tf.argmax(predicted_scores, axis=1)

	# for input, label in zip(inputs, predicted_labels):
	# 	print("Question: ", input)
	# 	print("Predicted label: ", label.numpy())


#classhomlet()



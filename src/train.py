#
# Copyright (c) 2020. Asutosh Nayak (nayak.asutosh@ymail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import RandomUniform, RandomNormal
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
from sklearn.utils.class_weight import compute_class_weight

from data_generator import DataGenerator, plt
from logger import Logger
from secrets import api_key
from utils import download_save, seconds_to_minutes

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

args = sys.argv  # a list of the arguments provided (str)
print("running stock_cnn.py", args)
pd.options.display.width = 0
company_code = args[1]
strategy_type = args[2]
ROOT_PATH = ".."
iter_changes = "fresh_rolling_train"  # label for changes in this run iteration
INPUT_PATH = os.path.join(ROOT_PATH, "stock_history", company_code)
OUTPUT_PATH = os.path.join(ROOT_PATH, "outputs", iter_changes)
LOG_PATH = OUTPUT_PATH + os.sep + "logs"
LOG_FILE_NAME_PREFIX = "log_{}_{}_{}".format(company_code, strategy_type, iter_changes)
PATH_TO_STOCK_HISTORY_DATA = os.path.join(ROOT_PATH, "stock_history")

if not os.path.exists(INPUT_PATH):
    os.makedirs(INPUT_PATH)
    print("Input Directory created", INPUT_PATH)

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    print("Output Directory created", OUTPUT_PATH)

BASE_URL = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED" \
           "&outputsize=full&apikey=" + api_key + "&datatype=csv&symbol="  # api key from alpha vantage service
data_file_name = company_code + ".csv"
PATH_TO_COMPANY_DATA = os.path.join(PATH_TO_STOCK_HISTORY_DATA, company_code, data_file_name)
print(INPUT_PATH)
print(OUTPUT_PATH)
print(PATH_TO_COMPANY_DATA)

logger = Logger(LOG_PATH, LOG_FILE_NAME_PREFIX)

print("Tensorflow devices {}".format(tf.test.gpu_device_name()))

start_time = time.time()
data_gen = DataGenerator(company_code, PATH_TO_COMPANY_DATA, OUTPUT_PATH, strategy_type, False, logger)
# exit here, since training is done with stock_keras.ipynb.
# comment sys.exit() if you want to try out rolling window training.
sys.exit()

################################## Model creation #######################################


def f1_weighted(y_true, y_pred):
    y_true_class = tf.math.argmax(y_true, axis=1, output_type=tf.dtypes.int32)
    y_pred_class = tf.math.argmax(y_pred, axis=1, output_type=tf.dtypes.int32)
    conf_mat = tf.math.confusion_matrix(y_true_class, y_pred_class)  # can use conf_mat[0, :], tf.slice()
    # precision = TP/TP+FP, recall = TP/TP+FN
    rows, cols = conf_mat.get_shape()
    size = y_true_class.get_shape()[0]
    precision = tf.constant([0, 0, 0])  # change this to use rows/cols as size
    recall = tf.constant([0, 0, 0])
    class_counts = tf.constant([0, 0, 0])

    def get_precision(i, conf_mat):
        print("prec check", conf_mat, conf_mat[i, i], tf.reduce_sum(conf_mat[:, i]))
        precision[i].assign(conf_mat[i, i] / tf.reduce_sum(conf_mat[:, i]))
        recall[i].assign(conf_mat[i, i] / tf.reduce_sum(conf_mat[i, :]))
        tf.add(i, 1)
        return i, conf_mat, precision, recall

    def tf_count(i):
        elements_equal_to_value = tf.equal(y_true_class, i)
        as_ints = tf.cast(elements_equal_to_value, tf.int32)
        count = tf.reduce_sum(as_ints)
        class_counts[i].assign(count)
        tf.add(i, 1)
        return count

    def condition(i, conf_mat):
        return tf.less(i, 3)

    i = tf.constant(3)
    i, conf_mat = tf.while_loop(condition, get_precision, [i, conf_mat])

    i = tf.constant(3)
    c = lambda i: tf.less(i, 3)
    b = tf_count(i)
    tf.while_loop(c, b, [i])

    weights = tf.math.divide(class_counts, size)
    numerators = tf.math.multiply(tf.math.multiply(precision, recall), tf.constant(2))
    denominators = tf.math.add(precision, recall)
    f1s = tf.math.divide(numerators, denominators)
    weighted_f1 = tf.reduce_sum(tf.math.multiply(f1s, weights))
    return weighted_f1


def f1_metric(y_true, y_pred):
    """
    this calculates precision & recall
    """

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # mistake: y_pred of 0.3 is also considered 1
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    # y_true_class = tf.math.argmax(y_true, axis=1, output_type=tf.dtypes.int32)
    # y_pred_class = tf.math.argmax(y_pred, axis=1, output_type=tf.dtypes.int32)
    # conf_mat = tf.math.confusion_matrix(y_true_class, y_pred_class)
    # tf.Print(conf_mat, [conf_mat], "confusion_matrix")

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


get_custom_objects().update({"f1_metric": f1_metric, "f1_weighted": f1_weighted})

params = {'batch_size': 60, 'conv2d_layers': {'conv2d_do_1': 0.0, 'conv2d_filters_1': 30,
                                               'conv2d_kernel_size_1': 2, 'conv2d_mp_1': 2, 'conv2d_strides_1': 1,
                                               'kernel_regularizer_1':0.0, 'conv2d_do_2': 0.01, 'conv2d_filters_2': 10,
                                               'conv2d_kernel_size_2': 2, 'conv2d_mp_2': 2, 'conv2d_strides_2': 2,
                                               'kernel_regularizer_2':0.0, 'layers': 'two'},
           'dense_layers': {'dense_do_1': 0.07, 'dense_nodes_1': 100, 'kernel_regularizer_1':0.0, 'layers': 'one'},
           'epochs': 3000, 'lr': 0.001, 'optimizer': 'adam', 'input_dim_1': 15, 'input_dim_2': 15, 'input_dim_3': 3}


def create_model_cnn(params):
    model = Sequential()

    print("Training with params {}".format(params))
    # (batch_size, timesteps, data_dim)
    # x_train, y_train = get_data_cnn(df, df.head(1).iloc[0]["timestamp"])[0:2]
    conv2d_layer1 = Conv2D(params["conv2d_layers"]["conv2d_filters_1"],
                           params["conv2d_layers"]["conv2d_kernel_size_1"],
                           strides=params["conv2d_layers"]["conv2d_strides_1"],
                           kernel_regularizer=regularizers.l2(params["conv2d_layers"]["kernel_regularizer_1"]),
                           padding='valid', activation="relu", use_bias=True,
                           kernel_initializer='glorot_uniform',
                           input_shape=(params['input_dim_1'],
                                        params['input_dim_2'], params['input_dim_3']))
    model.add(conv2d_layer1)
    if params["conv2d_layers"]['conv2d_mp_1'] == 1:
        model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(params['conv2d_layers']['conv2d_do_1']))
    if params["conv2d_layers"]['layers'] == 'two':
        conv2d_layer2 = Conv2D(params["conv2d_layers"]["conv2d_filters_2"],
                               params["conv2d_layers"]["conv2d_kernel_size_2"],
                               strides=params["conv2d_layers"]["conv2d_strides_2"],
                               kernel_regularizer=regularizers.l2(params["conv2d_layers"]["kernel_regularizer_2"]),
                               padding='valid', activation="relu", use_bias=True,
                               kernel_initializer='glorot_uniform')
        model.add(conv2d_layer2)
        if params["conv2d_layers"]['conv2d_mp_2'] == 1:
            model.add(MaxPool2D(pool_size=2))
        model.add(Dropout(params['conv2d_layers']['conv2d_do_2']))

    model.add(Flatten())

    model.add(Dense(params['dense_layers']["dense_nodes_1"], activation='relu'))
    model.add(Dropout(params['dense_layers']['dense_do_1']))

    if params['dense_layers']["layers"] == 'two':
        model.add(Dense(params['dense_layers']["dense_nodes_2"], activation='relu',
                        kernel_regularizer=params['dense_layers']["kernel_regularizer_1"]))
        model.add(Dropout(params['dense_layers']['dense_do_2']))

    model.add(Dense(3, activation='softmax'))
    if params["optimizer"] == 'rmsprop':
        optimizer = optimizers.RMSprop(lr=params["lr"])
    elif params["optimizer"] == 'sgd':
        optimizer = optimizers.SGD(lr=params["lr"], decay=1e-6, momentum=0.9, nesterov=True)
    elif params["optimizer"] == 'adam':
        optimizer = optimizers.Adam(learning_rate=params["lr"], beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', f1_metric])
    # from keras.utils.vis_utils import plot_model use this too for diagram with plot
    model.summary(print_fn=lambda x: print(x + '\n'))
    return model


def check_baseline(pred, y_test):
    e = np.equal(pred, y_test)
    print("TP class counts", np.unique(y_test[e], return_counts=True))
    print("True class counts", np.unique(y_test, return_counts=True))
    print("Pred class counts", np.unique(pred, return_counts=True))
    holds = np.unique(y_test, return_counts=True)[1][2]  # number 'hold' predictions
    logger.append_log("baseline acc:", str((holds / len(y_test) * 100)))


model = create_model_cnn(params)
best_model_path = os.path.join(OUTPUT_PATH, 'best_model_keras')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                   patience=100, min_delta=0.0001)
# csv_logger = CSVLogger(os.path.join(OUTPUT_PATH, 'log_training_batch.log'), append=True)
rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.02, patience=10, verbose=1, mode='min',
                        min_delta=0.001, cooldown=1, min_lr=0.0001)
mcp = ModelCheckpoint(best_model_path, monitor='val_loss', verbose=0,
                      save_best_only=True, save_weights_only=False, mode='min', period=1)


def plot_history(history, count):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['f1_metric'])
    plt.plot(history.history['val_f1_metric'])
    plt.title('Model Metrics')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train_loss', 'val_loss', 'train_acc', 'val_acc', 'f1', 'val_f1'], loc='upper left')
    plt.savefig(os.path.join(OUTPUT_PATH, 'plt_{}'.format(count)))


count = 0
while True:
    logger.append_log("training for batch number {}".format(count))
    x_train, y_train, x_cv, y_cv, x_test, y_test, df_batch_train, df_batch_test, sample_weights, is_last_batch = \
        data_gen.get_rolling_data_next(None, 12)

    if os.path.exists(best_model_path) and count > 0:
        model = load_model(best_model_path)
        logger.append_log("loaded previous best model")

    history = model.fit(x_train, y_train, epochs=params['epochs'], verbose=0,
                        batch_size=64, shuffle=True,
                        validation_data=(x_cv, y_cv),
                        callbacks=[es, mcp, rlp]
                        , sample_weight=sample_weights)

    count = count + 1
    plot_history(history, count)
    min_arg = np.argmin(np.array(history.history['val_loss']))
    logger.append_log("Best val_loss is {} and corresponding train_loss is {}".format(
        history.history['val_loss'][min_arg], history.history['loss'][min_arg]))
    test_res = model.evaluate(x_test, y_test, verbose=0)
    logger.append_log("keras evaluate result =" + str(test_res)+", metrics:"+str(model.metrics_names))
    pred = model.predict(x_test)
    check_baseline(np.argmax(pred, axis=1), np.argmax(y_test, axis=1))
    conf_mat = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(pred, axis=1))
    logger.append_log('\n'+str(conf_mat))
    labels = [0, 1, 2]
    f1_weighted = f1_score(np.argmax(y_test, axis=1), np.argmax(pred, axis=1), labels=None,
                           average='weighted', sample_weight=None)
    logger.append_log("F1 score (weighted) " + str(f1_weighted))
    logger.append_log(
        "F1 score (macro) " + str(f1_score(np.argmax(y_test, axis=1), np.argmax(pred, axis=1), labels=None,
                                           average='macro', sample_weight=None)))
    logger.append_log(
        "F1 score (micro) " + str(f1_score(np.argmax(y_test, axis=1), np.argmax(pred, axis=1), labels=None,
                                           average='micro',
                                           sample_weight=None)))  # weighted and micro preferred in case of imbalance

    for i, row in enumerate(conf_mat):
        logger.append_log("precision of class {} = {}".format(i, np.round(row[i] / np.sum(row), 2)))

    if is_last_batch:
        break

logger.append_log("Complete training finished in {}".format(seconds_to_minutes(time.time() - start_time)))
logger.flush()

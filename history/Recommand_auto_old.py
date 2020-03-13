#!/usr/bin/env python
# coding: utf-8

import time
import random
import pandas as pd
import numpy as np
from sklearn import preprocessing
import warnings
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.models

warnings.filterwarnings('ignore')


def time_string_now():
    _time_str = time.strftime("%Y_%m_%d_%u_%H%M%S", time.localtime())
    return _time_str


# normalize function
def normalize_datas(datas, init, end):
    min_max_scale = preprocessing.MinMaxScaler(feature_range=(init, end))
    scaled_datas = min_max_scale.fit_transform(datas)
    return scaled_datas


# data to train_data and test data
def generate_data(data, label):
    msk = np.random.rand(len(data)) < 0.8
    print(msk)
    train_data = data[msk]
    train_label = label[msk]
    test_data = data[~msk]
    test_label = label[~msk]
    return train_data, train_label, test_data, test_label


def labeling(normalize_data, filter_data):
    # 24 0~1 input
    # Labeling
    # 推薦情境
    # 使用情況優先度:網路量大 通話量大  簡訊量大   大: value >median
    # {'None':0, 'Apple':1, 'Samsung':2, 'HTC':3, 'ASUS':4,
    #                                                  'Sony':5, '小米':6, 'OPPO':7, 'Huawei':8, 'Panasonic':9, 'gift':10}
    # 終端設備: Apple(Apple,None) HTC(HTC,ASUS,小米,OPPO,Huawei,gift)  Samsung(Samsung,Sony,Panasonic)
    # 設定九種資費方案
    # label 0: 網路量大 Apple
    # label 1: 網路量大 HTC
    # label 2: 網路量大 Samsung
    # label 3: 通話量大 Apple
    # label 4: 通話量大 HTC
    # label 5: 通話量大 Samsung
    # label 6: 簡訊量大 Apple
    # label 7: 簡訊量大 HTC
    # label 8: 簡訊量大 Samsung
    # 若資費不符合上述條件 推薦 網路大方案
    # 若手機不符合上述條件 推薦 Apple
    label_counter = 0
    _labeling = []

    # get median
    data_vol_median = np.median(normalize_data[:, 7])
    dur_vol_median = np.median(normalize_data[:, 5])
    msg_vol_median = np.median(normalize_data[:, 6])
    print('data_vol_median:' + str(data_vol_median))
    print('dur_vol_median:' + str(dur_vol_median))
    print('msg_vol_median:' + str(msg_vol_median))
    for data in normalize_data:
        if data[7] >= data_vol_median:  # data[7] 網路量
            if filter_data['GIFT01'][label_counter] == 'Apple' or filter_data['GIFT01'][label_counter] == 'None':
                _labeling.append(0)
            elif filter_data['GIFT01'][label_counter] == 'HTC' or filter_data['GIFT01'][label_counter] == 'ASUS' or \
                    filter_data['GIFT01'][label_counter] == '小米' or filter_data['GIFT01'][label_counter] == 'OPPO' or \
                    filter_data['GIFT01'][label_counter] == 'Huawei' or filter_data['GIFT01'][label_counter] == 'gift':
                _labeling.append(1)
            elif filter_data['GIFT01'][label_counter] == 'Samsung' or filter_data['GIFT01'][label_counter] == 'Sony' \
                    or filter_data['GIFT01'][label_counter] == 'Panasonic':
                _labeling.append(2)
        elif data[5] >= dur_vol_median:  # data[5] 通話量
            if filter_data['GIFT01'][label_counter] == 'Apple' or filter_data['GIFT01'][label_counter] == 'None':
                _labeling.append(3)
            elif filter_data['GIFT01'][label_counter] == 'HTC' or filter_data['GIFT01'][label_counter] == 'ASUS' or \
                    filter_data['GIFT01'][label_counter] == '小米' or filter_data['GIFT01'][label_counter] == 'OPPO' or \
                    filter_data['GIFT01'][label_counter] == 'Huawei' or filter_data['GIFT01'][label_counter] == 'gift':
                _labeling.append(4)
            elif filter_data['GIFT01'][label_counter] == 'Samsung' or filter_data['GIFT01'][label_counter] == 'Sony' \
                    or filter_data['GIFT01'][label_counter] == 'Panasonic':
                _labeling.append(5)

        elif data[6] >= msg_vol_median:  # data[6] 簡訊量
            if filter_data['GIFT01'][label_counter] == 'Apple' or filter_data['GIFT01'][label_counter] == 'None':
                _labeling.append(6)
            elif filter_data['GIFT01'][label_counter] == 'HTC' or filter_data['GIFT01'][label_counter] == 'ASUS' or \
                    filter_data['GIFT01'][label_counter] == '小米' or filter_data['GIFT01'][label_counter] == 'OPPO' or \
                    filter_data['GIFT01'][label_counter] == 'Huawei' or filter_data['GIFT01'][label_counter] == 'gift':
                _labeling.append(7)
            elif filter_data['GIFT01'][label_counter] == 'Samsung' or filter_data['GIFT01'][label_counter] == 'Sony' \
                    or filter_data['GIFT01'][label_counter] == 'Panasonic':
                _labeling.append(8)

        else:
            _labeling.append(0)

        label_counter += 1

    print(type(normalize_data))
    print("len(labeling)")
    print(len(_labeling))
    print('\nlabel_counter')
    print('0:' + str(_labeling.count(0)))
    print('1:' + str(_labeling.count(1)))
    print('2:' + str(_labeling.count(2)))
    print('3:' + str(_labeling.count(3)))
    print('4:' + str(_labeling.count(4)))
    print('5:' + str(_labeling.count(5)))
    print('6:' + str(_labeling.count(6)))
    print('7:' + str(_labeling.count(7)))
    print('8:' + str(_labeling.count(8)))

    label = np.asarray(_labeling)
    return label


def label_process(label):
    labels = np_utils.to_categorical(label)
    return labels


def data_preprocess_for_database(new_data):
    cols = ['GENDER', 'VIP', 'BATCHNO', 'PRICE', 'MONTHLY_FEE_CLASS', 'GIFT01', 'AGE_CLASS', 'DUR_VOL', 'MSG_VOL',
            'DATA_VOL']
    filter_data = new_data[cols]
    ##########
    filter_data['BATCHNO'] = filter_data['BATCHNO'].fillna('not偏鄉')
    filter_data['BATCHNO'] = filter_data['BATCHNO'].map({'NCC偏鄉': 1, 'not偏鄉': 0}).astype(int)
    ##########
    filter_data['GENDER'] = filter_data['GENDER'].map({'女': 0, '男': 1}).astype(int)
    ###########
    # fill mean value to the null data

    ##
    dur_vol_mean = filter_data['DUR_VOL'].mean()
    msg_vol_mean = filter_data['MSG_VOL'].mean()
    data_vol_mean = filter_data['DATA_VOL'].mean()
    ##

    filter_data['DUR_VOL'] = filter_data['DUR_VOL'].fillna(dur_vol_mean).astype(int)
    filter_data['MSG_VOL'] = filter_data['MSG_VOL'].fillna(msg_vol_mean).astype(int)
    filter_data['DATA_VOL'] = filter_data['DATA_VOL'].fillna(data_vol_mean).astype(int)
    ################
    price_column = filter_data['PRICE']
    price_name = pd.value_counts(filter_data['PRICE'])
    price_name_index = price_name.index
    price_name_index_sorted = sorted(price_name_index)
    ##
    price_name_class_mapping = {label: index for index, label in enumerate(price_name_index_sorted)}
    ##
    price_column = price_column.map(price_name_class_mapping).astype(int)
    filter_data['PRICE'] = price_column
    ##################
    filter_data['MONTHLY_FEE_CLASS'] = filter_data['MONTHLY_FEE_CLASS'].map(
        {'(1)低低資費0~199': 1, '(2)低資費200~399': 2, '(3)中資費400~699': 3, '(4)中高資費700~899': 4, '(5)高資費900~': 5}
    ).astype(int)
    #########
    # GIFT01 preprocess
    # 1. '無第一終端' transfer to 'None'
    # 2. cell phone transfer to  Company Brand
    # 3. other gift transfer to 'gift'
    # 4. encoding
    # encoding: 'None':0, 'Apple':1, 'Samsung':2, 'HTC':3, 'ASUS':4,
    # 'Sony':5, '小米':6, 'OPPO':7, 'Huawei':8, 'Panasonic':9, 'gift':10
    filter_data['GIFT01'][:] = filter_data['GIFT01'][:].replace('無第一終端', 'None')
    counter = 0
    for item in filter_data['GIFT01']:
        #     index=item.find('Apple')
        if item.find('Apple') != -1:
            filter_data['GIFT01'][counter] = 'Apple'
        elif item.find('Sony') != -1:
            filter_data['GIFT01'][counter] = 'Sony'
        elif item.find('OPPO') != -1:
            filter_data['GIFT01'][counter] = 'OPPO'
        elif item.find('Samsung') != -1:
            filter_data['GIFT01'][counter] = 'Samsung'
        elif item.find('HTC') != -1:
            filter_data['GIFT01'][counter] = 'HTC'
        elif item.find('小米') != -1:
            filter_data['GIFT01'][counter] = '小米'
        elif item.find('好禮自由選') != -1:
            filter_data['GIFT01'][counter] = 'gift'
        elif item.find('ASUS') != -1 or item.find('Asus') != -1:
            filter_data['GIFT01'][counter] = 'ASUS'
        elif item.find('Huawei') != -1:
            filter_data['GIFT01'][counter] = 'Huawei'
        elif item.find('Panasonic') != -1:
            filter_data['GIFT01'][counter] = 'Panasonic'
        elif item.find('None') != -1:
            filter_data['GIFT01'][counter] = 'None'
        else:
            filter_data['GIFT01'][counter] = 'gift'
        counter += 1
    #######################################
    # Age processing
    #
    age_column = filter_data['AGE_CLASS']
    print("\nage_column")
    print(age_column)
    age_name = pd.value_counts(filter_data['AGE_CLASS'])
    print("\nage_name distribute")
    print(age_name)
    age_name_index = age_name.index
    print("\nage_name_index")
    print(age_name_index)
    # preprocessor
    counter = 0
    for item in age_column:
        if item == '100+':
            age_column[counter] = '991~100+'
        if item == '7～14':
            age_column[counter] = '07～14'
        if item == '0～6':
            age_column[counter] = '00～06'
        elif item.find('(未成年)') != -1:
            age_column[counter] = age_column[counter].replace('(未成年)', '')
        elif item.find('(限制行為能力)') != -1:
            age_column[counter] = age_column[counter].replace('(限制行為能力)', '')
        elif item.find('(無行為能力)') != -1:
            age_column[counter] = age_column[counter].replace('(無行為能力)', '')
        counter += 1

    age_name = pd.value_counts(age_column)
    age_name_index_sorted = sorted(age_name.index)
    age_name_index_sorted[-1] = '100+'
    print("\nage_name_index_sorted")
    print(age_name_index_sorted)

    counter = 0
    for item in age_column:
        if item == '991~100+':
            age_column[counter] = '100+'
        counter += 1
    print(age_column)
    ##
    age_name_class_mapping = {label: index for index, label in enumerate(age_name_index_sorted)}
    ##
    print('\nage_name_class_mapping')
    print(age_name_class_mapping)
    age_column = age_column.map(age_name_class_mapping).astype(int)
    print(age_column)

    filter_data['AGE_CLASS'] = age_column
    #########################################################
    # OneHot encoding
    # 1、離散特徵的大小沒有意義時，比如color：[red,blue],使用one-hot encoding
    # 2、離散特徵的大小具有意義時，比如size:[X,XL,XXL],使用數值映射{X:1,XL:2,XXL:3}说明：对于有大小意义的离散特征，直接使用映射就可以了，{'XL':3,'L':2,'M':1}
    one_hot_df = pd.get_dummies(data=filter_data, columns=["GIFT01"])
    print(one_hot_df[:2])
    print("########################################")
    one_hot_df = pd.get_dummies(data=one_hot_df, columns=["VIP"])
    print(one_hot_df[:2])
    print(one_hot_df)
    #########################
    # normalize data from 0~1
    features = one_hot_df.values
    normalize_data = normalize_datas(features, 0, 1)

    return filter_data, normalize_data, dur_vol_mean, msg_vol_mean, data_vol_mean, price_name_class_mapping, \
           age_name_class_mapping


def plot_confusion_matrix(model, normalize_test_data, test_label):
    all_probability_class = model.predict_classes(normalize_test_data)
    print(test_label.shape)
    print(pd.crosstab(test_label, all_probability_class, rownames=['label'], colnames=['predict']))


def generate_data_label(normalize_data, filter_data):
    label = labeling(normalize_data, filter_data)
    train_data, train_label, test_data, test_label = generate_data(normalize_data, label)
    train_labels = label_process(train_label)
    test_labels = label_process(test_label)
    return train_data, train_label, test_data, test_label, train_labels, test_labels


def output_result(result):
    result_mapping = ["0-LargeData Apple", "1-LargeData HTC", "2-LargeData Samsung", "3-LargeVol Apple",
                      "4-LargeVol HTC", "5-LargeVol Samsung", "6-LargeMsg Apple", "7-LargeMsg HTC",
                      "8-LargeMsg Samsung"]
    return result_mapping[int(result)]


def final_output(data):
    _output_string = "predict:" + output_result(np.where(data == np.amax(data))[0]) + ":" + str(np.amax(data))
    print(_output_string)
    return _output_string


def prediction_output(model, normalizedata):
    all_probability = model.predict(normalizedata)

    # final_output(all_probability)
    return all_probability


#     print("label:" + output_result(test_label[test_number]))
def prediction_one_output(model, normalizedata):
    normalized_datas = np.array([normalizedata, normalizedata])
    all_probability = model.predict(normalized_datas)
    print(all_probability[0])
    _string = final_output(all_probability[0])
    return _string


#     print("label:" + output_result(test_label[test_number]))


# Train
def train_process(train_data, train_labels, test_data, test_labels):
    model = Sequential()
    model.add(Dense(units=40, input_dim=24, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=30, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=9, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    train_history = model.fit(x=train_data, y=train_labels, validation_split=0.1, epochs=30, batch_size=30, verbose=2)
    print("training done")
    # save model
    time_str = time_string_now()
    save_weight_string = "./recommand_model/" + time_str + "_weight.h5"
    save_model_string = "./recommand_model/" + time_str + "_model.h5"
    model.save_weights(save_weight_string)
    print("save weight:" + save_weight_string)
    model.save(save_model_string)
    print("save model:" + save_model_string)
    # Evaluate
    print("Evaluate ")
    scores = model.evaluate(x=test_data, y=test_labels)
    print("\nscore:" + str(scores[1]))
    return model


def evaluate_model(model, normalize_data, number):
    all_probability = model.predict(normalize_data)
    print("\nall_probability:" + str(all_probability[number]))
    print("\nMax probability:" + str(all_probability[number].max()))


# save data
def save_data(_normalize_data, _filter_data):
    _time_str = time_string_now()
    _save_np_string = "./recommand_model/" + _time_str + "_normalize_data.npy"
    _save_pd_string = "./recommand_model/" + _time_str + "_filter_data.pkl"
    np.save(_save_np_string, _normalize_data)
    _filter_data.to_pickle(_save_pd_string)


# load data
def load_data(normalize_data_path_name="./recommand_model/2019_10_01_normalize_data.npy",
              filter_data_path_name="./recommand_model/2019_10_01_filter_data.pkl", ):
    _normalize_data = None
    _normalize_data = np.load(normalize_data_path_name)

    _filter_data = None
    _filter_data = pd.read_pickle(filter_data_path_name)
    return _normalize_data, _filter_data

def load_data(normalize_data_path_name="./recommand_model/2019_10_01_normalize_data.npy"):
    _normalize_data = None
    _normalize_data = np.load(normalize_data_path_name)
    return _normalize_data



def import_csv_to_dataframe(_filepath='./aptg_data/ai_data_set_utf8_201908.csv'):
    _aptg_data = pd.read_csv(_filepath, encoding='utf-8')
    print(_aptg_data)
    _aptg_data.head()
    return _aptg_data


# train_data
def train_program(_dataframe):
    filter_data, normalize_data, dur_vol_mean, msg_vol_mean, data_vol_mean, price_name_class_mapping, \
    age_name_class_mapping = data_preprocess_for_database(
        _dataframe)
    # saving data
    time_str = time_string_now()
    save_np_string = "./recommand_model/" + time_str + "_normalize_data.npy"
    save_pd_string = "./recommand_model/" + time_str + "_filter_data.pkl"
    np.save(save_np_string, normalize_data)
    filter_data.to_pickle(save_pd_string)
    print("\nsaving np:" + save_np_string)
    print("\nsaving pd:" + save_pd_string)

    # generate_data_label
    train_data, train_label, test_data, test_label, train_labels, test_labels = generate_data_label(normalize_data,
                                                                                                    filter_data)

    # train
    model = train_process(train_data, train_labels, test_data, test_labels)
    number = 1
    evaluate_model(model, normalize_data, number)
    id_string = random.randint(1, normalize_data.shape[0])
    print("Random id:" + str(id_string))
    prediction_one_output(model, normalize_data[id_string])


def test_function():
    print("test")


def recommand(_model, _normalize_data, _id_string):
    _string = prediction_one_output(_model, _normalize_data[_id_string])
    return _string


def recommand_all(_model, _normalize_data):
    all_probability = prediction_output(_model, _normalize_data)
    return all_probability


def recommand_init(normalize_data_path_name="./recommand_model/2019_10_01_normalize_data.npy",
                   model_data_path_name="./recommand_model/model_2019_09_27.h5"
                   ):
    # load data
    _normalize_data = load_data(normalize_data_path_name)
    # load model
    _model = keras.models.load_model(model_data_path_name)
    return _normalize_data, _model


def main_train():
    aptg_df = import_csv_to_dataframe()
    train_program(aptg_df)


def main():
    normalize_data, model = recommand_init()
    id_string = random.randint(1, normalize_data.shape[0])
    print("Random id:" + str(id_string))
    recommand(model, normalize_data, id_string)
    print("Precalculate")
    all_probability = recommand_all(model, normalize_data)
    id_string = random.randint(1, normalize_data.shape[0])
    print("Random id:" + str(id_string))
    _string = final_output(all_probability[id_string])
    return _string


if __name__ == '__main__':
    main_train()
    # main()

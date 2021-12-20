import jieba
import numpy as np
import pandas as pd
import jieba.analyse
import tensorflow.keras as keras
from keras.callbacks import History
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import LSTM
from keras.layers import Dense
from matplotlib import pyplot as plt


def padding(word_sequences_list: list, max_length: int) -> np.ndarray:
    res = []
    for text in word_sequences_list:
        if len(text) > max_length:
            text = text[:max_length]
        else:
            text = text + [0 for i in range(max_length - len(text))]
        res.append(text)

    return np.array(res)


if __name__ == '__main__':
    # 读取平衡处理后的外卖评价数据集
    data_all_df = pd.read_csv('./Data/data_balance.csv', index_col=None)

    sentence_list = []
    label_list = []
    # 将数据转化为字典形式
    for sentence, label in zip(data_all_df['review'], data_all_df['label']):
        sentence_list.append(sentence)
        label_list.append(label)

    # 分词
    sentence_list = [".".join(jieba.cut(sentence, cut_all=False))
                     for sentence in sentence_list]

    # 使用词汇表序列化
    json_string = open('../Result/word.json', 'r', encoding='utf-8').read()
    word_processor = keras.preprocessing.text.tokenizer_from_json(json_string)
    word_sequences_list = word_processor.texts_to_sequences(sentence_list)

    # 截断或补齐
    word_sequences_processed_list = padding(word_sequences_list, 40)

    # 验证集比例、数目
    val_split = 0.2
    val_counts = int(val_split * len(label_list))

    # 切分验证集
    val_x = word_sequences_processed_list[-val_counts:]
    val_y = np.array(label_list[-val_counts:])
    train_x = word_sequences_processed_list[:-val_counts]
    train_y = np.array(label_list[:-val_counts])

    # 选择模型
    model = keras.Sequential()

    # 构建网络
    model = Sequential()

    model.add(Embedding(20000, 32))
    model.add(Conv1D(filters=32,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    # 训练模型
    history: History = model.fit(train_x,
                                 train_y,
                                 batch_size=64,
                                 epochs=5,
                                 validation_data=(val_x, val_y))

    # 保存模型
    model.save('../Result/model')

    # 显示训练历史
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(loc='upper right')
    plt.show()

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.show()

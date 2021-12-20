import jieba
import pandas as pd
import tensorflow.keras as keras


if __name__ == '__main__':
    # 读取平衡处理后的外卖评价数据集
    data_all_df = pd.read_csv('./Data/data_balance.csv', index_col=None)

    sentence_list = []
    label_list = []
    # 将数据转化为列表
    for sentence, label in zip(data_all_df['review'], data_all_df['label']):
        sentence_list.append(sentence)
        label_list.append(label)

    # 分词
    sentence_list = [".".join(jieba.cut(sentence, cut_all=False))
                     for sentence in sentence_list]

    # 构建词汇表
    word_processor = keras.preprocessing.text.Tokenizer(num_words=20000,
                                                        filters='。，：；“”《》（）!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ',
                                                        oov_token='<UNK>')
    word_processor.fit_on_texts(sentence_list)

    # 保存词汇表
    with open('../Result/word.json', 'w') as f:
        f.write(word_processor.to_json())
        f.close()

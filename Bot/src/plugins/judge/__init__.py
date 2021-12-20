import jieba
import numpy as np
import tensorflow.keras as keras
from keras.models import load_model
from keras.models import Sequential
from nonebot import on_command
from nonebot import logger
from nonebot.adapters.cqhttp import Bot, MessageEvent


judge = on_command('评价')


word_processor = keras.preprocessing.text.tokenizer_from_json(
    open('../Result/word.json', 'r', encoding='utf-8').read())
model: Sequential = load_model('../Result/model')
model.predict([[0 for i in range(40)]])


def padding(word_sequences_list: list, max_length: int) -> np.ndarray:
    res = []
    for text in word_sequences_list:
        if len(text) > max_length:
            text = text[:max_length]
        else:
            text = text + [0 for i in range(max_length - len(text))]
        res.append(text)

    return np.array(res)


@judge.handle()
async def handle_judge(bot: Bot, event: MessageEvent):
    review = str(event.message).strip()

    # 分词
    sentence_list = [".".join(jieba.cut(review, cut_all=False))]

    # 序列化
    word_sequences_list = word_processor.texts_to_sequences(sentence_list)

    # 截断或补齐长度
    word_sequences_processed_list = padding(word_sequences_list, 40)

    # 预测
    res = model.predict(word_sequences_processed_list)

    if res[0][0] > 0.5:
        msg = '好评'
    else:
        msg = '差评'

    logger.debug(f"预测输出: {res[0][0]}")

    await judge.send(msg + f" | 预测输出: {res[0][0]} | 分词结果: " + sentence_list[0])

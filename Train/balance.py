import pandas as pd


if __name__ == '__main__':
    # 读取外卖评价数据集
    data_all_df = pd.read_csv('./Data/waimai_10k.csv', index_col=None)

    # 筛选出正面评价和负面评价
    data_positive_df = data_all_df[data_all_df['label'] == 1]
    data_negative_df = data_all_df[data_all_df['label'] == 0]

    # 求出正面评价和负面评价数量中的最小值，用于构造平衡数据集
    min_size = min(data_positive_df.shape[0], data_negative_df.shape[0])

    # 通过抽样方法构造平衡数据集
    data_balance_df = pd.concat([data_positive_df.sample(min_size, replace=data_positive_df.shape[0] < min_size),
                                 data_negative_df.sample(min_size, replace=data_negative_df.shape[0] < min_size)])

    # 通过抽样方法打乱数据集
    data_balance_df = data_balance_df.sample(data_balance_df.shape[0])

    # 显示处理得到的平衡数据集
    print(f"评论数目: {data_balance_df.shape[0]}")
    print(f"正面评价: {data_balance_df[data_balance_df['label'] == 1].shape[0]}")
    print(f"负面评价: {data_balance_df[data_balance_df['label'] == 0].shape[0]}")

    print(data_balance_df.sample(10))

    # 保存
    data_balance_df.to_csv('./Data/data_balance.csv', index=False)

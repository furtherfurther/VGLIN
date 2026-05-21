# import pickle
#
# # 打开并加载.pkl文件
# with open('./data/iemocap_multimodal_features.pkl', 'rb') as file:
#     data = pickle.load(file)
#
# # 现在data中包含.pkl文件中的数据
# print(data)

# 手动解析提取的混淆矩阵（基于文本结构，以下是模拟的解析逻辑）
# 例如，提取为 NumPy 数组：
import numpy as np

# 手动解析提取的混淆矩阵（基于文本结构，以下是模拟的解析逻辑）
# confusion_matrix = np.array([
#     [101, 1, 7, 6, 27, 2],
#     [0, 203, 11, 3, 0, 28],
#     [21, 13, 283, 12, 10, 45],
#     [0, 0, 4, 125, 0, 41],
#     [37, 0, 23, 2, 235, 2],
#     [3, 10, 47, 47, 5, 269]
# ])

confusion_matrix = np.array(
    # [
    #     [1042, 47, 9, 27, 73, 10, 48],
    #     [35, 169, 4, 5, 33, 2, 33],
    #     [17, 6, 11, 5, 3, 2, 6],
    #     [74, 15, 17, 30, 24, 20, 28],
    #     [84, 26, 3, 5, 262, 4, 18],
    #     [23, 5, 0, 4, 1, 13, 22],
    #      [72, 40, 5, 7, 34, 13, 174]
    # ]
    # [
    #     [1045, 29, 5, 33, 74, 5, 65],
    #     [40, 169, 1, 3, 32, 3, 33],
    #     [17, 4, 10, 7, 3, 1, 8],
    #     [75, 18, 16, 35, 17, 18, 29],
    #     [86, 23, 1, 6, 260, 3, 23],
    #     [22, 5, 1, 3, 0, 14, 23],
    #     [68, 35, 2, 14, 28, 6, 192]
    # ]
    # [[1045, 33, 2, 40, 59, 13, 64],
    #  [43, 164, 0, 3, 27, 3, 41],
    #  [10, 4, 5, 11, 4, 5, 13],
    #  [65, 14, 0, 74, 22, 3, 30],
    #  [76, 32, 0, 6, 257, 2, 29],
    #  [22, 4, 0, 4, 0, 19, 25],
    #  [67, 31, 1, 10, 21, 9, 206]
    #  ]
   [[1029, 38, 8, 28, 68, 11, 74],
    [38, 170, 2, 3, 30, 2, 36],
    [18, 4, 9, 7, 2, 1, 9],
    [70, 13, 7, 66, 10, 4, 38],
    [90, 18, 1, 1, 275, 2, 15],
    [20, 4, 1, 2, 1, 16, 24],
    [54, 36, 2, 10, 22, 11, 210]
    ]
)
#
# confusion_matrix = np.array([
#    [102, 1, 7, 5, 27, 2],
#     [0, 210, 9, 3, 0, 23],
#     [21, 13, 283, 12, 10, 45],
#     [0, 0, 4, 129, 0, 37],
#     [37, 0, 21, 2, 237, 2],
#     [3, 10, 47, 47, 5, 269]
# ])

confusion_matrix = np.array([
   [103, 1, 7, 5, 26, 2],
    [0, 212, 8, 3, 0, 22],
    [22, 13, 281, 12, 11, 45],
    [1, 0, 4, 128, 0, 37],
    [36, 0, 19, 2, 240, 2],
    [3, 10, 47, 46, 5, 270]
])

confusion_matrix = np.array([
   [1060, 31, 2, 40, 54, 8, 61],
    [38, 174, 0, 3, 25, 3, 38],
    [12, 4, 15, 11, 0, 1, 7],
    [65, 14, 0, 74, 22, 3, 30],
    [67, 31, 0, 4, 274, 2, 24],
    [22, 4, 0, 4, 0, 19, 25],
    [64, 29, 1, 10, 19, 9, 213]
])

# 计算 precision, recall, f1-score, support 和 weighted avg f1-score
def calculate_metrics(cm):
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    support = np.sum(cm, axis=1)
    accuracy = np.trace(cm) / np.sum(cm)

    # 去除可能的 NaN（0除0 的情况）
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1_score = np.nan_to_num(f1_score)

    # Weighted Avg F1-Score
    weighted_avg_f1 = np.sum(f1_score * support) / np.sum(support)
    return precision, recall, f1_score, support, accuracy, weighted_avg_f1


precision, recall, f1_score, support, accuracy, weighted_avg_f1 = calculate_metrics(confusion_matrix)

# 输出结果
# print("\n各类别指标（保留四位小数）：")
print(f"\nAcc: {round(accuracy, 4):.4f}")
print(f"F1-Score: {round(weighted_avg_f1, 4):.4f}\n")  # Weighted Avg F1-Score

print("   Precision    Recall    F1-Score    Support")
for i, s in enumerate(support):
    print(f"{i}  {round(precision[i], 4):.4f}       {round(recall[i], 4):.4f}    {round(f1_score[i], 4):.4f}      {s}")
print("\n")
print(confusion_matrix)





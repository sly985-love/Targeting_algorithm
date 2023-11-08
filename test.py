# a = [[0, '偏右下方', 8.58], [1, '偏左下方', 9.03], [2, '偏右上方', 9.73], [3, '偏右上方', 9.37], [4, '偏右下方', 6.61],
#                   [5, '偏右下方', 10.01]]
# for i in range(0, len(a)):
#     fx = a[i][1]
#     cj = a[i][2]
#     print(a[i][1], a[i][2])

result = [[0, '偏右下方', 8.58], [1, '偏左下方', 9.03], [2, '偏右上方', 9.73], [3, '偏右上方', 9.37], [4, '偏右下方', 6.61],
          [5, '偏右下方', 10.01]]

print("Predicted result: ", result)  # 在django中将识别结果打印出来
# result[[num,score,direction],....]
print(len(result))
for i in range(0, len(result)):
    score = result[i][1]
    direction = result[i][2]
    print(result[i][1], result[i][2])
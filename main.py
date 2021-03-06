from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
from numpy.dual import lstsq

df = pd.read_csv('./data/train.csv', na_values='NR', encoding='big5')

# Choose factors
factors = df[u'測項'].unique()
factors = np.delete(factors, 10) # delete('RAINFALL')
# factors = [u'PM2.5']
# factors = [u'PM10', u'PM2.5', u'NO2']

def prepare_data(df):
    days = df[u'日期'].unique()
    test = days.shape[0]
    days = days[:test]
    frames = []
    for day in days:
        dfByDay = df[df[u'日期'].isin([day]) & df[u'測項'].isin(factors)].iloc[:, 3:]
        dfByDay.index = factors
        frames.append(dfByDay)

    # 將橫軸各個指標的的 24h 照日期順序串起來 (1/1 AMB_TEMP + 1/ AMB_TEMP + ....)
    # 最後會是 (指標數量, 24*天) 維的陣列
    training = pd.concat(frames, axis=1)

    # 線性方程式的 x 應該在 col 的位子，所以將陣列轉置
    training = training.T
    # 取九小時資料排成一排
    m = training.shape[0]
    d = []
    # sliding window count: 24*天 - 8(剩八必須得停)
    for i in range(0, m - 8):
        # row i ~ row i + 8，產生 [9小時, 指標數] 的陣列
        x = training.iloc[i:i + 9]
        # 將 [9小時, 指標數] 攤成 1 row
        # => x11 +...+ x19 + x21 + ... + x29 + ... + xn1 + ... + xn9
        x = x.T.values.flatten()
        # 所有 x 加入 d 後，d 便為 X * W + b 的 X matrix
        d.append(x)

    # y 為 第 10 個小時到 m - 9 小時的 array
    output = training[8:]['PM2.5'].values.flatten()
    return d, output

# Variable for regression
training_epochs = 100000
display_step = 10
learning_rate = 0.0000001
with tf.name_scope('InputData'):
    X = tf.placeholder("float", [1, factors.shape[0] * 9], name="X")
with tf.name_scope('LabelData'):
    Y = tf.placeholder("float", [1, 1], name="Y")
# Set model weights
W = tf.Variable(tf.zeros([factors.shape[0] * 9, 1]), name="weight")

# OP for regression
b = tf.Variable(tf.zeros([1]), name="bias")
with tf.name_scope('Model'):
    predictedY = tf.add(tf.matmul(X, W), b)
with tf.name_scope('Loss'):
    loss = tf.reduce_sum(tf.square((Y - predictedY)))
with tf.name_scope('Gradient'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

lossS = tf.abs(Y - predictedY)

# Initializing the variables
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)

# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    # $ tensorboard --logdir="C:\cms\tensorflow_linear_regression\summary" --port 6006
    train_writer = tf.summary.FileWriter('summary', sess.graph)

    sess.run(init)
    trainX = []
    trainY = []
    arrayX, arrayY = (prepare_data(df))

    for (df_x, sr_y) in zip(np.asarray(arrayX), np.asarray(arrayY)):
        trainX.append(np.asmatrix(df_x))
        trainY.append(np.asmatrix(sr_y))

    # 為預先準備理論最佳值 W & b
    tmpArrayX = pd.DataFrame(arrayX);
    tmpArrayX['b'] = 1
    arrayXWithB = tmpArrayX.as_matrix();
    # print("arrayX:", tmpArrayX.as_matrix(), "\narrayY:", np.asmatrix(arrayX))
    # print("trainX:", trainX, "\ntrainY:", trainY)

    WR = lstsq(arrayXWithB, np.asmatrix(arrayY).T)[0]
    print("WR:", WR, " size:", len(WR))
    print("lstsq:", np.sum(np.abs(np.subtract(np.asmatrix(arrayY).T, np.matmul(tmpArrayX, WR)))))

    # 開始 train loop
    for epoch in range(training_epochs):
        for (x, y) in zip(trainX, trainY):
            _, summary = sess.run([optimizer, merged_summary_op], feed_dict={X: x, Y: y})
            train_writer.add_summary(summary, epoch)

        if epoch % display_step == 0:
            result = 0
            for (x, y) in zip(trainX, trainY):
                tmp = sess.run(lossS, feed_dict={X: x, Y: y})
                result += tmp
            # 每 train 1000 次顯示 w & b result
            if epoch % (display_step * 100) == 0:
                print("Loss=", result, "\nW=", sess.run(W), "\nb=", sess.run(b), '\n')
            WP = np.vstack([sess.run(W), sess.run(b)])
            WE = np.sum(np.abs(np.subtract(WP, WR)))
            # 顯示 train 出來的 [w,b] 和 理論值的 vector 誤差 WE 以及 training error
            print("epoch=", epoch, " WE=", WE, " Loss=", result)
        epoch = epoch + 1

    result = 0
    for (X, Y) in zip(trainX, trainY):
        result += sess.run(lossS, feed_dict={x: X, y: Y})
    print("Final Loss=", result, "W=", sess.run(W), "b=", sess.run(b), '\n')

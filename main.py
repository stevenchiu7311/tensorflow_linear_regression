from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
from numpy.dual import lstsq


def train(y):
    df = pd.read_csv('./data/train.csv', na_values='NR', encoding='big5')

    days = df[u'日期'].unique()
    test = days.shape[0]
    print(test)
    days = days[:test]
    # delete('RAINFALL')
    factors = df[u'測項'].unique()
    factors = np.delete(factors, 10)

    # factors = [u'PM2.5']
    # factors = [u'PM10', u'PM2.5', u'NO2']
    # print(factors)
    # print(days)

    frames = []
    for day in days:
        dfByDay = df[df[u'日期'].isin([day]) & df[u'測項'].isin(factors)].iloc[:, 3:]
        dfByDay.index = factors
        frames.append(dfByDay)
    training = pd.concat(frames, axis=1)

    training = training.T
    print(training, '\n\n')

    # 取九小時資料排成一排
    m = training.shape[0]
    d = []
    for i in range(0, m - 8):
        x = training.iloc[i:i + 9]
        x = x.T.values.flatten()
        d.append(x)
    # x = pd.DataFrame(d).T
    # x['b'] = 1
    # print('xt1', x)
    y.append(training[8:]['PM2.5'].values.flatten())
    # print("y", y);
    # print('#x', x.shape[0])
    return d

training_epochs = 100000
display_step = 10
learning_rate = 0.0000001
X = tf.placeholder("float", [1, 153])
Y = tf.placeholder("float", [1, 1])
# Set model weights
W = tf.Variable(tf.zeros([153, 1]), name="weight")

b = tf.Variable(tf.zeros([1]), name="bias")
predictedY = tf.add(tf.matmul(X, W), b)
loss = tf.reduce_sum(tf.square((Y - predictedY)))
lossS = tf.abs(Y - predictedY)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    trainX = []
    trainY = []
    arrayX = []
    arrayY = []
    arrayX = (train(arrayY))

    for (df_x, sr_y) in zip(np.asarray(arrayX), np.asarray(arrayY[0])):
        trainX.append(np.asmatrix(df_x))
        trainY.append(np.asmatrix(sr_y))

    tmpArrayX = pd.DataFrame(arrayX);
    tmpArrayX['b'] = 1
    arrayXWithB = tmpArrayX.as_matrix();
    # print("arrayX:", tmpArrayX.as_matrix(), "\narrayY:", np.asmatrix(arrayX))
    # print("trainX:", trainX, "\ntrainY:", trainY)

    WR = lstsq(arrayXWithB, np.asmatrix(arrayY).T)[0]
    print("WR:", WR, " size:", len(WR))
    print("lstsq:", np.sum(np.abs(np.subtract(np.asmatrix(arrayY).T, np.matmul(tmpArrayX, WR)))))

    for epoch in range(training_epochs):
        for (x, y) in zip(trainX, trainY):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        if epoch % display_step == 0:
            result = 0
            for (x, y) in zip(trainX, trainY):
                tmp = sess.run(lossS, feed_dict={X: x, Y: y})
                result += tmp
            if epoch % (display_step * 100) == 0:
                print("Training cost=", result, "\nW=", sess.run(W), "\nb=", sess.run(b), '\n')
            WP = np.vstack([sess.run(W), sess.run(b)])
            WE = np.sum(np.abs(np.subtract(WP, WR)))
            print("epoch=", epoch, " WE=", WE, " Training cost=", result)
        epoch = epoch + 1

    result = 0
    for (X, Y) in zip(trainX, trainY):
        result += sess.run(lossS, feed_dict={x: X, y: Y})
    print("Final Training cost=", result, "W=", sess.run(W), "b=", sess.run(b), '\n')

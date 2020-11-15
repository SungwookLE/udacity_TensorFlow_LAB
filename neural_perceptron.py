import numpy as np
import matplotlib.pyplot as plt
# feedforward는 이해를 햇는데. (멀티 레이어 에서)
# feedback 의 수식을 잘 모르겟네 그니까 딥레이어의 업데이트 law는 어떻게 되는거지? (11/15)


data_a = np.random.randn(100, 2)
data_a[(data_a[:, 0] >= 0) & (data_a[:, 1] >= 0)] = [-0.1, -0.1]

data_b = np.random.randn(100, 2)
data_b[(data_b[:, 0] <= 0) | (data_b[:, 1] <= 0)] = [+0.1, +0.1]


label_a = np.ones((100, 1))
label_b = np.zeros((100, 1))

group_a = np.concatenate((data_a, label_a), axis=1)
group_b = np.concatenate((data_b, label_b), axis=1)


def activate_step(t):
    if t >= 0:
        y = 1
    else:
        y = 0

    return y


def activate_sigmoid(t):
    y = 1 / (1 + np.exp(-t))

    return y


def activate_relu(t):
    if t >= 1:
        y = 1
    elif t >= 0:
        y = t
    else:
        y = 0
    return y


def logistic_regression(data, Learn_rate, W, b, opt=("sigmoid")):

    W1 = W[0]
    W2 = W[1]
    y_hat = np.zeros((len(data[:, 2]), 1))

    for i in range(len(data[:, 2])):
        if opt == "sigmoid":
            y_hat[i] = activate_sigmoid(
                W[0] * data[i, 0] + W[1] * data[i, 1] + b)
        elif opt == "relu":
            y_hat[i] = activate_relu(W[0] * data[i, 0] + W[1] * data[i, 1] + b)
        else:
            y_hat[i] = activate_step(W[0] * data[i, 0] + W[1] * data[i, 1] + b)

        W1 = W1 + Learn_rate * (data[i, 2] - y_hat[i]) * (data[i, 0])
        W2 = W2 + Learn_rate * (data[i, 2] - y_hat[i]) * (data[i, 1])
        b = b + Learn_rate * (data[i, 2] - y_hat[i]) * 1

    W_new = [W1, W2]
    return W_new, b


W_h1 = np.random.randn(2, 1)
b_h1 = np.random.randn(1)
Learn_rate = 0.01

W_h2 = np.random.randn(2, 1)
b_h2 = np.random.randn(1)

iteration = 2000
total = np.concatenate((group_a, group_b), axis=0)

# first layer
for i in range(iteration):
    W_h1, b_h1 = logistic_regression(
        total, Learn_rate, W_h1, b_h1, opt="sigmoid")
    W_h2, b_h2 = logistic_regression(
        total, Learn_rate, W_h2, b_h2, opt="step")


print("W_h1=", W_h1, "b_h1=", b_h1, sep='\n')
print("W_h2=", W_h2, "b_h2=", b_h2, sep='\n')


predict_h1 = activate_sigmoid(
    W_h1[0] * total[:, 0] + W_h1[1] * total[:, 1] + b_h1)


predict_h2 = np.zeros((1, len(total[:, 0])))
for i in range(len(total[:, 0])):
    predict_h2[0, i] = activate_step(
        W_h2[0] * total[i, 0] + W_h2[1] * total[i, 1] + b_h2)

"""
predict_h2 = np.zeros((1, len(total[:, 0])))
for i in range(len(total[:, 0])):
    predict_h2[0, i] = activate_step(
        W_h2[0] * total[i, 0] + W_h2[1] * total[i, 1] + b_h2)
"""

# second layer
W = np.random.randn(2, 1)
b = np.random.randn(1)
Learn_rate = 0.05


res = np.vstack([np.array(predict_h1), np.array(predict_h2)])
total = np.vstack([res, total[:, 2]]).T

for i in range(iteration):
    W, b = logistic_regression(total, Learn_rate, W, b, opt="sigmoid")

predict = activate_sigmoid(W[0] * total[:, 0] + W[1] * total[:, 1] + b)


# Visualization
plt.tight_layout()
plt.plot(group_a[:, 0], group_a[:, 1], '.r')
plt.plot(group_b[:, 0], group_b[:, 1], '.b')

plt_x = np.zeros((len(predict), 1))
plt_y_h1 = np.zeros((len(predict), 1))
plt_y_h2 = np.zeros((len(predict), 1))
for i in range(len(predict)):
    plt_x[i] = (i - 100) * 0.025
    plt_y_h1[i] = -W_h1[0] / W_h1[1] * plt_x[i] - b_h1 / W_h1[1]
    plt_y_h2[i] = -W_h2[0] / W_h2[1] * plt_x[i] - b_h2 / W_h2[1]

    if (np.around(predict[i]) == total[i, 2]):
        if i < 100:
            plt.plot(group_a[i, 0], group_a[i, 1], 'ro')
        else:
            plt.plot(group_b[i-100, 0], group_b[i-100, 1], 'bo')


plt.plot(plt_x, plt_y_h1, '--r')
plt.plot(plt_x, plt_y_h2, '--b')
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.legend(('group A', 'group B'))
plt.show()

# feedforward는 이해를 햇는데. (멀티 레이어 에서)
# feedback 의 수식을 잘 모르겟네 그니까 딥레이어의 업데이트 law는 어떻게 되는거지? (11/15)

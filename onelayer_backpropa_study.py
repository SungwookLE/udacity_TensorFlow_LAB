import numpy as np
import matplotlib.pyplot as plt

# Prepare data_set
data_a = np.random.randn(100, 2)
data_a[(data_a[:, 0] >= -0.1) & (data_a[:, 1] >= -0.1)] = - \
    0.5 * abs(np.random.rand(1, 2))

data_b = np.random.randn(100, 2)
data_b[(data_b[:, 0] <= 0.1) | (data_b[:, 1] <= 0.1)
       ] = 0.5 * abs(np.random.rand(1, 2))

label_a = np.ones((100, 1))
label_b = np.zeros((100, 1))

group_a = np.concatenate((data_a, label_a), axis=1)
group_b = np.concatenate((data_b, label_b), axis=1)

data_set = np.concatenate((group_a, group_b), axis=0)
features = data_set[:, 0:2]
labels = data_set[:, 2]

# Initialization Parameter
n_feature = 2
n_output = 1
iteration = 1000
learn_rate = 0.01
W = np.random.randn(n_feature, 1)
b = np.zeros(n_output)

# Activate function


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
    if t >= 0:
        y = t
    else:
        y = 0
    return y

# ForwardPropagation


def forward_prop(X, W, b, opt='sigmoid'):
    net_out = np.zeros(len(X[:, 0]))
    activate_out = np.zeros(len(X[:, 0]))

    for i in range(len(X[:, 0])):
        net_out[i] = (np.matmul(X[i, :], W) + b)

        if (opt == 'sigmoid'):
            activate_out[i] = activate_sigmoid(net_out[i])
        elif (opt == 'relu'):
            activate_out[i] = activate_relu(net_out[i])

    return activate_out

# BackwardPropagation


def backward_prop(X, Y, W, b, learn_rate=0.01):
    net_out = np.zeros(len(Y))
    activate_out = np.zeros(len(Y))

    for i in range(len(Y)):
        net_out[i] = (np.matmul(X[i, :], W) + b)
        activate_out[i] = activate_sigmoid(net_out[i])

        W[0] = W[0] + learn_rate*(Y[i]-activate_out[i]) * \
            (1-activate_out[i])*activate_out[i]*X[i, 0]
        W[1] = W[1] + learn_rate*(Y[i]-activate_out[i]) * \
            (1-activate_out[i])*activate_out[i]*X[i, 1]
        b = b + learn_rate * (Y[i] - activate_out[i]) * \
            (1 - activate_out[i]) * activate_out[i] * 1

    return W, b


line_x = np.linspace(-3, 3, 100)
line_y = -W[0] / W[1] * line_x - b / W[1]
plt.plot(line_x, line_y, '--k', label='initial line')

# Run the whole batch (one shot)
for i in range(iteration):
    activate_out = forward_prop(features, W, b, opt='sigmoid')
    W, b = backward_prop(features, labels, W, b, 0.01)


# Visualization
line_x = np.linspace(-3, 3, 100)
line_y = -W[0] / W[1] * line_x - b / W[1]
accuracy = 0
for i in range(len(labels)):
    if i < 100:
        if activate_out[i] > 0.7:
            plt.plot(features[i, 0], features[i, 1], '.r')
            accuracy += 1
    else:
        if activate_out[i] < 0.3:
            plt.plot(features[i, 0], features[i, 1], '.b')
            accuracy += 1

plt.scatter(group_a[:, 0], group_a[:, 1], color='red',
            facecolors='none', label='group_a')
plt.scatter(group_b[:, 0], group_b[:, 1], color='blue',
            facecolors='none', label='group_b')

plt.plot(line_x, line_y, 'b', label='final line')
plt.tight_layout()
plt.xlim((-3, 3))
plt.ylim((-3, 3))
plt.legend()
plt.title('1 layer: sigmoid')
plt.text(-2, -2, 'accuracy:' + str(accuracy/200*100) + ' %')

print('Accuracy: ', accuracy/200 * 100, '%')
plt.show()

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
n_hidden = 2
n_output = 1
iteration = 1000
learn_rate = 0.01

W_hidden = np.random.randn(n_feature, n_hidden)
b_hidden = np.zeros(n_hidden)

W_out = np.random.randn(n_hidden, n_output)
b_out = np.zeros(n_output)

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
    if t > 0:
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


def backward_prop(cache, X, Y, W, b, learn_rate=0.01, operate=False, operate_method='sigmoid'):
    # works as sigmoid

    net_out = np.zeros(len(Y))
    activate_out = np.zeros(len(Y))

    if operate == False:
        for i in range(len(Y)):
            net_out[i] = (np.matmul(X[i, :], W) + b)
            activate_out[i] = activate_sigmoid(net_out[i])

            W[0] = W[0] + learn_rate*(Y[i]-activate_out[i]) * \
                (1-activate_out[i])*activate_out[i]*X[i, 0]
            W[1] = W[1] + learn_rate*(Y[i]-activate_out[i]) * \
                (1-activate_out[i])*activate_out[i]*X[i, 1]
            b = b + learn_rate * (Y[i] - activate_out[i]) * \
                (1 - activate_out[i]) * activate_out[i] * 1

    else:
        for i in range(len(Y)):
            if operate_method == 'relu':
                net_out[i] = (np.matmul(X[i, :], W) + b)
                activate_out[i] = activate_relu(net_out[i])

                if net_out[i] > 0:
                    W[0] = W[0] + learn_rate * \
                        (-cache[i]) * 1 * X[i, 0]
                    W[1] = W[1] + learn_rate * \
                        (-cache[i]) * 1 * X[i, 1]
                    b = b + learn_rate * (-cache[i]) * 1
                else:
                    W[0] = W[0]
                    W[1] = W[1]
                    b = b
            else:
                net_out[i] = (np.matmul(X[i, :], W) + b)
                activate_out[i] = activate_sigmoid(net_out[i])

                W[0] = W[0] + learn_rate * \
                    (-cache[i]) * (activate_out[i]) * \
                    (1 - activate_out[i]) * X[i, 0]
                W[1] = W[1] + learn_rate * \
                    (-cache[i]) * (activate_out[i]) * \
                    (1 - activate_out[i]) * X[i, 1]
                b = b + learn_rate * \
                    (-cache[i]) * (activate_out[i]) * (1 - activate_out[i])

    return W, b


cache_node1 = np.zeros(len(labels))
cache_node2 = np.zeros(len(labels))
cache_empty = np.zeros(len(labels))

# Run the whole batch (one shot)
for i in range(iteration):
    hidden_node1_activate_out = forward_prop(
        features, (W_hidden[0][0], W_hidden[1][0]), b_hidden[0], opt='relu')
    hidden_node2_activate_out = forward_prop(
        features, (W_hidden[0][1], W_hidden[1][1]), b_hidden[1], opt='relu')

    hidden_activate_out = np.array(
        [hidden_node1_activate_out, hidden_node2_activate_out]).T

    output_activate_out = forward_prop(
        hidden_activate_out, W_out, b_out, opt='sigmoid')

    W_out, b_out = backward_prop(cache_empty,
                                 hidden_activate_out, labels, W_out, b_out, 0.01)

    for i in range(len(labels)):
        cache_node1[i] = -(labels[i] - output_activate_out[i]) * \
            output_activate_out[i] * (1 - output_activate_out[i]) * W_out[0]
        cache_node2[i] = -(labels[i] - output_activate_out[i]) * \
            output_activate_out[i] * (1 - output_activate_out[i]) * W_out[1]

    [W_hidden[0][0], W_hidden[1][0]], b_hidden[0] = backward_prop(cache_node1,
                                                                  features, labels, [W_hidden[0][0], W_hidden[1][0]], b_hidden[0], 0.01, True, 'relu')
    [W_hidden[0][1], W_hidden[1][1]], b_hidden[1] = backward_prop(cache_node2,
                                                                  features, labels, [W_hidden[0][1], W_hidden[1][1]], b_hidden[1], 0.01,  True, 'relu')


# Visualization
line_x = np.linspace(-3, 3, 100)
line_y1 = -W_hidden[0][0] / W_hidden[1][0] * \
    line_x - b_hidden[0] / W_hidden[1][0]
line_y2 = -W_hidden[0][1] / W_hidden[1][1] * \
    line_x - b_hidden[1] / W_hidden[1][1]
accuracy = 0
for i in range(len(labels)):
    if i < 100:
        if output_activate_out[i] > 0.7:
            plt.plot(features[i, 0], features[i, 1], '.r')
            accuracy += 1
    else:
        if output_activate_out[i] < 0.3:
            plt.plot(features[i, 0], features[i, 1], '.b')
            accuracy += 1

plt.scatter(group_a[:, 0], group_a[:, 1], color='red',
            facecolors='none', label='group_a')
plt.scatter(group_b[:, 0], group_b[:, 1], color='blue',
            facecolors='none', label='group_b')

plt.plot(line_x, line_y1, '--r', label='line_node1')
plt.plot(line_x, line_y2, '--b', label='line_node2')

plt.tight_layout()
plt.xlim((-3, 3))
plt.ylim((-3, 3))
plt.legend()
plt.tight_layout()
plt.text(-2, -2, 'accuracy:' + str(accuracy/200*100) + ' %')
plt.title('2 layer: Hidden layers are relu')
print('Accuracy: ', accuracy/200 * 100, '%')
plt.show()

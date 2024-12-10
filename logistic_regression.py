import numpy as np
import pandas as pd
import math


# ----------------------------------------
# Define function
# ----------------------------------------
def calc_logit(w: np.ndarray, b: np.ndarray, x_i: np.ndarray):
    return np.dot(w, x_i) + b


def calc_sigmoid(w: np.ndarray, b: np.ndarray, x_i: np.ndarray):
    z = calc_logit(w, b, x_i)
    return 1 / (1 + np.exp(-z))


def calc_y_class(y_hat: float, threshold: float):
    if y_hat > threshold:
        return 1
    else:
        return 0


def calc_loss_function(y_i: float, y_hat_i: float):
    return -y_i * math.log(y_hat_i) - (1 - y_i) * math.log(1 - y_hat_i)


def calc_cost_function(x_train: np.ndarray, y_train: np.ndarray, w: np.ndarray, b: float):
    m = x_train.shape[0]
    total = 0

    for i in range(m):
        x_i = x_train[i]
        y_i = y_train[i]
        y_hat_i = calc_sigmoid(w, b, x_i)
        loss = calc_loss_function(y_i, y_hat_i)
        total = total + loss

    return total / m


def calc_dj_per_dw(x_train: np.ndarray, y_train: np.ndarray, w: np.ndarray, b: float):
    m = x_train.shape[0]
    total = 0

    for i in range(m):
        x_i = x_train[i]
        y_i = y_train[i]
        y_hat_i = calc_sigmoid(w, b, x_i)
        total = total + ((y_hat_i - y_i) * x_i)

    return 1 / m * total


def calc_dj_per_db(x_train: np.ndarray, y_train: np.ndarray, w: float, b: float):
    m = x_train.shape[0]
    total = 0

    for i in range(m):
        x_i = x_train[i]
        y_i = y_train[i]
        y_hat_i = calc_sigmoid(w, b, x_i)
        total = total + (y_hat_i - y_i)

    return 1 / m * total


def find_optimum_w_and_b(x_train: np.ndarray, y_train: np.ndarray, alpha: float, w_init: np.ndarray, b_init: float,
                         batch_count: int):
    w_optimum = w_init
    b_optimum = b_init
    m = x_train.shape[0]
    iteration_history = np.zeros((batch_count, 10))

    for iter in range(batch_count):
        iteration_history[iter][0] = w_optimum[0]  # w1
        iteration_history[iter][1] = w_optimum[1]  # w2
        iteration_history[iter][2] = b_optimum  # b
        iteration_history[iter][3] = calc_cost_function(x_train, y_train, w_optimum, b_optimum)  # j_wb

        dj_per_dw = calc_dj_per_dw(x_train, y_train, w_optimum, b_optimum)
        dj_per_db = calc_dj_per_db(x_train, y_train, w_optimum, b_optimum)

        w_temp = w_optimum - alpha * dj_per_dw
        b_temp = b_optimum - alpha * dj_per_db
        w_optimum = w_temp
        b_optimum = b_temp

        iteration_history[iter][4] = dj_per_dw[0]  # dj/dw1
        iteration_history[iter][5] = dj_per_dw[1]  # dj/dw2
        iteration_history[iter][6] = dj_per_db  # dj/db
        iteration_history[iter][7] = w_optimum[0]  # New w1
        iteration_history[iter][8] = w_optimum[1]  # New w2
        iteration_history[iter][9] = b_optimum  # New b

    return w_optimum, b_optimum, iteration_history


# ----------------------------------------
# Running
# ----------------------------------------
x_train = np.array([[34.6236596245169, 78.0246928153624],
                    [30.286710768226, 43.894997524001],
                    [35.8474087699387, 72.9021980270836],
                    [60.1825993862097, 86.3085520954682],
                    [79.0327360507101, 75.3443764369103]])

y_train = np.array([0, 0, 0, 1, 1])
alpha = 0.001
w_init = np.array([-0.00082978, 0.00220324])
b_init = -8

batch_count = 1000
threshold = 0.5

w_optimum, b_optimum, iteration_data = find_optimum_w_and_b(x_train, y_train, alpha, w_init, b_init, batch_count)

print(f"x_train\n {x_train}\n")
print(f"y_train\n {y_train}\n")
print(f"Optimum w\n{w_optimum}\n")
print(f"Optimum b\n{b_optimum}\n")

pd.set_option('display.width', 150)
dataframe = pd.DataFrame(iteration_data)
dataframe.columns = ["w1", "w2", "b", "Cost (j_wb)", "dj / dw1", "dj / dw2", "dj / db", "New w1", "New w2", "New b"]
print(dataframe)

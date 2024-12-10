import numpy as np
import pandas as pd


# ----------------------------------------
# Define function
# ----------------------------------------
def calc_linear_regression_model(w, b, x_i):
    return np.dot(w, x_i) + b


def calc_cost_function(x_train: np.ndarray, y_train: np.ndarray, w: np.ndarray, b: float):
    m = x_train.shape[0]
    total = 0

    for i in range(m):
        x_i = x_train[i]
        y_i = y_train[i]
        y_hat_i = calc_linear_regression_model(w, b, x_i)
        total = total + ((y_hat_i - y_i) ** 2)

    return 1 / (2 * m) * total


def calc_dj_per_dw(x_train: np.ndarray, y_train: np.ndarray, w: np.ndarray, b: float):
    m = x_train.shape[0]
    total = 0

    for i in range(m):
        x_i = x_train[i]
        y_i = y_train[i]
        y_hat_i = calc_linear_regression_model(w, b, x_i)
        total = total + ((y_hat_i - y_i) * x_i)

    return 1 / m * total


def calc_dj_per_db(x_train: np.ndarray, y_train: np.ndarray, w: float, b: float):
    m = x_train.shape[0]
    total = 0

    for i in range(m):
        x_i = x_train[i]
        y_i = y_train[i]
        y_hat_i = calc_linear_regression_model(w, b, x_i)
        total = total + (y_hat_i - y_i)

    return 1 / m * total


def find_optimum_w_and_b(x_train: np.ndarray, y_train: np.ndarray, alpha, w_init: np.ndarray, b_init, batch_count):
    w_optimum = w_init
    b_optimum = b_init
    m = x_train.shape[0]
    iteration_history = np.zeros((batch_count, 16))

    for iter in range(batch_count):
        iteration_history[iter][0] = w_optimum[0]  # w1
        iteration_history[iter][1] = w_optimum[1]  # w2
        iteration_history[iter][2] = w_optimum[2]  # w3
        iteration_history[iter][3] = w_optimum[3]  # w4
        iteration_history[iter][4] = b_optimum  # b
        iteration_history[iter][5] = calc_cost_function(x_train, y_train, w_optimum, b_optimum)  # j_wb

        dj_per_dw = calc_dj_per_dw(x_train, y_train, w_optimum, b_optimum)
        dj_per_db = calc_dj_per_db(x_train, y_train, w_optimum, b_optimum)

        w_temp = w_optimum - alpha * dj_per_dw
        b_temp = b_optimum - alpha * dj_per_db
        w_optimum = w_temp
        b_optimum = b_temp

        iteration_history[iter][6] = dj_per_dw[0]  # dj/dw1
        iteration_history[iter][7] = dj_per_dw[1]  # dj/dw2
        iteration_history[iter][8] = dj_per_dw[2]  # dj/dw3
        iteration_history[iter][9] = dj_per_dw[3]  # dj/dw4
        iteration_history[iter][10] = dj_per_db  # dj/db
        iteration_history[iter][11] = w_optimum[0]  # New w1
        iteration_history[iter][12] = w_optimum[1]  # New w2
        iteration_history[iter][13] = w_optimum[2]  # New w3
        iteration_history[iter][14] = w_optimum[3]  # New w4
        iteration_history[iter][15] = b_optimum  # New b

    return w_optimum, b_optimum, iteration_history


# ----------------------------------------
# Running
# ----------------------------------------
x_train = np.array([[2104, 5, 1, 45],
                    [1416, 3, 2, 40],
                    [1534, 3, 2, 30],
                    [852, 2, 1, 36],
                    [1500, 2, 1, 54]])
y_train = np.array([460, 232, 315, 178, 120])
alpha = 0.0000001
w_init = np.array([0.01, 0.01, 0.01, 0.01])
b_init = 0.01

batch_count = 1000

w_optimum, b_optimum, iteration_data = find_optimum_w_and_b(x_train, y_train, alpha, w_init, b_init, batch_count)

print(f"x_train\n {x_train}\n")
print(f"y_train\n {y_train}\n")
print(f"Optimum w\n{w_optimum}\n")
print(f"Optimum b\n{b_optimum}\n")

pd.set_option('display.width', 150)
dataframe = pd.DataFrame(iteration_data)
dataframe.columns = ["w1", "w2", "w3", "w4", "b", "Cost (j_wb)", "dj / dw1", "dj / dw2", "dj / dw3", "dj / dw4",
                     "dj / db", "New w1", "New w2", "New w3", "New w4", "New b"]
print(dataframe)
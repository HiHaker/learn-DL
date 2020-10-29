# 这里使用的假设函数是 y = wx + b，参数分别为 w b
# 所以代价函数，或者说损失函数，loss = (wx + b - y)^2，将所有样本点带入计算得到一个总的损失
# 计算损失的函数
def compute_error_line_given_points(w, b, points):
    total_error = 0
    for point in points:
        x = point[0]
        y = point[1]
        total_error = total_error + (w*x + b - y)**2
    return total_error / float(len(points))

# 一步梯度下降
# 计算的是偏导数
def step_gradient(w_current, b_current,
                  points, learning_rate):
    w_gradient = 0
    b_gradient = 0
    N = float(len(points))
    for point in points:
        x = point[0]
        y = point[1]
        w_gradient = w_gradient + 2*(w_current*x + b_current - y)*x / N
        b_gradient = b_gradient + 2*(w_current*x + b_current - y) / N

    w_new = w_current - learning_rate * w_gradient
    b_new = b_current - learning_rate*b_gradient

    return w_new, b_new

# 梯度下降算法
def gradient_runner(points, w_start, b_start,
                    learning_rate, num_iteration):
    w = w_start
    b = b_start

    # 执行梯度下降算法的次数
    for i in range(num_iteration):
        w, b = step_gradient(w, b, points, learning_rate)

    return w, b
import random
import matplotlib.pyplot as plt
import numpy as np


def create_data(data_dim: int, data_set_size: int):
    ds = {
        'x': [],
        'y': []
    }
    w = random.randint(-5, 5)
    for i in range(data_set_size):
        x = np.array([random.randint(-100, 100) for j in range(data_dim)])
        if data_dim == 1: x = x[0]
        if x<0:
            y = 0
        else:
            y= 1
        ds['x'].append(x)
        ds['y'].append(y)
    return ds


def normalize_ds(_ds: dict):
    x_data = _ds['x']

    mean_x_data = np.mean(x_data)
    std_x_data = np.std(x_data)
    x_data_normalized = (x_data - mean_x_data) / std_x_data

    _ds['x'] = x_data_normalized

    return _ds


def plot_ds(_ds, _w, _b, name: str):
    if _w is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax.scatter(_ds['x'], ds['y'])
        fig.savefig('./img/' + name + '.png')  # save the figure to file
        plt.close(fig)
    else:
        x_l = []
        y_l = []
        dd = np.linspace(start=min(_ds['x']), stop=max(_ds['x']), num=100)
        for i in range(len(dd)):
            x_i = dd[i]
            z = _w * x_i + b
            y_i = 1 / (1 + np.e ** (-z))
            x_l.append(x_i)
            y_l.append(y_i)

        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax.plot(x_l, y_l, color='red')
        ax.scatter(_ds['x'], ds['y'], color='green')
        fig.savefig('./img/' + name + '.png')  # save the figure to file
        plt.close(fig)
        pass


def log_reg(_ds: dict, _w: float, _b: float):
    lr = 0.1
    x_data = _ds['x']
    y_data = _ds['y']
    d_len = len(x_data)
    ''''''
    bs_size = 10
    loss = 0
    for j in range(d_len // bs_size):
        d_c_0 = 0
        d_c_1 = 0
        for i in range(bs_size):
            x = x_data[i]
            y = y_data[i]
            z = _w * x + b
            y_h = 1/(1+np.e**(-z))
            '''loss'''
            l0 = -((y * np.log(y_h)) + ((1-y) * np.log(1 - y_h)))

            loss += l0 / bs_size
            '''derivative'''
            # d_loss_d_yh = - (2.0 * (y - y_h))  # WE are going to minimize the loss
            # d_yh_d_z = z * (1-z)  # yh = 1/(1+np.e**(-z)) => d_yh/d_w = z* (1-z)
            # d_z_d_w = x
            # d_z_d_b = 1  # yh = w x + b => d_yh/d_b = 1
            # d_c_0 += (d_loss_d_yh * d_yh_d_z*d_z_d_w) / bs_size
            # d_c_1 += (d_loss_d_yh * d_yh_d_z*d_z_d_b) / bs_size

            d_c_0 += 1 * x * (y_h-y)
            d_c_1 += 1 * (y_h-y)

        _w = _w - lr * d_c_0
        _b = _b - lr * d_c_1
        pass

    return _w, _b, loss


if __name__ == '__main__':
    data_dim = 1
    ds = create_data(data_dim=data_dim, data_set_size=100)
    plot_ds(ds, None, None, 'Log_reg_data_norm')
    ds = normalize_ds(ds)
    # ''''''
    w = np.random.uniform(-0, 1)
    b = 0  # np.random.uniform(-1,1)
    epoch = 50
    for i in range(epoch):
        plot_ds(ds, w, b, 'Log_reg_data_norm' + str(i))
        w, b, loss = log_reg(_ds=ds, _w=w, _b=b)
        print(loss)
        # if i % 10 == 0 or i == epoch-1:
    # ''''''

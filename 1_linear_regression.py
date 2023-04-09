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
        x = np.array([random.randint(-10, 10) for j in range(data_dim)])
        if data_dim == 1: x = x[0]
        y = (w * x) + random.randint(-20, 20)  # we want to learn sum operation

        ds['x'].append(x)
        ds['y'].append(y)
    return ds


def normalize_ds(_ds: dict):
    x_data = _ds['x']
    y_data = _ds['y']
    mean_x_data = np.mean(x_data)
    std_x_data = np.std(x_data)

    mean_y_data = np.mean(y_data)
    std_y_data = np.std(y_data)

    x_data_normalized = (x_data - mean_x_data) / std_x_data
    y_data_normalized = (y_data - mean_y_data) / std_y_data
    _ds['x'] = x_data_normalized
    _ds['y'] = y_data_normalized
    return _ds


def plot_ds(_ds, _w, _b, name: str):
    if _w is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax.scatter(_ds['x'], ds['y'])
        fig.savefig('./img/' + name+'.png')  # save the figure to file
        plt.close(fig)
    else:
        min_x = min(_ds['x'])
        y_0 = _w * min_x + b
        max_x = max(_ds['x'])
        y_1 = _w * max_x + b

        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax.plot([min_x, max_x], [y_0, y_1], color='red')
        ax.scatter(_ds['x'], ds['y'], color='green')
        fig.savefig('./img/' + name + '.png')  # save the figure to file
        plt.close(fig)
        pass


def lin_reg(_ds: dict, _w: float, _b: float):
    lr = 0.1
    x_data = _ds['x']
    y_data = _ds['y']
    d_len = len(x_data)
    ''''''
    bs_size = 10
    loss = 0
    for j in range(d_len//bs_size):
        d_c_0 = 0
        d_c_1 = 0
        for i in range(bs_size):
            x = x_data[i]
            y = y_data[i]
            y_h = _w * x + b
            '''loss'''
            loss += ((y - y_h) ** 2)/bs_size
            '''derivative'''
            d_loss_d_yh = 2.0 * (y - y_h)
            d_yh_d_w = x  # yh = w x + b => d_yh/d_w = x
            d_yh_d_b = 1  # yh = w x + b => d_yh/d_b = 1
            d_c_0 += -1 * (d_loss_d_yh * d_yh_d_w)/bs_size
            d_c_1 += -1 * (d_loss_d_yh * d_yh_d_b)/bs_size

        _w = _w - lr * d_c_0
        _b = _b - lr * d_c_1
        pass

    return _w, _b, loss


if __name__ == '__main__':
    data_dim = 1
    ds = create_data(data_dim=data_dim, data_set_size=100)
    plot_ds(ds, None, None, 'data')
    ds = normalize_ds(ds)
    plot_ds(ds, None, None, 'data_norm')

    # ''''''
    w = np.random.uniform(-1, 1)
    b = 0 #np.random.uniform(-1,1)
    epoch = 10
    for i in range(epoch):
        plot_ds(ds, w, b, 'data_norm' + str(i))
        w, b, loss = lin_reg(_ds=ds, _w=w, _b=b)
        print(loss)
        # if i % 10 == 0 or i == epoch-1:
    # ''''''

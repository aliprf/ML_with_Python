import glob
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def create_data(data_dim: int, data_set_size: int):
    ds = {
        'x': []
    }
    for i in range(data_set_size):
        if i % data_dim == 0:
            x = np.array([random.randint(-10, 0) for j in range(data_dim)])
        elif i % data_dim == 1:
            x = np.array([random.randint(1, 5) for j in range(data_dim)])
        else:
            x = np.array([random.randint(10, 20) for j in range(data_dim)])
        ds['x'].append(x)
    return ds


def normalize_ds(_ds: dict):
    x_data = _ds['x']

    mean_x_data = np.mean(x_data)
    std_x_data = np.std(x_data)
    x_data_normalized = (x_data - mean_x_data) / std_x_data

    _ds['x'] = x_data_normalized

    return _ds


def plot_ds(_ds, _centers, name: str):
    ds_x = []
    ds_y = []
    c_x = []
    c_y = []
    for i in range(len(_ds)):
        ds_x.append(_ds[i][0])
        ds_y.append(_ds[i][1])
    for i in range(len(_centers)):
        c_x.append(_centers[i][0])
        c_y.append(_centers[i][1])

    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.scatter(c_x, c_y, color='red')
    ax.scatter(ds_x, ds_y, color='green')
    fig.savefig('./img/' + name + '.png')  # save the figure to file
    plt.close(fig)


def knn(_k: int, _ds_lst: list, _centers):
    if _centers is None:
        '''randomly init k centers'''


    clusters = {}
    for i in range(k):
        clusters[i] = []
    '''assign to cluster'''
    for i in range(len(_ds_lst)):
        selected_cluster_index = 0
        dist = math.dist(_ds_lst[i], _centers[0])

        for j in range(1, len(_centers)):
            if math.dist(_ds_lst[i], _centers[j]) < dist:
                selected_cluster_index = j
        clusters[selected_cluster_index].append(i)
    '''update centers '''
    for i in range(k):
        sum_x = 0
        sum_y = 0
        for j in range(len(clusters[i])):
            sum_x += _ds_lst[clusters[i][j]][0]
            sum_y += _ds_lst[clusters[i][j]][1]
        _centers[i] = [sum_x / j, sum_y / j]
    return _centers


def make_gif(frame_folder):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.png")]
    frame_one = frames[0]
    frame_one.save("knn.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=1)


if __name__ == '__main__':
    data_dim = 2
    ds = create_data(data_dim=data_dim, data_set_size=100)
    # plot_ds(ds, None, None, 'Log_reg_data_norm')
    ds = normalize_ds(ds)

    k = 3
    iterations = 10

    centers = []
    _ds_lst = ds['x']
    '''init centers'''
    for i in range(k):
        centers.append(_ds_lst[random.randint(0, len(_ds_lst) - 1)])

    for i in range(iterations):
        plot_ds(_ds=ds['x'], _centers=centers, name='KNN_' + str(i))
        centers = knn(k, _ds_lst, centers)
    make_gif('./img/')
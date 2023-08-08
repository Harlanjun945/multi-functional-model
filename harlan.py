# 相关模块的引入
import os
import time
from qutip import *
import matplotlib.pyplot as plt
import itertools
import matplotlib as mp
import numpy as np
from functools import partial
from joblib import Parallel, delayed
import pandas as pd
# -------------------------量子力学相关定义------------------------------------------------
num = 2  # 最大光子数
PI = np.pi  # 定义π
PI2 = 2*np.pi  # 定义2π
a = destroy(num)  # 湮灭算符定义
a_dag = a.dag()  # 创生算符定义
vac = fock(num, 0)  # 腔真空态定义
# 基矢定义
z0 = basis(2, 0)  # 比特0
z1 = basis(2, 1)  # 比特1
x0 = (z0+z1).unit()  # 比特 0+1
x1 = (z0-z1).unit()  # 比特 0-1
y0 = (z0+1j*z1).unit()  # 比特 0+j1
y1 = (z0-1j*z1).unit()  # 比特 0-j1
# 泡利算符定义
sx = sigmax()  # x泡利算符
sy = sigmay()  # y泡利算符
sz = sigmaz()  # z泡利算符
# -------------------------------------------常用函数---------------------------------------
# 0.                                         创建文件夹


def mkdir(path):  # 创建文件夹
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
    return None
# 1.                                          保存数据


def save(name, data):
    """
    Save the data as an Excel file with the given name.

    Parameters:
    - name (str): The name of the Excel file.
    - data (list): The data to be saved as an Excel file.

    Returns:
    None
    """
    mkdir('.\\data')  # 调用函数
    data = np.array(data)
    data = pd.DataFrame(data)
    data.to_excel('.\\data\\{}.xlsx'.format(name))		# ‘page_1’是写入excel的sheet名

# 2.                                          读取数据


def load(name):
    """
    Load data from an Excel file.

    Parameters:
    - name (str): The name of the Excel file (without the extension).

    Returns:
    - data (ndarray): The loaded data as a numpy array.
    """
    sheet = pd.read_excel(io='.\\data\\{}.xlsx'.format(name))
    to_array = np.array(sheet)
    data = np.delete(to_array, 0, axis=1)
    if data.shape[1] == 1:  # 一维数据特殊处理
        data = data.T[0]
    print(data)
    return data

# 3.                                           检索数据


def search():
    """
    Searches for files with the 'xlsx' extension in the './data' directory and prints the file names without the extension.
    """
    path = os.listdir('./data')
    data = [i for i in path if i.split('.')[1] == 'xlsx']
    for i in data:
        print(i.split('.')[0])

# 4.                                           删除数据


def delete(name):
    """
    Deletes a file with the given name.

    Parameters:
        name (str): The name of the file to be deleted.

    Returns:
        None
    """
    os.remove('.\\data\\{}.xlsx'.format(name))
# 5.                                           布洛赫球


def bloch_d(result, lg0, lg1):  # 动态布洛赫
    # result：mesolve函数返回的参数
    # lg0:布洛赫球态空间中的z轴量子态 ；lg1：布洛赫球态空间的-z轴量子态
    sx = lg0*lg1.dag() + lg1*lg0.dag()
    sz = lg0*lg0.dag() - lg1*lg1.dag()
    sy = 1j*(lg1*lg0.dag() - lg0*lg1.dag())
    b = Bloch()
    b.make_sphere()
    for i in np.arange(0, len(result.states)):
        """
        A function that calculates the Bloch vectors for a given set of quantum states.

        Args:
            result (type): The result returned by the mesolve function.
            lg0 (type): The quantum state along the z-axis in the Bloch sphere.
            lg1 (type): The quantum state along the -z-axis in the Bloch sphere.

        Returns:
            None
        """
        x = expect(sx, result.states[i])
        y = expect(sy, result.states[i])
        z = expect(sz, result.states[i])
        b.add_vectors([x, y, z])
        b.show()
        time.sleep(0.01)
        if i == len(result.states) - 1:
            pass
        else:
            b.clear()


def bloch_s(result, lg0, lg1):  # 静态布洛赫
    sx = lg0*lg1.dag() + lg1*lg0.dag()
    sz = lg0*lg0.dag() - lg1*lg1.dag()
    sy = 1j*(lg1*lg0.dag() - lg0*lg1.dag())
    b = Bloch()
    b.make_sphere()

    x = expect(sx, result.states)
    y = expect(sy, result.states)
    z = expect(sz, result.states)
    b.add_points([x, y, z])
    b.show()

# 6.                                    并行计算程序


def parallel(para, savebundle='False'):  # 多线程计算
    para_list_key = []
    para_supplementary_set = {}
    for i in para:
        if isinstance(para[i], list) or isinstance(para[i], np.ndarray):
            """
            A function that performs parallel computation using multiple threads.

            Args:
                para (dict): A dictionary containing the parameters for the computation. It can have both list and non-list parameters.
                savebundle (str, optional): A string indicating whether to save the computed data bundle. Defaults to 'False'.

            Returns:
                numpy.ndarray or float or tuple: The computed data set. The return type depends on the dimensions of the data set and the value of `savebundle` parameter.
                    - If the data set is three-dimensional, it returns a numpy.ndarray of shape (len(para[para_list_key[0]]), len(para[para_list_key[1]]), -1).
                    - If the data set is two-dimensional, it returns a numpy.ndarray of shape (len(para[para_list_key[0]]), -1).
                    - If the data set is one-dimensional, it returns a numpy.ndarray of shape (-1).
                    - If the data set is zero-dimensional, it returns a float.
                    - If `savebundle` is set to 'True', it returns a tuple containing the computed data bundle.

            Raises:
                ValueError: If the number of iterations exceeds two.

            """
            para_list_key.append(i)  # 创建列表用于存放参数字典内的列表参数的key
        else:
            para_supplementary_set.update({i: para[i]})  # 创建字典存放非列表参数
    if len(para_list_key) > 2:  # 该程序仅获取三维以内数据，超出则报错
        raise ValueError("Up to two iterations.")
    yx = [para[key] for key in para_list_key]  # 获取列表参数并依次存放于一个新的列表
    yxgrid = list(itertools.product(*yx))  # 将多个列表参数张量积为一个参数网络（x,y）
    para_set = []
    for i in yxgrid:
        ipara = {}
        para_supplementary_copy = para_supplementary_set.copy()
        ipara.update(para_supplementary_copy)
        for inumber, ivalue in enumerate(i):
            ipara.update({para_list_key[inumber]: ivalue})
        para_set.append(ipara)  # 参数参数网络依次将列表参数更新到字典内并存放于新的参数列表中
    parFunc = partial(func)  # 获取用于多线程计算的对象函数
    data_set = Parallel(n_jobs=6,
                        verbose=1)(delayed(parFunc)(para=i)
                                   for i in para_set)  # 根据参数列表para_set进行多线程计算
    data_bundle = (np.array(data_set).reshape(-1)).copy()
    dims = len(para_list_key) + (np.array(data_set).ndim-1)  # 获取该数据集的变量维度
    if dims == 3:
        data_set = np.array(data_set).reshape(
            len(para[para_list_key[0]]), len(para[para_list_key[1]]), -1)
        data_set = data_set[::-1, :, :]
    elif dims == 2:
        data_set = np.array(data_set).reshape(len(para[para_list_key[0]]), -1)
        data_set = data_set[::-1, :]
    elif dims == 1:
        data_set = np.array(data_set).reshape(-1)
    elif dims == 0:
        data_set = float(data_set[0])  # type: ignore
    if savebundle == 'True':
        return data_bundle  # 获得一维数据元组
    else:
        return data_set  # 根据变量维度分割变量数据形成数据集

# 7.                                           二维曲线图


def plot2D(ax, x, y, color, xlim, ylim, plot_label,
           xtick_keys, ytick_keys,
           xtick_values=None, ytick_values=None,
           legend='False', save='True'):
    linewidth = 0.5  # 坐标轴线宽
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['top'].set_linewidth(linewidth)
    # plt.grid(linestyle ='--', linewidth = 0.5) #网格线
    plt.rcParams['xtick.direction'] = 'in'  # x坐标轴刻度线朝内
    plt.rcParams['ytick.direction'] = 'in'  # y坐标轴刻度线朝内
    plt.plot(x, y,  '-', linewidth=1, markersize=3, color=color,
             label=plot_label)  # 画图
    plt.xlabel(r'x',
               fontdict={'family': 'Times New Roman', 'size': 10},
               labelpad=0.8)  # x轴标签参数设置
    plt.ylabel(r'y',
               fontdict={'family': 'Times New Roman', 'size': 10},
               labelpad=2)  # y轴标签参数设置
    plt.tick_params(labelsize=10)  # 坐标轴刻度标签字体大小设置
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]  # 坐标轴刻度字体设置
    plt.xticks(xtick_keys, xtick_values)  # type: ignore # x轴刻度标签显示
    plt.yticks(ytick_keys, ytick_values)  # type: ignore # y轴刻度标签显示
    plt.xlim(xlim)  # x轴范围
    plt.ylim(ylim)  # y轴范围
    if legend == 'True':
        leg = ax.legend(prop={'family': 'Times New Roman', 'size': 10}, fancybox=False,
                        frameon=False)  # 图例字体属性及形状属性
        leg.get_frame().set_edgecolor('black')  # 图例边框颜色
        leg.get_frame().set_linewidth(0.5)  # 图例边框粗细
    else:
        pass
    plt.tight_layout(pad=0.1)  # 紧密布局
    if save == 'True':
        plt.savefig('out.svg', format='svg')
    return ax
# 8.                                      画图热力图专用


def plot3D(ax, x, y, z, xlabel, ylabel, cblabel, vmin, vmax,
           xtick_keys, ytick_keys, cbtick_keys,
           xtick_values=None, ytick_values=None,
           save='True', cmap='jet', visual_value=[]):
    x = np.arange(len(x))
    y = np.arange(len(y))[::-1]
    linewidth = 0.5  # 坐标轴线宽
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['top'].set_linewidth(linewidth)
    plt.tick_params(bottom=False, top=False, left=False, right=False)  # 无刻度线
    cet = plt.imshow(z, vmin=vmin, vmax=vmax,
                     cmap=cmap, aspect='auto')  # 绘制热力图
    contour = plt.contour(x, y, z, visual_value,
                          linewidths=0.5, colors='k')  # 加轮廓线
    cl1 = plt.clabel(contour, fontsize=10, inline='False', colors='k')  # 加颜色条
    [tick.set(fontsize=10, family='Times New Roman')
     for tick in cl1]  # 轮廓线标签字体设置
    plt.tick_params(labelsize=10)  # 坐标轴刻度标签字体大小设置
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]  # 坐标轴刻度字体设置
    plt.xticks(xtick_keys, xtick_values)  # type: ignore
    plt.yticks(ytick_keys, ytick_values)  # type: ignore
    plt.xlabel(xlabel,
               fontdict={'family': 'Times New Roman', 'size': 10}, labelpad=0.8)
    plt.ylabel(ylabel,
               fontdict={'family': 'Times New Roman', 'size': 10}, labelpad=1)
    cb1 = plt.colorbar(cet, fraction=0.072, pad=0.04,
                       orientation='horizontal', location='top')  # 色条位置及大小
    cb1.outline.set_linewidth(0.5)  # 颜色条边框线宽
    cb1.set_ticks(cbtick_keys, update_ticks=True)  # 具体显示刻度值
    plt.setp(cb1.ax.xaxis.get_ticklabels(),
             fontsize=10, fontname="Times New Roman")  # 刻度字体属性
    cb1.ax.tick_params(direction='out', labelsize=10, width=0.5, pad=0.5)
    cb1.set_label(cblabel, x=-0.035, labelpad=-18,
                  fontdict={'family': 'Times New Roman', 'size': 10})  # 色条标题属性
    plt.tight_layout(pad=0.1)  # 紧密布局
    if save == 'True':
        plt.savefig('out.svg', format='svg')
    plt.show()
# --------------------------------------正文区-------------------------------------------
def func(para):  # 工作函数
    y = np.sin(para['theta'])
    return y


para = {'theta': np.arange(0, PI, 0.01)}
f1 = parallel(para)
f2 = -f1  # type: ignore
fig, ax = plt.subplots(figsize=(8/2.54, 6/2.54),
                       dpi=300)  # 图片大小和分辨率
plot2D(ax, para['theta'], f1, color='r',
       xlim=[0, PI], ylim=[-1, 1], plot_label=r'sin$\theta$',
       xtick_keys=[0, 1.57, 3.14], ytick_keys=[-1, 0, 1],
       xtick_values=[0, r'$\pi/2$', r'$\pi$'], save='False')
plot2D(ax, para['theta'], f2, color='b',
       xlim=[0, PI], ylim=[-1, 1], plot_label=r'sin$\theta$',
       xtick_keys=[0, 1.57, 3.14], ytick_keys=[-1, 0, 1],
       xtick_values=[0, r'$\pi/2$', r'$\pi$'])

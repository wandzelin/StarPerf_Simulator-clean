import numpy as np
import matplotlib.pyplot as plt

def plot_cdf(data_list, title="CDF Plot", xlabel="Value", ylabel="Cumulative Probability", save_path=None, dpi=300):
    """
    根据给定的list数据生成并绘制CDF图。

    Args:
        data_list (list): 包含数据的列表。
        title (str): 图表的标题。
        xlabel (str): X轴的标签。
        ylabel (str): Y轴的标签。
        save_path (str, optional): 图表保存路径，如果为None则不保存。
        dpi (int): 保存图片的分辨率，默认300。
    """
    if not data_list:
        print("Error: The input list is empty.")
        return

    # 1. 对数据进行排序
    sorted_data = np.sort(data_list)
    
    # 2. 计算每个数据点对应的累积概率
    # np.arange(1, len(sorted_data) + 1) 生成一个从1到N的数组
    # / len(sorted_data) 将其归一化为 [1/N, 2/N, ..., 1]
    y_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    # 3. 使用 Matplotlib 绘图
    plt.figure(figsize=(10, 6))  # 设置图表大小
    plt.plot(sorted_data, y_cdf, marker='.', linestyle='--', color='b')
    
    # 绘制阶梯图，这更符合离散数据的CDF定义
    plt.step(sorted_data, y_cdf, where='post', color='b')
    
    # 添加图表标题和坐标轴标签
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 添加网格线，使图表更易读
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 设置x轴和y轴的范围
    plt.xlim(sorted_data[0] - 1, sorted_data[-1] + 1)
    plt.ylim(0, 1.05)

    # 保存图表到指定位置（如果提供了save_path）
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"图表已保存到: {save_path}")
    
    # 显示图表
    plt.show()


def save_plot_only(fig, save_path, dpi=300, format='png'):
    """
    仅保存图表，不显示。
    
    Args:
        fig: matplotlib图表对象
        save_path (str): 保存路径
        dpi (int): 分辨率
        format (str): 图片格式，如'png', 'jpg', 'pdf', 'svg'
    """
    # 确保文件扩展名正确
    if not save_path.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
        save_path = f"{save_path}.{format}"
    
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"图表已保存到: {save_path}")


def plot_andSave(data_list, save_path, title="CDF Plot", xlabel="Value", ylabel="Cumulative Probability", dpi=300, show_plot=True):
    """
    绘制CDF图并保存，可选择是否显示。
    
    Args:
        data_list (list): 包含数据的列表
        save_path (str): 保存路径
        title (str): 图表标题
        xlabel (str): X轴标签
        ylabel (str): Y轴标签
        dpi (int): 保存分辨率
        show_plot (bool): 是否显示图表
    """
    if not data_list:
        print("Error: The input list is empty.")
        return
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 对数据进行排序
    sorted_data = np.sort(data_list)
    y_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    # 绘制图表
    ax.plot(sorted_data, y_cdf, marker='.', linestyle='--', color='b')
    ax.step(sorted_data, y_cdf, where='post', color='b')
    
    # 设置图表属性
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(sorted_data[0], sorted_data[-1])
    ax.set_ylim(0, 1.05)
    
    # 保存图表
    save_plot_only(fig, save_path, dpi, 'png')
    
    # 根据参数决定是否显示
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

import math
from math import radians, cos, sin, asin, sqrt
import numpy as np
from src.XML_constellation.constellation_routing.routing_policy_plugin.shortest_path import distance_between_satellite_and_user


def latilong_to_descartes(transformed_object , object_type , t=None):
    a = 6371000.0
    e2 = 0.00669438002290
    if object_type == "satellite":
        longitude = math.radians(transformed_object.longitude[t - 1])
        latitude = math.radians(transformed_object.latitude[t - 1])
        fac1 = 1 - e2 * math.sin(latitude) * math.sin(latitude)
        N = a / math.sqrt(fac1)
        # the unit of satellite height above the ground is meters
        h = transformed_object.altitude[t - 1] * 1000
        X = (N + h) * math.cos(latitude) * math.cos(longitude)
        Y = (N + h) * math.cos(latitude) * math.sin(longitude)
        Z = (N * (1 - e2) + h) * math.sin(latitude)
        return X, Y, Z
    else:
        longitude = math.radians(transformed_object.longitude)
        latitude = math.radians(transformed_object.latitude)
        fac1 = 1 - e2 * math.sin(latitude) * math.sin(latitude)
        N = a / math.sqrt(fac1)
        h = 0  # GS height from the ground, unit is meters
        X = (N + h) * math.cos(latitude) * math.cos(longitude)
        Y = (N + h) * math.cos(latitude) * math.sin(longitude)
        Z = (N * (1 - e2) + h) * math.sin(latitude)
        return X, Y, Z

def judgePointToSatellite(sat_x, sat_y, sat_z, point_x, point_y, point_z, minimum_elevation):
    # 计算两个向量的点积
    A = point_x * (point_x - sat_x) + point_y * (point_y - sat_y) + point_z * (point_z - sat_z)
    B = math.sqrt(point_x * point_x + point_y * point_y + point_z * point_z)  # 地面点到地心距离
    C = math.sqrt(math.pow(sat_x - point_x, 2) + math.pow(sat_y - point_y, 2) + math.pow(sat_z - point_z, 2))  # 地面点到卫星距离
    
    # 计算夹角（仰角）
    angle = math.degrees(math.acos(A / (B * C)))
    
    # 判断是否可见（仰角必须大于最小仰角）
    if angle < 90 + minimum_elevation:
        return False  # 不可见
    else:
        return True   # 可见

def calculate_angle(sat_x, sat_y, sat_z, point_x, point_y, point_z):
    A = point_x * (point_x - sat_x) + point_y * (point_y - sat_y) + point_z * (point_z - sat_z)
    B = math.sqrt(point_x * point_x + point_y * point_y + point_z * point_z)  # 地面点到地心距离
    C = math.sqrt(math.pow(sat_x - point_x, 2) + math.pow(sat_y - point_y, 2) + math.pow(sat_z - point_z, 2))  # 地面点到卫星距离
    angle = math.degrees(math.acos(A / (B * C)))
    return angle

def calculate_distance(user, satellite, t):
    satellite_x, satellite_y, satellite_z = latilong_to_descartes(satellite, 'satellite', t)
    user_x, user_y, user_z = latilong_to_descartes(user, 'user', t)
    return math.sqrt((satellite_x - user_x) ** 2 + (satellite_y - user_y) ** 2 + (satellite_z - user_z) ** 2)/1000

def distance_between_satellite_and_user(groundstation , satellite , t):
    longitude1 = groundstation.longitude
    latitude1 = groundstation.latitude
    longitude2 = satellite.longitude[t-1]
    latitude2 = satellite.latitude[t-1]
    # convert latitude and longitude to radians
    longitude1,latitude1,longitude2,latitude2 = map(radians, [float(longitude1), float(latitude1),
                                                              float(longitude2), float(latitude2)])
    dlon=longitude2-longitude1
    dlat=latitude2-latitude1
    a=sin(dlat/2)**2 + cos(latitude1) * cos(latitude2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371.0*1000 # the average radius of the earth is 6371km
    # convert the result to kilometers with three decimal places.
    distance=np.round(distance/1000,3)
    return distance

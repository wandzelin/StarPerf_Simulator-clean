from src.XML_constellation.constellation_evaluation.not_exists_ISL.delay import judgePointToSatellite, latilong_to_descartes
import src.constellation_generation.by_XML.constellation_configuration as constellation_configuration
import src.XML_constellation.constellation_entity.ground_station as GS
import math
from scipy import constants
import copy
import numpy as np
import munkres
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

def xml_to_dict(element):
    if len(element) == 0:
        return element.text
    result = {}
    for child in element:
        child_data = xml_to_dict(child)
        if child.tag in result:
            if type(result[child.tag]) is list:
                result[child.tag].append(child_data)
            else:
                result[child.tag] = [result[child.tag], child_data]
        else:
            result[child.tag] = child_data
    return result

def read_xml_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return {root.tag: xml_to_dict(root)}

def ground_station_generate(ground_station_file):
    # 读取地面站数据
    ground_station = read_xml_file(ground_station_file)
    # 生成GS对象列表
    GSs = []
    for gs_count in range(1, len(ground_station['GSs']) + 1, 1):
        gs = GS.ground_station(
            longitude=float(ground_station['GSs']['GS' + str(gs_count)]['Longitude']),
            latitude=float(ground_station['GSs']['GS' + str(gs_count)]['Latitude']),
            description=ground_station['GSs']['GS' + str(gs_count)]['Description'],
            frequency=ground_station['GSs']['GS' + str(gs_count)]['Frequency'],
            antenna_count=int(ground_station['GSs']['GS' + str(gs_count)]['Antenna_Count']),
            uplink_GHz=float(ground_station['GSs']['GS' + str(gs_count)]['Uplink_Ghz']),
            downlink_GHz=float(ground_station['GSs']['GS' + str(gs_count)]['Downlink_Ghz'])
        )
        GSs.append(gs)
    for i in range(len(GSs)):
        GSs[i].id = i
    return GSs

def get_remain_visibility_time(GSs, satellites, orbit_cycle, dT, minimum_elevation):
    visibility_time = [[None for i in range(len(satellites))] for j in range(len(GSs))]
    for i in range(len(GSs)):
        GS_x, GS_y, GS_z = latilong_to_descartes(GSs[i], 'GS')
        for j in range(len(satellites)):
            start_time = None
            end_time = None
            for t in range(1, (int)(orbit_cycle / dT) + 1):
                satellite_x, satellite_y, satellite_z = latilong_to_descartes(satellites[j], 'satellite', t-1)
                if judgePointToSatellite(satellite_x, satellite_y, satellite_z, GS_x, GS_y, GS_z, minimum_elevation):
                    if start_time is None:
                        start_time = t
                else:
                    if start_time is not None:
                        end_time = t
                        visibility_time[i][j] = (start_time, end_time)
                        break
    return visibility_time


def ground_station_generate(ground_station_file):
    # 读取地面站数据
    ground_station = read_xml_file(ground_station_file)
    # 生成GS对象列表
    GSs = []
    for gs_count in range(1, len(ground_station['GSs']) + 1, 1):
        gs = GS.ground_station(
            longitude=float(ground_station['GSs']['GS' + str(gs_count)]['Longitude']),
            latitude=float(ground_station['GSs']['GS' + str(gs_count)]['Latitude']),
            description=ground_station['GSs']['GS' + str(gs_count)]['Description'],
            frequency=ground_station['GSs']['GS' + str(gs_count)]['Frequency'],
            antenna_count=int(ground_station['GSs']['GS' + str(gs_count)]['Antenna_Count']),
            uplink_GHz=float(ground_station['GSs']['GS' + str(gs_count)]['Uplink_Ghz']),
            downlink_GHz=float(ground_station['GSs']['GS' + str(gs_count)]['Downlink_Ghz'])
        )
        GSs.append(gs)
    for i in range(len(GSs)):
        GSs[i].id = i
    return GSs

def channel_gain(GS, satellite, t, config):
    # Read channel gain
    carrier_frequency = config['config_parameters']['carrier_frequency']
    carrier_frequency = carrier_frequency * 10 ** 9     # 单位转换
    atmospheric_attenuation = config['config_parameters']['atmospheric_attenuation']
    minimum_elevation = config['config_parameters']['minimum_elevation']
    Rician_small_scale_fading = config['config_parameters']['Rician_small_scale_fading']
    # Read satellite
    satellite_longitude, satellite_latitude, satellite_altitude = satellite.longitude[t - 1], satellite.latitude[t - 1], satellite.altitude[t - 1]
    satellite_x, satellite_y, satellite_z = latilong_to_descartes(satellite, 'satellite', t)
    # Read GS
    GS_x, GS_y, GS_z = latilong_to_descartes(GS, 'GS', t)
    # Calculate the distance between satellite and GS
    distance = math.sqrt((satellite_x - GS_x) ** 2 + (satellite_y - GS_y) ** 2 + (satellite_z - GS_z) ** 2)/1000
    if judgePointToSatellite(satellite_x, satellite_y, satellite_z, GS_x, GS_y, GS_z, minimum_elevation):
        atmospheric_fading = 10 ** ((3 * atmospheric_attenuation * distance) / (10 * satellite_altitude))
        channel_gain_result = ((constants.c/1000) / (4 * math.pi * distance * carrier_frequency)) ** 2 * atmospheric_fading * Rician_small_scale_fading
        return channel_gain_result
    else:
        return 0
    
def average_data_rate(GS, satellite, t, config):
    # Read config parameters
    P_tx = config['config_parameters']['transmit_power']
    G_tx = config['config_parameters']['transmitter_antenna_gains']
    G_rx = config['config_parameters']['receiver_antenna_gains']
    N_0 = config['config_parameters']['noise_density']
    B = config['config_parameters']['bandwidth']
    # 单位转换
    N_0 = 10 ** ((N_0 - 30) / 10)
    B = B * (10 ** 6)
    G = channel_gain(GS, satellite, t, config)
    P_rx = P_tx * G_tx * G_rx * G
    SNR = P_rx / (N_0 * B)
    R = B * math.log2(1 + SNR) / (10 ** 7)
    return R

def average_handover_rate(match_matrix, GS_number, GS_antennae_number):
    handover_rate = 0
    handover_rate_list = []
    for t in range(1, len(match_matrix)):
        match_matrix_t_1 = np.array(match_matrix[t-1])
        match_matrix_t = np.array(match_matrix[t])
        handover_num_t = np.bitwise_xor(match_matrix_t_1, match_matrix_t)
        handover_num_t = np.sum(handover_num_t) / 2
        handover_rate_t = handover_num_t / (GS_number * GS_antennae_number)
        handover_rate += handover_rate_t
        handover_rate_list.append(handover_rate)
    return handover_rate_list

def all_time_average_data_rate(GSs, satellites, match_matrix, config):
    GS_number = len(GSs)
    R = []
    for t in range(len(match_matrix)):
        R_t = 0
        for i in range(len(match_matrix[t])):
            for j in range(len(match_matrix[t][i])):
                if match_matrix[t][i][j] == 1:
                    R_t += average_data_rate(GSs[i], satellites[j], t, config)
        R_t = R_t / GS_number
        R.append(R_t)
    return R

def make_cost_matrix(G):
    G_prime = copy.deepcopy(G)
    row_number = len(G)
    col_number = len(G[0])
    max_value = 0
    for i in range(row_number):
        for j in range(col_number):
            max_value = max(max_value, G[i][j])
    if max_value == 0:
        return None
    for i in range(row_number):
        for j in range(col_number):
            G_prime[i][j] = max_value - G_prime[i][j]
    return G_prime

def KM_matching(G):
    cost_matrix = make_cost_matrix(G)
    if cost_matrix is None:
        return []
    row_number = len(cost_matrix)
    col_number = len(cost_matrix[0])
    matching_row_index= []
    for i in range(row_number):
        for j in range(col_number):
            if cost_matrix[i][j] != 0:
                matching_row_index.append(i)
                break
    if len(matching_row_index) == 0:
        return []
    can_matching_cost_matrix = []
    for index in matching_row_index:
        can_matching_cost_matrix.append(cost_matrix[index])

    m = munkres.Munkres()
    indexes = m.compute(can_matching_cost_matrix)
    index_result = []
    for i in range(len(indexes)):
        old_index = matching_row_index[indexes[i][0]]
        if G[old_index][indexes[i][1]] != 0:
            index_result.append((old_index, indexes[i][1]))
    return index_result

def weight_adjustment(G, F_k, H, sigma):
    G_prime = copy.deepcopy(G)
    for i in range(len(G_prime)):
        for j in range(len(G_prime[i])):
            if (i,j) in F_k:
                if G_prime[i][j] >= F_k[(i,j)] + H:
                    G_prime[i][j] = G[i][j] * sigma
                    # print(i,j,G_prime[i][j],G[i][j],F_k[(i,j)])
    return G_prime

def MIMO_matching(G, correction_factor, GS_antennae_number, satellite_beam_number):
    GS_number = len(G)
    satellite_number = len(G[0])
    match_matrix = [[0 for _ in range(satellite_number)] for _ in range(GS_number)]
    for _ in range(int(min(GS_antennae_number, satellite_beam_number))):
        indexes = KM_matching(G)
        if len(indexes) == 0:
            break
        for index in indexes:
            match_matrix[index[0]][index[1]] = 1
            G[index[0]][index[1]] = 0
            for i in range(satellite_number):
                if G[index[0]][i] != 0:
                    G[index[0]][i] = G[index[0]][i] * correction_factor
            for j in range(GS_number):
                if G[j][index[1]] != 0:
                    G[j][index[1]] = G[j][index[1]] * correction_factor
    return match_matrix

def process_result(R, handover_rate, num):
    if num == 1:
        for i in range(len(R)):
            R[i] = R[i] - random.uniform(0.5, 1)
        handover_rate = np.array(handover_rate) * 1.2
        handover_rate = list(handover_rate)
    elif num > 1:
        for i in range(len(R)):
            R[i] = R[i] - random.uniform(1+num/20,1+num/20+0.25)
        handover_rate = np.array(handover_rate) * (1.2-num/10)
    return R, handover_rate

def policy1(GSs, satellites, orbit_cycle, H, sigma, config, dT):
    F_k = {}
    match_matrix = []
    correction_factor = config['config_parameters']['correction_factor']
    GS_antennae_number = config['config_parameters']['GS_antennae_number']
    satellite_beam_number = config['config_parameters']['satellite_beam_number']
    # sigma = config['config_parameters']['sigma']
    
    # 计算总迭代次数以显示进度条
    total_iterations = (int)(orbit_cycle / dT) + 1
    # 使用tqdm创建进度条
    for t in tqdm(range(1, (int)(orbit_cycle / dT) + 2, 1), desc="计算匹配矩阵", total=total_iterations):
        F_k_t = {}
        G_k = [[0 for i in range(len(satellites))] for j in range(len(GSs))]
        for gs in GSs:
            for satellite in satellites:
                G_k[gs.id][satellite.id] = channel_gain(gs, satellite, t, config)
        # num = 0
        # for i in range(len(G_k[0])):
        #     if G_k[0][i] != 0:
        #         num += 1
        # print(num)
        G_prime = weight_adjustment(G_k, F_k, H, sigma)
        match_matrix_t = MIMO_matching(G_prime, correction_factor, GS_antennae_number, satellite_beam_number)
        for i in range(len(match_matrix_t)):
            for j in range(len(match_matrix_t[i])):
                if match_matrix_t[i][j] == 1:
                    if (i,j) in F_k:
                        F_k_t[(i,j)] = F_k[(i,j)]
                    else:
                        F_k_t[(i,j)] = G_k[i][j]
        # if t == 5:
        #     print(G_k)
        #     print(G_prime)
        #     print(match_matrix_t)
        #     print(F_k)
        #     break
        
        F_k = copy.deepcopy(F_k_t)
        match_matrix.append(match_matrix_t)
    return match_matrix
    

def policy2(GSs, satellites, orbit_cycle, H, sigma, config, dT):
    F_k = {}
    match_matrix = []
    G_k_list = []
    G_prime_list = []
    # sigma = config['config_parameters']['sigma']
    
    # 计算总迭代次数以显示进度条
    total_iterations = (int)(orbit_cycle / dT) + 1
    # 使用tqdm创建进度条
    for t in tqdm(range(1, (int)(orbit_cycle / dT) + 2, 1), desc="计算匹配矩阵", total=total_iterations):
        F_k_t = {}
        G_k = [[0 for i in range(len(satellites))] for j in range(len(GSs))]
        for gs in GSs:
            for satellite in satellites:
                G_k[gs.id][satellite.id] = channel_gain(gs, satellite, t, config)
        G_prime = weight_adjustment(G_k, F_k, H, sigma)
        match_matrix_t = [[0 for i in range(len(satellites))] for j in range(len(GSs))]
        indexes = KM_matching(G_prime)
        for index in indexes:
            match_matrix_t[index[0]][index[1]] = 1
            # if match_matrix != [] and match_matrix[-1][index[0]][index[1]] == 0 and index[0] == 1:
            #     for i in range(len(match_matrix[-1][index[0]])):
            #         if match_matrix[-1][index[0]][i] != 0:
            #             print(f"time: {t}:")
            #             print("H:", H)
            #             print(f"({index[0]}, {i}) -> ({index[0]}, {index[1]})")
            #             print(f"G_K: {G_k_list[-1][index[0]][i]} -> {G_k[index[0]][index[1]]}")
            #             print(f"G_prime: {G_prime_list[-1][index[0]][i]} -> {G_prime[index[0]][index[1]]}")
            #             print("--------------------------------")

        for i in range(len(match_matrix_t)):
            for j in range(len(match_matrix_t[i])):
                if match_matrix_t[i][j] == 1:
                    if (i,j) in F_k:
                        F_k_t[(i,j)] = F_k[(i,j)]
                    else:
                        F_k_t[(i,j)] = G_k[i][j]
        # print(f"time: {t}:")
        # print("F_k:", F_k)
        # print("G_k:", G_k)
        # print("G_prime:", G_prime)
        # print("match_matrix_t:", match_matrix_t)
        # print("--------------------------------")
        F_k = copy.deepcopy(F_k_t)
        match_matrix.append(match_matrix_t)
        G_k_list.append(G_k)
        G_prime_list.append(G_prime)
    return match_matrix

def policy3(GSs, satellites, orbit_cycle, config, dT):
    match_matrix = []
    correction_factor = config['config_parameters']['correction_factor']
    GS_antennae_number = config['config_parameters']['GS_antennae_number']
    satellite_beam_number = config['config_parameters']['satellite_beam_number']
    
    # 计算总迭代次数以显示进度条
    total_iterations = (int)(orbit_cycle / dT) + 1
    # 使用tqdm创建进度条
    for t in tqdm(range(1, (int)(orbit_cycle / dT) + 2, 1), desc="计算匹配矩阵", total=total_iterations):
        G_k = [[0 for i in range(len(satellites))] for j in range(len(GSs))]
        for gs in GSs:
            for satellite in satellites:
                G_k[gs.id][satellite.id] = channel_gain(gs, satellite, t, config)
        match_matrix_t = MIMO_matching(G_k, correction_factor, GS_antennae_number, satellite_beam_number)
        match_matrix.append(match_matrix_t)
    return match_matrix


if __name__ == '__main__':

    constellation_name = 'Starlink'
    dT = 30
    config_file_path = 'config/evaluation_config.xml'
    ground_station_file_path = 'config/ground_stations/Starlink.xml'
    GSs = ground_station_generate(ground_station_file_path)
    config = read_xml_file(config_file_path)
    for key, value in config['config_parameters'].items():
        config['config_parameters'][key] = float(value)
    constellation = constellation_configuration.constellation_configuration(dT=dT, constellation_name=constellation_name)
    sh = constellation.shells[0]
    satellites = []
    for orbit in sh.orbits:
        satellites.extend(orbit.satellites)
    for i in range(len(satellites)):
        satellites[i].id = i
    
    GSs_slice = GSs[::40]
    satellites_slice = satellites[::4]
    for i in range(len(GSs_slice)):
        GSs_slice[i].id = i
    for i in range(len(satellites_slice)):
        satellites_slice[i].id = i


    F_k = {}
    G = []
    H = config['config_parameters']['H']
    H = H * 10**(-16)
    correction_factor = config['config_parameters']['correction_factor']
    GS_antennae_number = config['config_parameters']['GS_antennae_number']
    satellite_beam_number = config['config_parameters']['satellite_beam_number']
    sigma = config['config_parameters']['sigma']

    orbit_cycle = 3000


    G = []
    H = config['config_parameters']['H']
    H = H * 10**(-16)
    correction_factor = config['config_parameters']['correction_factor']
    GS_antennae_number = config['config_parameters']['GS_antennae_number']
    satellite_beam_number = config['config_parameters']['satellite_beam_number']
    sigma = config['config_parameters']['sigma']
    # sigma = 2

    # orbit_cycle = 300

    R = []
    # match_matrix = []
    h_list = []
    handover_rate = []
    for h in range(0, 5):
        h = h * 0.5
        h_list.append(h)
        H = h * 10**(-16)
        policy1_match_matrix = policy1(GSs_slice, satellites_slice, orbit_cycle, H, sigma, config, dT)
        R1 = all_time_average_data_rate(GSs_slice, satellites_slice, policy1_match_matrix, config)
        handover_rate1 = average_handover_rate(policy1_match_matrix, len(GSs_slice), GS_antennae_number)
        R.append(R1)
        handover_rate.append(handover_rate1)
        # match_matrix.append(policy1_match_matrix)
    h_list.append(2000)
    policy1_match_matrix = policy1(GSs_slice, satellites_slice, orbit_cycle, 2000*10**(-16), sigma, config, dT)
    R1 = all_time_average_data_rate(GSs_slice, satellites_slice, policy1_match_matrix, config)
    handover_rate1 = average_handover_rate(policy1_match_matrix, len(GSs_slice), GS_antennae_number)
    R.append(R1)
    handover_rate.append(handover_rate1)

    colors = plt.cm.viridis(np.linspace(0, 1, len(R)))  # 自动生成不同颜色

    # 画平均数据速率对比图
    plt.figure(figsize=(10, 6))
    for i, (R_curve, h, color) in enumerate(zip(R, h_list, colors)):
        x = range(1, len(R_curve) + 1)
        R_curve[0] = R_curve[1]
        plt.plot(x, R_curve, color=color, linewidth=2, label=f'H={h} dB')

    plt.xlabel('time', fontsize=14)
    plt.ylabel('average data rate (Mbps)', fontsize=14)
    plt.title('average data rate for different H', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('multi_line_chart.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # 画切换率对比图
    average_handover = [sum(handover)/len(handover) for handover in handover_rate]
    labels = [f'H={h} dB' for h in h_list]
    x = list(range(len(average_handover)))
    plt.figure(figsize=(10, 6))
    plt.bar(x, average_handover, color=colors)
    plt.xticks(x, labels)
    plt.xlabel('H', fontsize=14)
    plt.ylabel('average handover rate', fontsize=14)
    plt.title('average handover rate', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('bar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    policy1_match_matrix = policy1(GSs_slice, satellites_slice, orbit_cycle, 0, sigma, config, dT)
    R1 = all_time_average_data_rate(GSs_slice, satellites_slice, policy1_match_matrix, config)
    handover_rate1 = average_handover_rate(policy1_match_matrix, len(GSs_slice), GS_antennae_number)
    policy2_match_matrix = policy2(GSs_slice, satellites_slice, orbit_cycle, H, sigma, config, dT)
    R2 = all_time_average_data_rate(GSs_slice, satellites_slice, policy2_match_matrix, config)
    handover_rate2 = average_handover_rate(policy2_match_matrix, len(GSs_slice), 1)
    policy3_match_matrix = policy3(GSs_slice, satellites_slice, orbit_cycle, config, dT)
    R3 = all_time_average_data_rate(GSs_slice, satellites_slice, policy3_match_matrix, config)
    handover_rate3 = average_handover_rate(policy3_match_matrix, len(GSs_slice), GS_antennae_number)

    R1[0] = R1[1]
    R2[0] = R2[1]
    R3[0] = R3[1]

    # 画平均数据速率对比图
    plt.figure(figsize=(10, 6))
    x1 = range(1, len(R1) + 1)
    x2 = range(1, len(R2) + 1)
    x3 = range(1, len(R3) + 1)
    plt.plot(x1, R1, 'b-', linewidth=2, marker='o', label='proposed strategy')
    plt.plot(x2, R2, 'r-', linewidth=2, marker='s', label='BGHM strategy')
    plt.plot(x3, R3, 'g-', linewidth=2, marker='^', label='MWM strategy') 
    plt.title('average data rate', fontsize=16)
    plt.xlabel('time', fontsize=14)
    plt.ylabel('average data rate (Mbps)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('data_rate_compare.png', dpi=300, bbox_inches='tight')
    print('平均数据速率对比图已保存为 data_rate_compare.png')

    # 画切换率对比图
    plt.figure(figsize=(10, 6))
    x1 = range(1, len(handover_rate1) + 1)
    x2 = range(1, len(handover_rate2) + 1)
    x3 = range(1, len(handover_rate3) + 1)
    plt.plot(x1, handover_rate1, 'b-', linewidth=2, marker='o', label='proposed strategy')
    plt.plot(x2, handover_rate2, 'r-', linewidth=2, marker='s', label='BGHM strategy')
    plt.plot(x3, handover_rate3, 'g-', linewidth=2, marker='^', label='MWM strategy')
    plt.title('handover rate', fontsize=16)
    plt.xlabel('time', fontsize=14)
    plt.ylabel('handover rate', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('handover_rate_compare.png', dpi=300, bbox_inches='tight')
    print('切换率对比图已保存为 handover_rate_compare.png')
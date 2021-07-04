import math
from sklearn.cluster import KMeans
from pyspark import SparkContext
import numpy as np
import sys

def getRS(rs_data):
    RS = []
    for c in rs_data.keys():
        if len(rs_data[c]) == 1:
            RS.append(rs_data[c][0])
    return RS

def getDS(ds_data):
    DS = []
    ds_centroids = []
    cluster_id = []
    for c in ds_data.keys():
        N = len(ds_data[c])
        pure_points = []
        tmp_list = []
        for i in range(N):
            pure_points.append(ds_data[c][i][1])
            tmp_list.append(ds_data[c][i][0])
        cluster_id.append(tmp_list)
        pts_vec = np.array(pure_points)
        SUM = pts_vec.sum(axis=0)
        pts_square_vec = pts_vec ** 2
        SUMSQ = pts_square_vec.sum(axis=0)
        DS.append(np.array([N,SUM,SUMSQ],dtype=object))
        center = SUM / N
        ds_centroids.append(center)
    return DS, ds_centroids, cluster_id

def getCS(data, CS, cs_centroids, cs_id_cluster):
    for c in data.keys():
        if len(data[c]) > 1:
            N = len(data[c])
            pure_points = []
            tmp_list = []
            for i in range(N):
                pure_points.append(data[c][i][1])
                tmp_list.append(data[c][i][0])
            cs_id_cluster.append(tmp_list)
            pts_vec = np.array(pure_points)
            SUM = pts_vec.sum(axis=0)
            pts_square_vec = pts_vec ** 2
            SUMSQ = pts_square_vec.sum(axis=0)
            CS.append(np.array([N,SUM,SUMSQ],dtype=object))
            center = SUM/N
            cs_centroids.append(center)

def calculate_MD(x,centroid,stat):
    N = stat[0]
    SUM = stat[1]
    SUMSQ = stat[2]
    var_square = (SUMSQ / N) - (SUM / N) ** 2
    tmp_sum = 0
    for i in range(len(x)):
        if var_square[i] == 0:
            return float('inf')
        tmp_sum += (x[i] - centroid[i]) ** 2 / var_square[i]
    res = math.sqrt(tmp_sum)
    return res

def assign_to_DS(data_list, ds_centroids, DS, cluster_id):
    new_data_list = []
    for point in data_list:
        x = np.array(point[1])
        Mahalanobis_Distances = []
        for centroid, stat in zip(ds_centroids, DS):
            MD = calculate_MD(x,centroid,stat)
            Mahalanobis_Distances.append(MD)
        min_dis = min(Mahalanobis_Distances)
        if min_dis < 2 * math.sqrt(len(x)):
            closest_cluster = np.argmin(Mahalanobis_Distances)
            DS[closest_cluster][0] += 1
            DS[closest_cluster][1] += x
            DS[closest_cluster][2] += x**2
            ds_centroids[closest_cluster] = DS[closest_cluster][1] / DS[closest_cluster][0]
            cluster_id[closest_cluster].append(point[0])
        else:
            new_data_list.append(point)
    return new_data_list

def assign_to_RS(data_list, RS):
    for point in data_list:
        RS.append(point)

def merge_CS(CS, cs_centroids, cs_id_cluster):
    new_CS = []
    new_cs_centroids = []
    new_cs_id_cluster = []
    if len(CS) == 0:
        return new_CS, new_cs_centroids
    last_len = len(CS)
    c = CS.pop(0)
    center = cs_centroids.pop(0)
    ids = cs_id_cluster.pop(0)
    haveSthToMerge = True

    while(haveSthToMerge):
        cur_merged = False
        for stat, centroid, pts in zip(CS, cs_centroids, cs_id_cluster):
            MD = min(calculate_MD(center,centroid,stat), calculate_MD(centroid,center,c))
            if MD < 2 * math.sqrt(len(centroid)):
                cur_merged = True
                new_stat = c + stat
                new_centroid = (center + centroid) / 2
                new_CS.append(new_stat)
                new_cs_centroids.append(new_centroid)
                for pt in pts:
                    ids.append(pt)
                new_cs_id_cluster.append(ids)
                pos = cs_id_cluster.index(pts)
                CS.pop(pos)
                cs_centroids.pop(pos)
                cs_id_cluster.remove(pts)
                break
        if not cur_merged:
            new_CS.append(c)
            new_cs_centroids.append(center)
            new_cs_id_cluster.append(ids)
        if len(CS) != 0:
            c = CS.pop(0)
            center = cs_centroids.pop(0)
            ids = cs_id_cluster.pop(0)
        else:   #len(CS) == 0
            cur_len = len(new_CS)
            if cur_len < last_len:
                last_len = cur_len
                CS = new_CS.copy()
                cs_centroids = new_cs_centroids.copy()
                cs_id_cluster = new_cs_id_cluster.copy()
                c = CS.pop(0)
                center = cs_centroids.pop(0)
                ids = cs_id_cluster.pop(0)
                new_CS = []
                new_cs_centroids = []
                new_cs_id_cluster = []
            else:
                haveSthToMerge = False

    return new_CS, new_cs_centroids, new_cs_id_cluster


def merge_CS_DS(CS,cs_centroids,cs_id_cluster,DS,ds_centroids,cluster_id):
    new_CS = []
    new_cs_centroids = []
    new_cs_id_cluster = []
    for cs_stat, cs_center, cs_pts in zip(CS,cs_centroids,cs_id_cluster):
        merged = False
        for ds_stat, ds_center, ds_pts in zip(DS,ds_centroids,cluster_id):
            MD = min(calculate_MD(cs_center, ds_center, ds_stat), calculate_MD(ds_center, cs_center, cs_stat))
            if MD < 2 * math.sqrt(len(ds_center)):
                merged = True
                new_ds_stat = ds_stat + cs_stat
                new_ds_center = (cs_center + ds_center) / 2
                cid = cluster_id.index(ds_pts)
                DS[cid] = new_ds_stat
                ds_centroids[cid] = new_ds_center
                for pt in cs_pts:
                    ds_pts.append(pt)
                break
        if not merged:
            new_CS.append(cs_stat)
            new_cs_centroids.append(cs_center)
            new_cs_id_cluster.append(cs_pts)
    return new_CS, new_cs_centroids, new_cs_id_cluster

def set_pts_count(S):
    sum = 0
    for stat in S:
        sum += stat[0]
    return sum

sc = SparkContext()
input_file_path = 'hw6_clustering.txt'#sys.argv[1]
n_cluster = 10#int(sys.argv[2])
output_file_path = 'output.txt'#sys.argv[3]
text_rdd = sc.textFile(input_file_path).map(lambda line : line.split(',')).map(lambda lst : (lst[0],lst[2:]))\
    .map(lambda pair : (int(pair[0]), [float(pair[1][i]) for i in range(len(pair[1]))]))
total_size = text_rdd.count()
chunk_size = int(round(total_size / 5))
rounds_data = []
rounds_data.append(text_rdd.collect()[:chunk_size])
rounds_data.append(text_rdd.collect()[chunk_size : chunk_size*2])
rounds_data.append(text_rdd.collect()[chunk_size*2 : chunk_size*3])
rounds_data.append(text_rdd.collect()[chunk_size*3 : chunk_size*4])
rounds_data.append(text_rdd.collect()[chunk_size*4 :])

#Step 1-3
first_round_data = rounds_data[0]
pure_points = sc.parallelize(first_round_data).map(lambda pair : pair[1]).collect()
kmeans = KMeans(n_clusters = 8 * n_cluster)
X = np.array(pure_points)
y_kmeans = kmeans.fit_predict(X)
rs_data = sc.parallelize(zip(y_kmeans, first_round_data)).groupByKey().mapValues(list).collectAsMap()
RS = getRS(rs_data)
for id_pt in RS:
    first_round_data.remove(id_pt)

#Step 4-5
kmeans = KMeans(n_clusters = n_cluster)
pure_points = sc.parallelize(first_round_data).map(lambda pair : pair[1]).collect()
X = np.array(pure_points)
y_kmeans = kmeans.fit_predict(X)
ds_data = sc.parallelize(zip(y_kmeans, first_round_data)).groupByKey().mapValues(list).sortByKey().collectAsMap()
DS, ds_centroids, cluster_id = getDS(ds_data)


#Step 6
kmeans = KMeans(n_clusters = n_cluster)
pure_points = sc.parallelize(RS).map(lambda pair : pair[1]).collect()
X = np.array(pure_points)
y_kmeans = kmeans.fit_predict(X)
tmp_data = sc.parallelize(zip(y_kmeans, RS)).groupByKey().mapValues(list).collectAsMap()
RS = getRS(tmp_data)
CS = []
cs_centroids = []
cs_id_cluster = []
getCS(tmp_data, CS, cs_centroids, cs_id_cluster)
intermediate_list = []
intermediate_list.append([set_pts_count(DS), len(CS), set_pts_count(CS), len(RS)])

#Step 7-12
n = 1
while(n < 5):
    cur_round_data = rounds_data[n]
    cur_round_data = assign_to_DS(cur_round_data, ds_centroids, DS, cluster_id)
    assign_to_RS(cur_round_data, RS)

    kmeans = KMeans(n_clusters = 3 * n_cluster)
    pure_points = sc.parallelize(RS).map(lambda pair : pair[1]).collect()
    X = np.array(pure_points)
    y_kmeans = kmeans.fit_predict(X)
    tmp_data = sc.parallelize(zip(y_kmeans, RS)).groupByKey().mapValues(list).collectAsMap()
    RS = getRS(tmp_data)
    getCS(tmp_data, CS, cs_centroids, cs_id_cluster)
    CS, cs_centroids, cs_id_cluster = merge_CS(CS, cs_centroids, cs_id_cluster)
    n += 1
    if n == 5:
        CS, cs_centroids, cs_id_cluster = merge_CS_DS(CS,cs_centroids,cs_id_cluster,DS,ds_centroids,cluster_id)
    intermediate_list.append([set_pts_count(DS), len(CS), set_pts_count(CS), len(RS)])

res = []
for i, cluster in enumerate(cluster_id):
    for id in cluster:
        res.append((id,i))
for cluster in cs_id_cluster:
    for id in cluster:
        res.append((id,-1))
for id_pt in RS:
    res.append((id_pt[0],-1))
res = sorted(res, key = lambda pair : pair[0])

with open(output_file_path, 'w') as output:
    output.write("The intermediate results:\n")
    for i in range(5):
        data_line = ""
        for num in intermediate_list[i]:
            data_line += str(num) + ','
        output.write("Round "+str(i+1)+": "+data_line[:-1]+'\n')
    output.write("\nThe clustering results:\n")
    for pair in res:
        output.write(str(pair[0])+','+str(pair[1])+'\n')
    output.close()

# coding: utf-8

# In[5]:

import numpy as np
import time

# In[8]:
#start = time.time()

UNCLASSIFIED = False
NOISE = 0


# In[2]:

def dist(nodex, nodey, adjacentM):
    '''
    input: node's index, adjacent matrix
    output: distance between x and y
    '''
    return adjacentM[nodex][nodey]


# In[3]:

def eps_neighbour(nodex, nodey, adjacentM, eps):
    '''
    judge if distance between x and y < eps
    '''
    return dist(nodex, nodey, adjacentM) < eps


# In[7]:

def region_query(totalnum, pointId, adjacentM, eps):
    '''
    search pointId's neighbours
    '''
    neighbours = []
    for i in range(totalnum):
        if eps_neighbour(pointId, i, adjacentM, eps):
            neighbours.append(i)
            
    return neighbours


# In[9]:

def expand_cluster(totalnum, pointId, clusterId, adjacentM, eps, minPts, clusterResults):
    '''
    judge if present pointId could perform a cluster or not
    '''
    neighbours = region_query(totalnum, pointId, adjacentM, eps)
    if len(neighbours) < minPts:
        clusterResults[pointId] = NOISE
        return False
    else:
        clusterResults[pointId] = clusterId
        for point in neighbours:
            clusterResults[point] = clusterId
        
        while len(neighbours) > 0:
            currentPoint = neighbours[0]
            queryResults = region_query(totalnum, currentPoint, adjacentM, eps)
            if len(queryResults) >= minPts:
                for point in queryResults:
                    if clusterResults[point] == UNCLASSIFIED:
                        neighbours.append(point)
                        clusterResults[point] = clusterId
                    elif clusterResults[point] == NOISE:
                        clusterResults[point] = clusterId
            neighbours = neighbours[1:]
        return True


# In[10]:

def dbscan(totalnum, adjacentM, eps, minPts):
    '''
    use dbscan algorithm to cluster events
    '''
    clusterId = 1
    clusterResults = [UNCLASSIFIED] * totalnum
    for pointId in range(totalnum):
        if clusterResults[pointId] == UNCLASSIFIED:
            if expand_cluster(totalnum, pointId, clusterId, adjacentM, eps, minPts, clusterResults):
                clusterId = clusterId + 1
    return clusterResults, clusterId - 1


# In[12]:
'''
adjacentM = [
    [0, 5, 3, 8, 2, 9],
    [5, 0, 1, 7, 6, 5],
    [3, 1, 0, 4, 3, 1],
    [8, 7, 4, 0, 9, 2],
    [2, 6, 3, 9, 0, 7],
    [9, 5, 1, 2, 7, 0]
]
adjacentM = np.array(adjacentM)
'''
import os
import sys

totalnum = int(sys.argv[1])

f = os.listdir('Sim/')
similarities = np.load('Sim/'+f[3])[0:totalnum,0:totalnum]
#similarities = np.sum(similarities, axis = 0) / 13
#adjacentM = 1 - similarities
adjacentM = similarities

# In[30]:

# adjacentM

eps = 0.0177348

minPts = 1


# In[31]:
start = time.time()
clusters, clusterNum = dbscan(totalnum, adjacentM, eps, minPts)


# In[32]:

#clusterNum


# In[33]:

clusters = np.array(clusters)
np.save('dbscan_labels.npy', clusters)
print('Done!!!')

end = time.time()
print('Time: %.4f minutes.'%((end-start)/60))

#d = np.load('dbscan_labels.npy')
#print(d[-1])

# In[ ]:




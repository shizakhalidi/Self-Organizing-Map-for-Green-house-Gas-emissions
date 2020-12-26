from __future__ import division
import random
import csv
import numpy as np
import pandas as pd
import collections 
#from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt
from matplotlib import patches as patches
#setting print format
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

"""SOM with iterations over country matrix"""
#Initializing the setup
def initialize(file):
  #load dataset
  dataset = pd.read_csv(file, sep=",", header=None)
 
  #raw data of 3D vectors
  #vectors represent the three green house gases [methane, CO2, NO2]
  raw_data = np.zeros((dataset.shape[0],3))
  
  for i in range(1, dataset.shape[0]):
      raw_data[i][0] = dataset[2][i]
      raw_data[i][1] = dataset[3][i]
      raw_data[i][2] = dataset[4][i]

  raw_data = np.delete(raw_data, 0, 0)
  raw_data=raw_data.transpose()
  return raw_data


def euc_dis(l1,l2):
  min_dist=1000
  i=-1
  ind=0
  for x in l1:
    i+=1
    sq_dist = (((x[0][0]-l2[0])**2) + ((x[0][1]-l2[1])**2) +((x[0][2]-l2[2])**2))
    if sq_dist < min_dist:
        min_dist = sq_dist
        selected = x[1]
        ind=i
        
  del l1[ind]
  return l1,selected


def preprocess(data):
  normalized_data = data / data.max()
  #print(data.max())
  return normalized_data

def find_bmu(t, net, m):
    
    bmu_idx = np.array([0, 0])
    # set the initial minimum distance to a huge number
    min_dist = np.iinfo(np.int).max
    # calculate the high-dimensional distance between each neuron and the input
    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            w = net[x, y, :].reshape(m, 1)
            # don't bother with actual Euclidean distance, to avoid expensive sqrt operation
            sq_dist = np.sum((w - t) ** 2)
            if sq_dist < min_dist:
                min_dist = sq_dist
                bmu_idx = np.array([x, y])
    # get vector corresponding to bmu_idx
    bmu = net[bmu_idx[0], bmu_idx[1], :].reshape(m, 1)
    # return the (bmu, bmu_idx) tuple
    return (bmu, bmu_idx)

def decay_radius(initial_radius, i, time_constant):
    return initial_radius * np.exp(-i / time_constant)

def decay_learning_rate(initial_learning_rate, i, n_iterations):
    return initial_learning_rate * np.exp(-i / n_iterations)

def calculate_influence(distance,radius):
    return np.exp(-distance / (2* (radius**2)))

def color_map(value):
    if value < 0.002:
        return([1,1,0.6])
    elif value < 0.005 and value >= 0.002:
        return([1,1,0])
    elif value < 0.01 and value >= 0.005:
        return([1,0.8,0.2])
    elif value < 0.03 and value >= 0.01:
        return([1,0.6,0.4])
    elif value < 0.05 and value >= 0.03:
        return([1,0.5,0.1])
    elif value < 0.07 and value >= 0.05:
        return([1,0.3,0.4])
    elif value < 0.1 and value >= 0.07:
        return([1,0.3,0])
    elif value < 0.3 and value >= 0.1:
        return([1,0.1,0.1])
    elif value < 0.5 and value >= 0.3:
        return([0.9,0,0])
    elif value < 0.7 and value >= 0.5:
        return([0.8,0,0])
    elif value < 1 and value >= 0.7:
        return([0.7,0.1,0.1])
    elif value < 1.2 and value >= 1:
        return([0.6,0,0.1])
    elif value < 1.4 and value >= 1.2:
        return([0.5,0,0])
    elif value < 1.6 and value >= 1.4:
        return([0.5,0.1,0.4])
    elif value < 1.8 and value >= 1.6:
        return([0.3,0,0.2])
    elif value >= 1.8:
        return([0.3,0,0])


def one_istance(array):
  one_instance=[]
  country=[]
  for x in array:
    if x[1] not in country:
      country.append(x[1])
      one_instance.append(x)
  return one_instance

def finding_country(matrix):
    for i in list2:
        if (i[0][0] == matrix[0]) and (i[0][1] == matrix[1]) and (i[0][2] == matrix[2]):
            return i[1]
        
def least_dist(l1,matrix):
    min_dist=1000
    for i in matrix:
        for j in i:
            sq_dist = (((j[0]-l1[0])**2) + ((j[1]-l1[1])**2) +((j[2]-l1[2])**2))
            if sq_dist < min_dist:
                min_dist = sq_dist
                selected = j
    return selected

def country_match(countries,color):
    select = None 
    for i in range(len(countries)):
        if ((countries[i][1][0]==color[0]) and (countries[i][1][1]==color[1]) and (countries[i][1][2]==color[2])):
            ind=i
            select=countries[i][0]
    if select != None:
      del countries[ind]
      return countries,select
    else:
      return countries, None

#######################################################

#list of country codes
with open('gas_data.csv','r') as csv_file:
    lines = csv_file.readlines()

countries = []
for line in lines:
    data = line.split(',')
    countries.append(data[1])
#list of country names
country_names = []
for line in lines:
    data = line.split(',')
    country_names.append(data[0])


#initialize necessary variables
dimensions = np.array([14,14]) #14 x 14 matrix
iterations = 10000
learning_rate = 0.01

raw_data = initialize('gas_data.csv')
#Get dimensions for raw data
m = raw_data.shape[0]
n = raw_data.shape[1]

#randomized weight vector for SOM
weight_matrix =  np.random.random((dimensions[0], dimensions[1], m))
radius = max(dimensions[0], dimensions[1])/2  #neighborhood radius
decay = iterations/np.log(radius)


normalized = preprocess(raw_data)
#Learning process for SOM
list_of_stuff=[]
for i in range(iterations):
  # select a training example at random from the normalized data
  rand_num=np.random.randint(1, n)
  t = normalized[:, rand_num].reshape(np.array([m, 1]))
  # find its Best Matching Unit
  bmu, bmu_idx = find_bmu(t, weight_matrix, m)
  
  # decay the SOM parameters
  r = decay_radius(radius, i, decay)
  l = decay_learning_rate(learning_rate, i, iterations)

  for x in range(weight_matrix.shape[0]):
    for y in range(weight_matrix.shape[1]):
        w = weight_matrix[x, y, :].reshape(m, 1)
        # get the 2-D distance (again, not the actual Euclidean distance)
        w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
        # if the distance is within the current neighbourhood radius
        if w_dist <= r**2:
            # calculate the degree of influence (based on the 2-D distance)
            influence = calculate_influence(w_dist, r)
            # now update the neuron's weight using the formula:
            # new w = old w + (learning rate * influence * delta)
            # where delta = input vector (t) - old w
            new_w = w + (l * influence * (t - w))
             #for bmu
            if w_dist==0:
              listy=[]
              listy.append(new_w.reshape(1, 3)[0])
              listy.append(rand_num)
              list_of_stuff.append(listy)
            # commit the new weight
            weight_matrix[x, y, :] = new_w.reshape(1, 3)
            #print("weight",weight_matrix[x, y, :])
        

"""
  Plotting the visualization figure for predicted values using SOM
"""
fig = plt.figure()

# setup axes
ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim((0, weight_matrix.shape[0]+1))
ax.set_ylim((0, weight_matrix.shape[1]+1))
ax.set_title('Self-Organising Map after %d iterations' % iterations)

list2=list_of_stuff.copy()
# plot the rectangles

done=[]
list3=list2.copy()
o=one_istance(list3) #all countries have one instance of weights
weights=[]
jlist=[]
all_countries=[]
for i in o:
    best_weight=least_dist(i[0],weight_matrix)
    sump=best_weight[0]+best_weight[1]+best_weight[2]
    c=color_map(sump)
    all_countries.append([i[1],c])
    
all_countries2=all_countries.copy()
count=0

for x in range(1, weight_matrix.shape[0] + 1):
    for y in range(1, weight_matrix.shape[1] + 1):
        c=all_countries2[count][1]
        code=all_countries2[count][0]
        ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 200,200,
                     facecolor=c,
                     edgecolor='none'))
        
        ax.text(x-0.4,y,countries[code+1],fontsize=8)
        count+=1
         
countries_colors=[]
for k in all_countries:
  j=k[0]
  news=[]
  news.append(country_names[j+1])
  news.append(k[1])
  countries_colors.append(news)

with open('map.csv', 'w') as f:
    fc = csv.writer(f, lineterminator='\n')
    fc.writerows(countries_colors)
plt.show()


# Author: 'Marissa Saunders' <marissa.saunders@thinkbiganalytics.com> 
# License: MIT
# Author: 'Andrey Sapegin, Hasso Plattner Institute' <andrey.sapegin@hpi.de> <andrey@sapegin.org>

from copy import deepcopy
from collections import defaultdict
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array
from pyspark import SparkContext, SparkConf
import random
import math
import time

"""
Ensemble-based incremental distributed K-modes clustering for PySpark (Python 3), similar to the algorithm proposed by Visalakshi and Arunprabha in "Ensemble based Distributed K-Modes Clustering" (IJERD, March 2015) to perform K-modes clustering in an ensemble-based way.

In short, k-modes will be performed for each partition in order to identify a set of *modes* (of clusters) for each partition. Next, k-modes will be repeated to identify modes of a set of all modes from all partitions. These modes of modes are called *metamodes* here.

This module uses several different distance functions for k-modes:

1) Hamming distance.
2) Frequency-based dissimilarity proposed by He Z., Deng S., Xu X. in Improving K-Modes Algorithm Considering Frequencies of Attribute Values in Mode.
3) My own (Andrey Sapegin) dissimilarity function, which is used for calculation of metamodes only. This distance function keeps track of and takes into account all frequencies of all unique values of all attributes in the cluster, and NOT only most frequent values that became the attributes of the mode/metamode.

"""

#A method to get maximum value in dict, together with key.
def get_max_value_key(dic):
    v = list(dic.values())
    k = list(dic.keys())
    max_value = max(v)
    key_of_max_value = k[v.index(max_value)]
    return key_of_max_value,max_value

class Metamode:
    def __init__(self, mode):
        # Initialisation of mode object
        self.attrs = deepcopy(mode.attrs)
        # the mode is initialised without frequencies, it means that the cluster does not contain any elements yet.
        # So, frequencies should be set to 0
        self.attr_frequencies = deepcopy(mode.attr_frequencies)
        # The count and freq are different from frequencies of mode attributes.
        # They contain frequencies/counts for all values in the cluster,
        # and not just frequencies of the most frequent attributes (stored in the mode)
        self.count = deepcopy(mode.count)
        self.freq = deepcopy(mode.freq) # used only to calculate distance to modes
        # Number of members (modes) of the cluster with this metamode, initially set to 1 (contains mode from which initialisation was done)
        self.nmembers = 1
        self.nrecords = deepcopy(mode.nmembers)

    def calculate_freq(self):
        # create frequencies from counts by dividing each count on total number of values for corresponding attribute for corresponding cluster of this mode
        self.freq = [defaultdict(float) for _ in range(len(self.attrs))]
        for i in range(len(self.count)):
            self.freq[i] = {k: v / self.nrecords for k, v in self.count[i].items()}

    def add_member(self, mode):
        self.nmembers += 1
        self.nrecords += mode.nmembers
        for i in range(len(self.count)):
            # sum and merge mode count to metamode count
            self.count[i] = { k: self.count[i].get(k, 0) + mode.count[i].get(k, 0) for k in set(self.count[i]) | set(mode.count[i]) }

    def subtract_member(self, mode):
        self.nmembers -= 1
        self.nrecords -= mode.nmembers
        if (self.nmembers == 0):
            print("Warning! Last member removed from metamode! This situation should never happen in incremental k-modes!")
        for i in range(len(self.count)):
            # substract and merge mode count from metamode count
            self.count[i] = { k: self.count[i].get(k, 0) - mode.count[i].get(k, 0) for k in set(self.count[i]) | set(mode.count[i]) }

    def update_metamode(self):
        new_mode_attrs = []
        new_mode_attr_freqs = []
        for ind_attr, val_attr in enumerate(self.attrs):
            key,value = get_max_value_key(self.count[ind_attr])
            new_mode_attrs.append(key)
            new_mode_attr_freqs.append(value / self.nrecords)

        self.attrs = new_mode_attrs
        self.attr_frequencies = new_mode_attr_freqs
        self.calculate_freq()

class Mode:
    """
    This is the k-modes mode object 

    - Initialization:
            - just the mode attributes will be initialised
    - Structure:

            - the mode object
            -- consists of mode and frequencies of mode attributes
            - the frequency at which each of the values is observed for each category in each variable
                calculated over the cluster members (.freq)
    - Methods:

            - add_member(record): add a data point to the cluster
            - subtract_member(record): remove a data point from the cluster
            - update_mode: recalculate the centroid of the cluster based on the frequencies.

    """

    def __init__(self, record, mode_id):
        # Initialisation of mode object
        self.attrs = deepcopy(record)
        # the mode is initialised with frequencies, it means that the cluster contains record already.
        # So, frequencies should be set to 1
        self.attr_frequencies = [1]*len(self.attrs)
        # The count and freq are different from frequencies of mode attributes.
        # They contain frequencies/counts for all values in the cluster,
        # and not just frequencies of the most frequent attributes (stored in the mode)
        self.count = [defaultdict(int) for _ in range(len(self.attrs))]
        for ind_attr, val_attr in enumerate(record):
            self.count[ind_attr][val_attr] += 1
        self.freq = None # used only to calculate distance to metamodes, will be initialised within a distance function
        # Number of members of the cluster with this mode, initially set to 1
        self.nmembers = 1
        # index contains the number of the metamode, initially mode does not belong to any metamode, so it is set to -1
        self.index = -1
        self.mode_id = mode_id

    def calculate_freq(self):
        # create frequencies from counts by dividing each count on total number of values for corresponding attribute for corresponding cluster of this mode
        self.freq = [defaultdict(float) for _ in range(len(self.attrs))]
        for i in range(len(self.count)):
            self.freq[i] = {k: v / self.nmembers for k, v in self.count[i].items()}

    def add_member(self, record):
        self.nmembers += 1
        for ind_attr, val_attr in enumerate(record):
            self.count[ind_attr][val_attr] += 1

    def subtract_member(self, record):
        self.nmembers -= 1
        for ind_attr, val_attr in enumerate(record):
            self.count[ind_attr][val_attr] -= 1

    def update_mode(self):
        new_mode_attrs = []
        new_mode_attr_freqs = []
        for ind_attr, val_attr in enumerate(self.attrs):
            key,value = get_max_value_key(self.count[ind_attr])
            new_mode_attrs.append(key)
            new_mode_attr_freqs.append(value / self.nmembers)

        self.attrs = new_mode_attrs
        self.attr_frequencies = new_mode_attr_freqs

    def update_metamode(self, metamodes, similarity):
        # metamodes contains a list of metamode objects.  This function calculates which metamode is closest to the
        # mode contained in this object and changes the metamode to contain the index of this mode.
        # It also updates the metamode frequencies.

        if (similarity == "hamming"):
            diss = hamming_dissim(self.attrs,metamodes)
        elif (similarity == "frequency"):
            diss = frequency_based_dissim(self.attrs,metamodes)
        else: # if (similarity == "meta"):
            diss = all_frequency_based_dissim_for_modes(self,metamodes)

        new_metamode_index = np.argmin(diss)

        moved = 0

        if (self.index == -1):
            # First cycle through
            moved += 1
            self.index = new_metamode_index
            metamodes[self.index].add_member(self)
            metamodes[self.index].update_metamode()
        elif (self.index == new_metamode_index):
            pass
        else: #self.index != new_metamode_index:
            if (diss[self.index] == 0.0):
                print("Warning! Dissimilarity to old metamode was 0, but another new metamode has the same dissimilarity! KMetaModes is going to fail...")
                print("New metamode data: ")
                print(("Attributes: ",metamodes[new_metamode_index].attrs))
                print(("Attribute frequencies: ",metamodes[new_metamode_index].attr_frequencies))
                print(("Number of members: ",metamodes[new_metamode_index].nmembers))
                print(("Number of records: ",metamodes[new_metamode_index].nrecords))
                print(("Counts: ",metamodes[new_metamode_index].count))
                print()
            moved +=1
            metamodes[self.index].subtract_member(self)
            metamodes[self.index].update_metamode()
            metamodes[new_metamode_index].add_member(self)
            metamodes[new_metamode_index].update_metamode()
            self.index = new_metamode_index

        return (metamodes, moved)

def hamming_dissim(record, modes):
    """
    Hamming (simple matching) dissimilarity function
    adapted from https://github.com/nicodv/kmodes
    """
    list_dissim = []
    for cluster_mode in modes:
        sum_dissim = 0
        for elem1,elem2 in zip(record,cluster_mode.attrs):
            if (elem1 != elem2):
                sum_dissim += 1
        list_dissim.append(sum_dissim)
    return list_dissim

def frequency_based_dissim(record, modes):
    """
    Frequency-based dissimilarity function
    inspired by "Improving K-Modes Algorithm Considering Frequencies of Attribute Values in Mode" by He et al.
    """
    list_dissim = []
    for cluster_mode in modes:
        sum_dissim = 0
        for i in range(len(record)): #zip(record,cluster_mode.mode):
            #if (elem1 != elem2):
            if (record[i] != cluster_mode.attrs[i]):
                sum_dissim += 1
            else:
                sum_dissim += 1-cluster_mode.attr_frequencies[i]
        list_dissim.append(sum_dissim)
    return list_dissim

def all_frequency_based_dissim_for_modes(mode, metamodes):
    """
    My own frequency-based dissimilarity function for clustering of modes
    """
    list_dissim = []
    # mode.freq[i] is a set of frequencies for all values of attribute i in the original cluster of this mode
    # metamode.freq[i]
    if (mode.freq is None):
        mode.calculate_freq()
    # for each existing cluster metamode
    for metamode in metamodes:
        sum_dissim = 0
        ##if metamode.freq is None:
        ##    metamode.calculate_freq()
        # for each attribute in the mode
        for i in range(len(mode.attrs)):
            X = mode.freq[i]
            Y = metamode.freq[i]
            # calculate Euclidean dissimilarity between two modes
            sum_dissim += math.sqrt(sum((X.get(d,0) - Y.get(d,0))**2 for d in set(X) | set(Y)))
        list_dissim.append(sum_dissim)
    return list_dissim

class k_modes_record:

    """ A single item in the rdd that is used for training the k-modes 
    calculation.  

        - Initialization:
            - A tuple containing (Index, DataPoint)

        - Structure:
            - the index (.index)
            - the data point (.record)

        - Methods:
            - update_cluster(clusters): determines which cluster centroid is closest to the data point and updates the cluster membership lists appropriately.  It also updates the frequencies appropriately.
    """

    def __init__(self,record):
        self.record = record
        # index contains the number of the mode, initially record does not belong to any cluster, so it is set to -1
        self.index = -1
        self.mode_id = -1

    def update_cluster(self, clusters, similarity):
        # clusters contains a list of cluster objects.  This function calculates which cluster is closest to the
        # record contained in this object and changes the cluster to contain the index of this mode.
        # It also updates the cluster frequencies.

        if (similarity == "hamming"):
            diss = hamming_dissim(self.record,clusters)
        else: # if (similarity == "frequency"):
            diss = frequency_based_dissim(self.record,clusters)

        new_cluster = np.argmin(diss)

        moved = 0

        if (self.index == -1):
            # First cycle through
            moved += 1
            self.index = new_cluster
            self.mode_id = clusters[new_cluster].mode_id
            clusters[new_cluster].add_member(self.record)
            clusters[new_cluster].update_mode()
        elif (self.index == new_cluster):
                pass
        else: #self.index != new_cluster:
            if (diss[self.index] == 0.0):
                raise Exception("Warning! Dissimilarity to old mode was 0, but new mode with the dissimilarity 0 also found! K-modes failed...")
            moved +=1
            clusters[self.index].subtract_member(self.record)
            clusters[self.index].update_mode()
            clusters[new_cluster].add_member(self.record)
            clusters[new_cluster].update_mode()
            self.index = new_cluster
            self.mode_id = clusters[new_cluster].mode_id

        return (self,clusters, moved)

def iter_k_modes(iterator, similarity):
    """ 
    Function that is used with mapPartitionsWithIndex to perform a single iteration
    of the k-modes algorithm on each partition of data.

        - Inputs

            - *clusters*: is a list of cluster objects for all partitions, 
            - *n_clusters*: is the number of clusters to use on each partition

        - Outputs

            - *clusters*: a list of updated clusters,
            - *moved*: the number of data items that changed clusters
    """

    i = 0
    for element in iterator:
        records = element[0]
        partition_clusters = element[1]
        partition_moved = element[2]
        i += 1
    if (i != 1):
        raise Exception("More than 1 element in partition! This is not expected!")

    if (partition_moved == 0):
        yield (records,partition_clusters,partition_moved)
    else:
        partition_records = []
        partition_moved = 0
        # iterator should contain only 1 list of records
        for record in records:
            new_record,partition_clusters, temp_move = record.update_cluster(partition_clusters,similarity)
            partition_records.append(new_record)
            partition_moved += temp_move
        yield (partition_records,partition_clusters,partition_moved)

def partition_to_list(pindex,iterator,n_modes):
        #records
        partition_records = []
        for record in iterator:
            partition_records.append(record)

        #modes
        for _ in range(3):
            i = 0
            failed = 0
            partition_clusters = []
            for index,value in random.sample(list(enumerate(partition_records)),n_modes):
                partition_records[index].mode_id=pindex*n_modes+i
                partition_records[index].index=i
                # check if there is a mode with same counts already in modes:
                if (len(partition_clusters) > 0):
                    diss = hamming_dissim(partition_records[index].record, partition_clusters)
                    if (min(diss) == 0):
                        print("Warning! Two modes with distance between each other equals to 0 were randomly selected. KMetaModes can fail! Retrying random metamodes selection...")
                        failed = 1
                partition_clusters.append(Mode(partition_records[index].record,partition_records[index].mode_id))
                i = i + 1
            if (failed == 0):
                break
        if (failed == 1):
            raise Exception('KMetaModes failed! Cannot initialise a set of unique modes after 3 tries...')

        #moved
        partition_moved = 1
        yield (partition_records,partition_clusters,partition_moved)

def k_modes_partitioned(rdd, n_clusters, max_iter, similarity, seed = None):

    """
    Perform a k-modes calculation on each partition of data.

        - Input:
            - *data_rdd*: in the form (index, record). Make sure that the data is partitioned appropriately: i.e. spread across partitions, and a relatively large number of data points per partition.
            - *n_clusters*: the number of clusters to use on each partition
            - *max_iter*: the maximum number of iterations
            - *similarity*: the type of the dissimilarity function to use
            - *seed*:  controls the sampling of the initial clusters from the data_rdd

        - Output:
            - *clusters*: the final clusters for each partition
            - *rdd*: rdd containing the k_modes_record objects
    """

    # Create initial set of cluster modes by randomly taking {num_clusters} records from each partition
    # For each partition, only the corresponding subset of modes will be used
    #clusters = [Cluster(centroid.record) for centroid in rdd.takeSample(False, n_partitions * n_clusters, seed=None)]
    rdd = rdd.mapPartitionsWithIndex(lambda i,it: partition_to_list(i,it,n_clusters))

    # On each partition do an iteration of k modes analysis, passing back the final clusters. Repeat until no points move
    for iter_count in range(max_iter):
        print(("Iteration ", iter_count))
        # index is partition number
        # iterator is to iterate all elements in the partition
        rdd = rdd.mapPartitions(lambda it: iter_k_modes(it,similarity))

    new_clusters = []
    mode_indexes = []
    for partition_records,partition_clusters,partition_moved in rdd.collect():
        new_clusters.append(partition_clusters)
        partition_mode_indexes = []
        for record in partition_records:
            mode_indexes.append(record.mode_id)
    return (new_clusters,mode_indexes)

def k_metamodes_local(all_modes, n_clusters, max_iter, similarity, seed = None):
    for _ in range(3):
        failed = 0
        # initialise metamodes
        metamodes = []
        i = 0
        for index,value in random.sample(list(enumerate(all_modes)),n_clusters):
            if (all_modes[index].nmembers == 0):
                print("Warning! Mode without members identified!")
                print(("Attributes: ",all_modes[index].attrs))
                print(("Attribute frequencies: ",all_modes[index].attr_frequencies))
                print(("Counts: ",all_modes[index].count))
                print(("Frequencies: ",all_modes[index].freq))
                print()
            all_modes[index].calculate_freq()
            all_modes[index].index = i
            # check if there is a metamode with same counts already in metamodes:
            if (len(metamodes) > 0):
                diss = all_frequency_based_dissim_for_modes(all_modes[index], metamodes)
                if (min(diss) == 0):
                    print("Warning! Two metamodes with distance between each other equals to 0 were randomly selected. KMetaModes can fail! Retrying random metamodes selection...")
                    failed = 1
            metamodes.append(Metamode(all_modes[index]))
            i = i + 1
        if (failed == 0):
            break

    if (failed == 1):
        raise Exception('KMetaModes failed! Cannot initialise a set of unique metamodes after 3 tries...')

    # do an iteration of k-modes analysis, passing back the final metamodes. Repeat until no points move
    moved = 1
    iter_count = 0
    while moved != 0:
        moved = 0

        print(("Iteration ", iter_count))
        iter_count +=1
        iteration_start = time.time()
        for mode in all_modes:
            metamodes, temp_move = mode.update_metamode(metamodes,similarity)
            moved += temp_move

        print(("Iteration ",iter_count-1, "finished within ",time.time()-iteration_start,", moved = ",moved))

        if (iter_count >= max_iter):
            break

    return metamodes

class IncrementalPartitionedKMetaModes:

        """
	Example on how to run k-modes clustering on data:

    		n_modes=36
		partitions=10
		max_iter=10
	    	fraction = 50000 * partitions / (kmdata.count() * 1.0)
	    	data = data.rdd.sample(False,fraction).toDF()
	
	    	method=IncrementalPartitionedKMetaModes(n_partitions = partitions, n_clusters = n_modes,max_dist_iter = max_iter,local_kmodes_iter = max_iter, similarity = "frequency", metamodessimilarity = "hamming")
    	
		cluster_metamodes = method.calculate_metamodes(kmdata)
	
	Now the metamodes can be used, for example, to find the distance from each original data record to all metamodes using one of the existing distance functions, for example:

                def distance_to_all(record):
    		    sum_distance = 0
		    for diss in frequency_based_dissim(record, cluster_metamodes):
			sum_distance += diss
    		    drow = record.asDict()
                    drow["distance"] = sum_distance
                    return Row(**drow)
                data_with_distances = data.repartition(partitions).rdd.map(lambda record: distance_to_all(record))
        """
        def __init__(self, n_partitions, n_clusters, max_dist_iter, local_kmodes_iter, similarity="hamming", metamodessimilarity="hamming"):

            self.n_clusters = n_clusters
            self.n_partitions = n_partitions
            self.max_dist_iter = max_dist_iter
            self.local_kmodes_iter = local_kmodes_iter
            self.similarity = similarity
            self.metamodessimilarity = metamodessimilarity

        def calculate_metamodes(self, kmdata):
            """ Compute distributed k-modes clustering.
            """
            # repartition and convert to RDD
            data_rdd = kmdata.repartition(self.n_partitions).rdd
            print(("Number of partitions: ",data_rdd.getNumPartitions()))
            rdd = data_rdd.map(lambda x: k_modes_record(x))
            print(("Number of partitions after converting to k-modes-records: ",rdd.getNumPartitions()))

            # Calculate the modes for each partition and return the clusters and an indexed rdd.
            print("Starting parallel incremental k-modes...")
            start = time.time()
            modes,self.mode_indexes = k_modes_partitioned(rdd,self.n_clusters,self.max_dist_iter,self.similarity)
            print(("Modes calculated within ",time.time()-start,". Starting calculation of metamodes..."))

            # Calculate the modes for the set of all modes
            # 1) prepare rdd with modes from all partitions
            self.all_modes = []
            print(("Number of partitions: ",len(modes)))
            for one_partition_modes in modes:
                print(("Number of modes in partition: ",len(one_partition_modes)))
                for mode in one_partition_modes:
                    self.all_modes.append(mode)
            print(("Total number of modes: ",len(self.all_modes)))

            # 2) run k-modes on single partition
            self.metamodes = k_metamodes_local(self.all_modes,self.n_clusters,self.local_kmodes_iter, self.metamodessimilarity)

            return self.metamodes

        def get_modes(self):
	"""
	returns all modes (not metamodes!) from all partitions
        """
            return self.all_modes

        def get_mode_indexes(self):
	"""
	returns a list with corresponding mode ID (which is unique) for each original record (not a metamode ID!)
	"""
            return self.mode_indexes

"""
    Given a software, find similar software using source code
    Currently based on software name that exist in the dataset

    TODO: find similar software using source code that is not
    in the existing pool
"""


from LASCAD.LDA.Clustering import Clustering
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from heapq import heappop, heappush
from scipy.stats import entropy
import multiprocessing
import os
from ast import literal_eval as make_tuple


class SimilarSoftwareEngine:

    def __init__(self, NUM_TOPICS=50, max_df=0.5, min_df=0.1, n_clusters=20, maxTopSimilar=100,
                                 dataset='showcase1', verbose=True, normalize=True, loadSaved=False):

        self.clustering = Clustering(NUM_TOPICS, max_df, min_df, dataset, verbose, normalize)
        self.projects = self.clustering.proj_topic.index.values
        self.n_clutsers = n_clusters
        self.maxTopSimilar = maxTopSimilar
        similarAppsFilename = '../results/similarApps/' + 'similarApps_' + self.clustering.suffix + '.csv'

        if loadSaved:
            self.projectsMap = pd.read_csv(similarAppsFilename)
            self.projectsMap.drop('QueryProject', axis=1, inplace=True)
            self.projectsMap = self.projectsMap.as_matrix()
            for i, row in enumerate(self.projectsMap):
                self.projectsMap[i] = [make_tuple(j) for j in row]

        else:
            self.createDistanceMatrix()
            df = pd.DataFrame(self.projectsMap)
            df.index = self.clustering.proj_topic.index
            df.index.name = 'QueryProject'
            df.to_csv(similarAppsFilename)

    # -------------------------------------------------------------------------

    def createDistanceMatrix(self):
        """Pre-compute the distance matrix between every two projects in the dataset
            For each project, store a heap with (key,value) = (projectName, distance)
        """
        # self.clustering.find_categories(n_clusters=self.n_clutsers)
        mat = self.clustering.proj_topic #self.clustering.proj_cat_

        # symmetric matrix, but find all matrix for O(1) search latter
        self.projectsMap = []

        # Run in parallel, return self.maxTopSimilar apps to each software in mat.index
        pool = multiprocessing.Pool(4)
        self.projectsMap = pool.map(self.getSimilarAppsForOneApp, mat.index)

    # ------------------------------------------------------------------

    def getSimilarAppsForOneApp(self, i):

        print('getSimilarAppsForOneApp: ', i)

        mat = self.clustering.proj_topic
        n = self.projects.shape[0]
        tempHeap = []
        for j in mat.index:
            if i == j:
                distance = 1  # avoid the same software
            else:

                v1 = mat.loc[i]
                v2 = mat.loc[j]
                if isinstance(v1, pd.DataFrame):  # if returned multiple
                    v1 = v1.iloc[0]
                if isinstance(v2, pd.DataFrame):  # if returned multiple
                    v2 = v2.iloc[0]
                distance = SimilarSoftwareEngine.JSD(v1, v2)
                # distance = cosine(v1, v2)
                if np.isclose(distance, 0.0):  # same application with diff names (forks)
                    distance = 1

            heappush(tempHeap, (distance, j))

        # sort out
        return [heappop(tempHeap) for k in range(self.maxTopSimilar)]

    # --------------------------------------------------------------------

    def getSimilarSoftwareApps(self, querySoftware, topSimilar=5):
        """
        :param querySoftware: can be project name or project index
        :param topSimilar: number of returned similar apps
        :return: list of tuples: [(similarity_score, similar_app_name), ...]
        """

        queryIndex = querySoftware
        if isinstance(querySoftware, str):
            queryIndex = np.where(self.projects == querySoftware)[0][0]
        return self.projectsMap[queryIndex][:topSimilar]

    # -------------------------------------------------------------------

    # Jensen-Shannon Divergence (between two distributions)
    @staticmethod
    def JSD(P, Q):
        _P = P / np.linalg.norm(P, ord=1)
        _Q = Q / np.linalg.norm(Q, ord=1)
        _M = 0.5 * (_P + _Q)
        return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

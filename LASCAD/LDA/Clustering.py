from __future__ import print_function, division
import os
from time import time
import sklearn
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster.bicluster import SpectralBiclustering

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.spatial.distance import cosine
import json
import matplotlib.gridspec as gridspec

# pd.set_option('display.mpl_style', 'default')
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 60)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
SPINE_COLOR = 'gray'


def load_config(config_file):
    print(os.getcwd())
    with open(config_file) as data_file:
        config_data = json.load(data_file)
    return config_data

try:
    config_files = load_config('../config/const.json')
except:
    config_files = load_config('config/const.json')

base_dir = config_files['base_dir']


class Clustering(object):

    def __init__(self, NUM_TOPICS=50, max_df=0.8, min_df=0.2, dataset='showcase_noStem2',
                    verbose=True, normalize=True, n_clusters=20, categ_method='LASCAD'):

        # Constants
        self.NUM_TOPICS = NUM_TOPICS
        self.max_df = max_df
        self.min_df = min_df
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.categ_method = categ_method

        self.suffix = dataset + '_' + str(NUM_TOPICS) + '_' + str(max_df) + '_' + str(min_df)

        self.projects_details = pd.read_csv(os.path.join(base_dir, 'results/'
                                            + dataset + '/projects_details.csv'),
                                            index_col=0)
        self.topic_word_raw = pd.read_csv(os.path.join(base_dir,
                                          'results/'+dataset+'/topic_word_raw_'+self.suffix+'.csv'),
                                          index_col=0)
        self.proj_topic = pd.read_csv(os.path.join(base_dir,
                                      'results/'+dataset+'/project_topic_'+self.suffix+'.csv'),
                                      index_col=0)
        try:
            self.true_labels = pd.read_csv(os.path.join(base_dir, 'results', self.dataset,
                                           self.dataset+'_projects_categories.csv'), index_col=0)
        except:
            print('Warning: File not found: ', os.path.join(base_dir, 'results', self.dataset,
                                           self.dataset+'_projects_categories.csv'))
            self.true_labels = None

        #  normalize values
        if normalize:
            for i in self.proj_topic.index:
                self.proj_topic.loc[i, self.proj_topic.columns[0:self.NUM_TOPICS]] = \
                    self.proj_topic.loc[i, self.proj_topic.columns[0:self.NUM_TOPICS]] \
                    / (self.proj_topic.loc[i, self.proj_topic.columns[0:self.NUM_TOPICS]].max())

        # clusters: contains projects categories, prog language, date, and topic memberships
        self.clusters = pd.merge(self.projects_details, self.proj_topic, right_index=True,
                                                                         left_index=True)
        self.clusters.reset_index(inplace=True)
        self.clusters.sort_values('group', inplace=True)
        self.clusters.set_index('index', inplace=True)

        # extract only columns with topics membership
        self.proj_topic = self.proj_topic[self.proj_topic.columns[:self.NUM_TOPICS]]

        if verbose:
            print(self.suffix)
            print('Project Details shape: ', self.projects_details.shape)
            print('Topic word raw shape: ', self.topic_word_raw.shape)
            print('Project Topic shape: ', self.proj_topic.shape)
            if self.true_labels is not None:
                print('True labels shape: ', self.true_labels.shape)

        # projects_details.rename(columns={'index': 'project'}, inplace=True)

    # ---------------------------------------------------------------------------------------------

    def get_bicluster(self, data):
        # Biclustering
        model = SpectralBiclustering(n_clusters=data.shape[1], random_state=0)
        print(data.sum(axis=0))
        print(data.sum(axis=1))
        model.fit(data.fillna(0))
        fit_data = data.iloc[np.argsort(model.row_labels_)]
        fit_data = fit_data.iloc[:, np.argsort(model.column_labels_)]

        return fit_data

    def plot_heatmaps(self, proj_topic, customName=''):

        # Biclustering
        # fit_data = self.get_bicluster(proj_topic)
        # print(fit_data)


        clusters = pd.merge(self.projects_details, proj_topic, right_index=True, left_index=True)
        clusters.reset_index(inplace=True)
        clusters.sort_values('group', inplace=True)
        clusters.set_index('index', inplace=True)

        # fig = sns.clustermap(clusters.iloc[:,2:])
        # fig.savefig(os.path.join(base_dir, 'results/'
        #                          + self.dataset
        #                          + '/heatmap'
        #                          + customName
        #                          + '_' + self.suffix
        #                          + '_' + str(self.n_clusters) + '.png'),
        #             bbox_inches='tight', dpi=350)

        categories = clusters.groupby('group')

        #-- 3x2 plot
        plt.figure(figsize=(11, 11))
        gs = gridspec.GridSpec(3, 2,
                               width_ratios=[1, 1.25],
                               height_ratios=[5.5, 4.5, 3.3])

        # #-- 2x3 plot
        # plt.figure(figsize=(14, 7))
        # gs = gridspec.GridSpec(2, 3,
        #                        width_ratios=[1, 1, 1],
        #                        height_ratios=[6.5, 5])

        names = ['Data Visulization', 'Machine Learning', 'Games Engines',
                 'Web Framework', 'Text Editor',  'Web Games']

        new_names = ['Data Visulization', 'Machine Learning', 'Game Engine',
                     'Web Framework', 'Text Editor', 'Web Game']

        topics_order = [14,19,15,  7, 4,  13,16,9,2,0,6,  5,1, 10,17,3,  18,11, 12,8]
        for i, name in enumerate(names):
            g = categories.get_group(name)
            # data = self.get_bicluster(g.iloc[:,2:])
            data = g.loc[:,topics_order]
            # data = g.iloc[:,2:]
            sub_ax = plt.subplot(gs[i])
            # fig = sns.clustermap(g.iloc[:,2:])
            sns.heatmap(data, ax=sub_ax, cbar=(i%2 != 0))# cmap="RdBu_r")
            sub_ax.set_title(new_names[i], fontsize=12.5)
            sub_ax.set_ylabel('')
            if i > 3:
                sub_ax.set_xlabel('Category index', fontdict={'fontsize': 11})
            labels = range(1,21) # sub_ax.get_xticklabels()

            sub_ax.set_xticklabels(labels, rotation=60)
            # plt.setp(sub_ax.get_xticklabels(), visible=False)
            labels = sub_ax.get_yticklabels()
            sub_ax.set_yticklabels(labels, rotation=0)


        plt.tight_layout()

        plt.savefig(os.path.join(base_dir, 'results', self.dataset,
                                 'topic_projects_heatmap'
                                 + customName
                                 + '_' + self.suffix
                                 + '_' +str(self.n_clusters) + '.png'),
                    bbox_inches='tight', dpi=350)

        clusters.iloc[:, topics_order].to_csv(os.path.join(base_dir, 'results', self.dataset, 'project_cat_ordered.csv'))

        # return clusters


    # -----------------------------------------------------------------

    # Helper function for LACT
    def get_cluster_matrix(self, mat, sim, unique, membershipThreshold):
        categ = pd.DataFrame(index=mat.index)
        # merge similar columns (topics)
        for row in sim:
            cat_name = str(row[0]) + '_' + str(row[1])
            for i, proj_index in enumerate(mat.index):  # for every project
                if mat.iloc[i, row[0]] > membershipThreshold  \
                            or mat.iloc[i, row[1]] > membershipThreshold:
                    categ.loc[proj_index, cat_name] = max(mat.iloc[i, row[0]], mat.iloc[i, row[1]])
                else:
                    categ.loc[proj_index, cat_name] = 0

        # for unique topics copy as separate category
        for t in range(self.NUM_TOPICS):
            if unique[t] == 1:
                categ.loc[:, t] = list(map(lambda x: x if x>membershipThreshold else 0, mat.iloc[:, t]))

        for cat in categ: # for each category
            if np.isclose(categ[cat].sum(), 0):
                del categ[cat]      # remove empty categories

        # print('------------Cat Shape: ', categ.shape)
        # print(categ.head())

        return categ


    # Get projects mem
    def get_cluster_matrix_sum(self, mat, sim, unique):
        categ = pd.DataFrame()
        # merge similar columns (topics)
        for row in sim:
            cat_name = str(row[0]) + '_' + str(row[1])
            categ[cat_name] = mat.iloc[:, row[0]] + mat.iloc[:, row[1]]

        # for unique topics copy as separate category
        for i in range(self.NUM_TOPICS):
            if unique[i] == 1:
                categ[i] = mat.iloc[:, i]
        return categ


    def LACT(self, mergingThreshold=0.8, membershipThreshold=0.02, verbose=False):
        """
        Finding CATEGORIES: joining topics based on their words
            ---> topic can belong to many categories
        """
        if verbose:
            print('Finding Categories using LACT')

        mat = self.topic_word_raw.T
        n = self.NUM_TOPICS
        sim = []
        unique = [1] * n
        for i in range(n):
            for j in range(i + 1, n):
                s = 1 - cosine(mat.iloc[:, i], mat.iloc[:, j])
                if s > mergingThreshold:
                    if verbose:
                        print('topic{}-topic{}: {}'.format(i, j, s))
                    sim.append([mat.columns[i], mat.columns[j]])
                    # mark topic i and topic j as non unique categories
                    unique[i], unique[j] = 0, 0


        self.proj_cat_ = self.get_cluster_matrix(self.proj_topic, sim, unique, membershipThreshold)

        if verbose:
            print(sim)
            print('Proj_cat_ shape: ', self.proj_cat_.shape)

        return self.proj_cat_.shape[1]

    # -----------------------------------------------

    def find_categories(self, n_clusters=None, verbose=False, normalize=False):

        # If n_clusters not set, use the class value
        if not n_clusters:
            n_clusters = self.n_clusters

        if verbose: print('Finding categories using Clustering')
        X = self.topic_word_raw
        # X_norm = sklearn.preprocessing.Normalizer().fit_transform(X)
        X_norm = sklearn.preprocessing.scale(X)

        self.labels = AgglomerativeClustering(n_clusters=n_clusters,
                                         affinity='cosine',
                                         linkage='complete').fit_predict(X_norm)

        # labels = sklearn.cluster.KMeans(n_clusters=n_clusters).fit_predict(X_norm)

        if verbose:
            print("Labels: ", self.labels)

        # --- topic_word
        self.cat_word = pd.DataFrame(columns=self.topic_word_raw.columns)
        # merge similar clusters (topics)
        for i, label in enumerate(self.labels):
            if label in self.cat_word.index:
                self.cat_word.loc[label] = self.cat_word.loc[label] + X.iloc[i]  ##### or X_norm
            else:
                self.cat_word.loc[label] = X.iloc[i]  ##### or X_norm

         # ---- Project_cat
        mat = self.proj_topic

        proj_cat = pd.DataFrame()
        for i, label in enumerate(self.labels):
            if label in proj_cat.columns:
                # print('summing {} + {}'.format(proj_cat[label], mat.ix[:,i]))
                proj_cat[label] = proj_cat[label] + mat.iloc[:, i]
            else:
                # print('assiging: {}'.format(mat.ix[:,i]))
                proj_cat[label] = mat.iloc[:, i]


        if normalize:
            for i in proj_cat.index:
                proj_cat.iloc[i, :] /= (proj_cat.iloc[i, :].max())

        self.proj_cat_ = proj_cat
        if verbose:
            print(self.proj_cat_.head())

        self.labels = pd.DataFrame(self.labels, columns=['Cat Num'])

    # ----------------------------------------------------------------------

    @staticmethod
    def get_top_words(cat_words, feature_names, n_top_words):

        df = pd.DataFrame(columns=['word' + str(i) for i in range(n_top_words)])
        df_freq = pd.DataFrame(columns=['word' + str(i) for i in range(n_top_words)])

        for topic_idx, topic in cat_words.iterrows():
            df.loc['Cat#' + str(topic_idx)] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            df_freq.loc['Cat#' + str(topic_idx)] = [topic[i] for i in topic.argsort()[:-n_top_words - 1:-1]]

        return df, df_freq

    # --------------  Generate clustering accuracy  -------------------------

    @staticmethod
    def precision_soft(index, pred, true):

        intersect = np.arange(0)
        n = 0
        # print(index, '--->', pred.loc[index][pred.loc[index]>0].index.values)
        for topic in pred.loc[index][pred.loc[index] > 0].index:
            c_pred_i = topic  # pred.loc[index].idxmax() # the column(topics) of max membership
            c_pred = pred[pred[c_pred_i] > 0].index.values  # all the cluster of 'index'
            c_true = true[true == true[index]].index.values  # true cluster members
            intersect2 = np.intersect1d(c_pred, c_true)
            if intersect2.shape[0] > intersect.shape[0]:  # choose larger cluster
                intersect = intersect2
                n = c_pred.shape[0]

        p = intersect.shape[0] / n if n>0 else 0
        return p


    @staticmethod
    def recall_soft(index, pred, true):

        intersect = np.arange(0)
        for topic in pred.loc[index][pred.loc[index] > 0].index:
            c_pred_i = topic  # pred.loc[index].idxmax() # the column(topics) of max membership
            c_pred = pred[pred[c_pred_i] > 0].index.values  # all the cluster of 'index'
            c_true = true[true == true[index]].index.values  # true cluster members
            intersect2 = np.intersect1d(c_pred, c_true)
            if intersect2.shape[0] > intersect.shape[0]:  # choose larger cluster
                intersect = intersect2

        return intersect.shape[0] / c_true.shape[0]


    @staticmethod
    def get_F_score(precision, recall):
        if ((precision+recall) != 0):
            return 2 * precision * recall / (precision+recall)
        else:
            return 0


    @staticmethod
    def binarizeMatrix(mat):
        """ Return mat with the max value in each row=1, others=0"""
        max_clusters = mat.idxmax(axis=1)
        bin_mat = mat.copy()
        for i in bin_mat.index:
            bin_mat.iloc[i, max_clusters[i]] = 100

        # remove values below the threshold
        bin_mat = bin_mat.clip_lower(99).replace(99, 0).replace(100, 1)

        return bin_mat

    @staticmethod
    def removeBelowThresold(mat, threshold):
        """ remove values below the given threshold """
        return mat.clip_lower(threshold).replace(threshold, 0)


    def calculate_accuracy(self, proj_cat, verbose=True):

        labels = self.clusters[self.clusters.columns[0]]
        print(proj_cat.head())
        if verbose:
            print('num of projects per predicted cluster')
            num_proj_per_cluster = proj_cat.astype(bool).sum(axis=1)
            print(num_proj_per_cluster)

            print('# Number of projects per true clusters')
            print(labels.value_counts())

        precision = 0
        for i in proj_cat.index:
            p = Clustering.precision_soft(i, proj_cat, labels)
            precision += p

        precision = precision / labels.shape[0]
        print('Precision', precision)

        recall = 0
        for i in proj_cat.index:
            p = Clustering.recall_soft(i, proj_cat, labels)
            recall += p

        recall = recall / labels.shape[0]
        print('Recall', recall)

        f_score = Clustering.get_F_score(precision, recall)
        print('F score: ', f_score)

        return [precision, recall, f_score]

    # --------------------  MudaBlue accuracy method ----------------------

    def cluster_labeling(self, pred_proj_cat, true_proj_cat):

        # taken = dict([(true, 0) for true in true_proj_cat])
        for pred in pred_proj_cat:
            max_cat_count, max_cat_name = 0, str(pred)+'None'
            for true in true_proj_cat:
                similar = (pred_proj_cat[pred].astype(bool) & true_proj_cat[true].astype(bool)).sum()
                if similar > max_cat_count:  # and taken[true] < 2:
                    max_cat_count = similar
                    max_cat_name = true
                    # taken[true] += 1
            # Rename the column to the true cat with highest projects intersection
            if self.categ_method == 'LASCAD':
                self.labels.loc[self.labels['Cat Num'] == pred, 'Cat Name'] = max_cat_name
            pred_proj_cat.rename(columns={pred: max_cat_name}, inplace=True)


    @staticmethod
    def precision_recall_soft_mudaBlue(project_index, pred, true, verbose=False):

        # get a list of the labels of this project_index
        try:
            pred_labels_list = pred.loc[project_index] [pred.loc[project_index].nonzero()[0]].index.values
        except:
            print('error in proj index: ', project_index)
            print(pred.loc[project_index].shape)
            pred_labels_list = []
        pred_labels_list = [str(i) for i in pred_labels_list] # to string
        pred_labels_list = np.unique(pred_labels_list)
        true_labels_list = true.loc[project_index][true.loc[project_index].nonzero()[0]].index.values
        intersect = np.intersect1d(pred_labels_list, true_labels_list)
        if verbose:
            print('pred_labels_list: ', pred_labels_list)
            print('true_labels_list: ', true_labels_list)
            print('Intersect: ', intersect)

        p = intersect.shape[0] / len(pred_labels_list) if len(pred_labels_list) > 0 else 0
        r = intersect.shape[0] / len(true_labels_list) if len(true_labels_list) > 0 else 0

        return p, r


    def calculate_accuracy_mudaBlueMethod(self, proj_cat, verbose=False):

        # proj_cat_true tabels
        # true_labels = self.clusters[self.clusters.columns[0]]
        # true_labels = pd.read_csv(os.path.join(base_dir, 'results', self.dataset,
        #                                    self.dataset+'_projects_categories.csv'), index_col=0)

        if verbose:
            print('True labels:')
            print(self.true_labels.head())
            print('Predicted labels:')
            print(proj_cat.head())
            print('num of projects per predicted cluster')
            num_proj_per_cluster = proj_cat.astype(bool).sum(axis=1)
            print(num_proj_per_cluster)

            print('# Number of projects per true clusters')
            print(self.true_labels.astype(bool).sum(axis=1))


        precision, recall = 0, 0
        accuracy_log = pd.DataFrame(columns=['precision', 'recall'])
        for i in proj_cat.index:
            p, r = Clustering.precision_recall_soft_mudaBlue(i, proj_cat, self.true_labels)
            accuracy_log.loc[i] = [p, r]
            precision += p
            recall += r

        accuracy_log = accuracy_log
        print('Saving accuracy log..')
        accuracy_log.to_csv(os.path.join(base_dir, 'results', 'categorization_accuracy',
                                  self.categ_method + '_projects_accuracy_log_' + self.dataset + '.csv'))

        precision = precision / self.true_labels.shape[0]
        recall = recall / self.true_labels.shape[0]
        print('Precision', precision)
        print('Recall', recall)

        f_score = Clustering.get_F_score(precision, recall)
        print('F score: ', f_score)

        return [precision, recall, f_score]


# ------------------------------------------------------------------

def testClustering(NUM_TOPICS=50, max_df=0.8, min_df=0.2, dataset='showcase_noStem2', n_clusters=20,
                            verbose=True, plot_heatmap=False, categ_method='LASCAD', normalize=False):

    # 1- Initialize and read saved LDA output:
    test = Clustering(NUM_TOPICS=NUM_TOPICS,
                      max_df=max_df,
                      min_df=min_df,
                      dataset=dataset,
                      n_clusters=n_clusters,
                      categ_method=categ_method
                      )
    # if plot_heatmap: test.plot_heatmaps(test.proj_topic, customName='original_topics')


    # 2- Find categories:
    if categ_method == 'LACT':
        n_clusters = test.LACT(mergingThreshold=0.8, membershipThreshold=0.02, verbose=verbose)
    else:
        test.find_categories(verbose=verbose, normalize=normalize)
        # Remove membership values below a threshold (balance between precision and recall)
        test.proj_cat_ = test.removeBelowThresold(test.proj_cat_, threshold=0.3)

    if plot_heatmap: test.plot_heatmaps(test.proj_cat_, customName='pred_categories_'+categ_method) # proj_categories

    print('Predicted labels:')
    print(test.proj_cat_.head())
    print(test.proj_cat_.shape)

    # Label clusters
    test.cluster_labeling(test.proj_cat_, test.true_labels)

    # Save topic, cat, mapping
    if categ_method == 'LASCAD':
        test.labels.to_csv(os.path.join(base_dir, 'results/' + dataset + '/topic_cat_labels_'
                                                  + test.suffix + '_' + categ_method + '.csv'))

    print('Predicted NEW labels:')
    print(test.proj_cat_.head())
    print(test.proj_cat_.shape)
    print('Count per category:')
    print(test.proj_cat_.astype(bool).sum(axis=0))
    test.proj_cat_.to_csv(os.path.join(base_dir,
                    'results/' + dataset + '/project_cat_' + test.suffix + '_' + categ_method + '.csv'))

    # test.cat_word.to_csv(os.path.join(base_dir, 'results/' + dataset + '/cat_word_raw_'
    #                                   + test.suffix + '_' + categ_method + '.csv'))
    #
    #
    # df, df_freq = test.get_top_words(test.cat_word, test.cat_word.columns.values, 100)
    # df.to_csv(os.path.join(base_dir, 'results/' + dataset + '/cat_word_'
    #                            + test.suffix + '_' + categ_method + '.csv'))
    #
    # df_freq.to_csv(os.path.join(base_dir, 'results/' + dataset + '/cat_word_freq_'
    #                        + test.suffix + '_' + categ_method + '.csv'))

    # 4- Binarize matrix (optional)
    # bin_mat = test.binarizeMatrix(mat) # remove values below the threshold
    # if plot_heatmap: test.plot_heatmaps(bin_mat, customName='binarized')
    # mat = bin_mat

    # [precision, recall, f_score] = test.calculate_accuracy(mat)
    [precision, recall, f_score] = test.calculate_accuracy_mudaBlueMethod(test.proj_cat_)


    return [precision, recall, f_score], n_clusters

# -----------------------------------------------------------------------------

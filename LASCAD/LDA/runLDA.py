# ***************** Topic Modeling  *******************
# *****************************************************

from __future__ import print_function
import os
import json
import sys
import time
from os import listdir
from os.path import isdir

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pandas as pd
import numpy as np
#import pickle
#import matplotlib.pyplot as plt

# import lda

# pd.set_option('display.mpl_style', 'default')
# pd.set_option('display.width', 5000)
# pd.set_option('display.max_columns', 60)

# -------------------------------------------------------------------


def load_config(config_file):
    print(os.getcwd())
    with open(config_file) as data_file:
        config_data = json.load(data_file)
    return config_data


config_files = load_config('../config/const.json')
base_dir = config_files['base_dir']
# out_dir = os.path.join(base_dir , config_files['out_dir'])
out_dir = config_files['out_dir']
# Constants:


# -------------------------------------------------------------------

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

# -------------------------------------------------------------------

def get_top_words(model, feature_names, n_top_words):
    df = pd.DataFrame(columns=['word'+str(i) for i in range(n_top_words)])
    df_freq = pd.DataFrame(columns=['word'+str(i) for i in range(n_top_words)])
    for topic_idx, topic in enumerate(model.components_):
        df.loc['topic#'+str(topic_idx)] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        df_freq.loc['topic#'+str(topic_idx)] = [topic[i] for i in topic.argsort()[:-n_top_words - 1:-1]]

    return df, df_freq

# -------------------------------------------------------------------

def print_full(x):
    pd.set_option('display.max_colwidth', 1000)
    print(x)
    pd.reset_option('display.max_rows')

# -------------------------------------------------------------------

def run_lda_sklearn(X, n_topics):
    lda = LatentDirichletAllocation(n_topics=n_topics,
                                max_iter=1000,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0,
                                n_jobs=1)

    projects_topics = lda.fit_transform(X)
    print('n_itations: ', lda.n_iter_)
    return projects_topics, lda

# -------------------------------------------------------------------

def run_lda_other(X, n_topics):
    lda = lda.LDA(n_topics=n_topics, n_iter=500, random_state=1)
    lda.fit_transform(X)

    return lda.doc_topic_, lda

# -------------------------------------------------------------------

def get_projects_data_tags(selected_projects):
    """Return array or strings of the project tags inside each project
        Read projects into strings
    """
    projects_data = []
    project_names = []
    for i, project_name in enumerate(selected_projects):

        # For each snapshot of the project
        snapshots = []
        project_path = os.path.join(out_dir, project_name+'-tags')
        try:
            snapshots = [os.path.join(project_path, p)
                         for p in listdir(project_path) if isdir(os.path.join(project_path, p))]
        except FileNotFoundError:
             print('------Project Not found: '+project_name)
        for snapshot in snapshots:
            project_names.append('_'.join(snapshot.split('/')[-2:]))
            # print(project_names[-1])
            processed_path = os.path.join(snapshot, config_files['final_processed'])
            with open(processed_path, 'r') as myfile:
                projects_data.append(myfile.read().replace(r'\n', ' '))

    return project_names, projects_data

# ----------------------------------------------------------------

def get_projects_data(selected_projects):
    '''Return array or strings of the project
        Read projects into strings
    '''
    projects_data = []
    for i, project_name in enumerate(selected_projects):
        processed_path = os.path.join(base_dir, out_dir, project_name.lower(),
                                       config_files['final_processed'])
        #processed_path = os.path.join(out_dir, project_name+'_'+config_files['final_processed'])
        with open(processed_path, 'r', encoding="utf-8") as myfile:
            temp = myfile.read().replace(r'\n', ' ')
            # print(project_name, ', Size=', len(temp))
            projects_data.append(temp)

    return projects_data

# ----------------------------------------------------------------

# *****************   RUN LDA test  ****************************
# --------------------------------------------------------------

def run_LDA_part(args):
    run_LDA(**args)


def run_LDA(projects_data, projects_names, test, n_topics, min_df, max_df,
                                    n_top_words=50, n_features=50000):

    suffix = test + '_' + str(n_topics) + '_' + str(max_df) + '_' + str(min_df)
    print('------suffix:', suffix)

    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    count_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=n_features,
                                       stop_words='english')
    # tfidf_vectorizer = TfidfTransformer()

    t0 = time.time()
    counts = count_vectorizer.fit_transform(projects_data)
    # tfidf = tfidf_vectorizer.fit_transform(counts)
    X = counts  # <---------
    tf_feature_names = count_vectorizer.get_feature_names()
    # tf_feature_names = tfidf_vectorizer.get_feature_names()
    print("done in %0.2f min." % ((time.time() - t0)/60.))
    print('X shape:', X.shape)

    print("Fitting LDA models with tf features")

    t0 = time.time()
    # projects_topics, lda = run_lda_other(X, n_topics) # counts
    projects_topics, lda = run_lda_sklearn(X, n_topics)  # counts
    print("Fit LDA done.. getting frequent words..")
    topic_word, topic_word_freq = get_top_words(lda, tf_feature_names, n_top_words)
    print("done in %0.2f min." % ((time.time() - t0)/60.))

    # -------------------------------------------------------------------
    # Save lda into a pickle file
    # pickle.dump(lda, open(os.path.join(base_dir, 'results/lda_'+suffix+'.p'), 'wb'))

    pd.DataFrame(lda.components_, columns=tf_feature_names).to_csv(os.path.join(base_dir, 'results', test,
                                        config_files['topic_word_raw'] + suffix + '.csv'))
    topic_word.to_csv(os.path.join(base_dir,  'results', test,
                                        config_files['topic_word'] + suffix + '.csv'))
    topic_word_freq.to_csv(os.path.join(base_dir,  'results', test,
                                        config_files['topic_word_freq'] + suffix + '.csv'))

    projects_topics = pd.DataFrame(projects_topics, columns=['topic' + str(i) for i in range(n_topics)])
    projects_topics['project'] = projects_names
    projects_topics['project'] = projects_topics['project'].apply(lambda x: x.split('_')[0].split('-')[0])
    projects_topics['date'] = '2016-10'
    projects_topics.index = projects_names
    projects_topics.to_csv(os.path.join(base_dir, 'results', test,
                                        config_files['project_topic'] + suffix + '.csv'))

    print('LDA of ' + suffix + ' is ALL Done............')
    # lda = pickle.load(open("lda_5_1.p", "rb"))
    return topic_word, topic_word_freq, projects_topics

# ----------------------------------------------------------------------

def run_LDA_showcases():

    n_features = 50000  # const
    n_top_words = 50 # const

    test = config_files['dataset_showcase_noStem2']
    # test = config_files['dataset_showcases'] # 'showcase1'

    project_details = load_config(os.path.join(base_dir, config_files['showcases_data']))
    project_details = pd.DataFrame(project_details).T
    project_details.to_csv(os.path.join(base_dir, 'results', test, config_files['projects_details']))
    projects_names = project_details.index.values
    projects_data = get_projects_data(projects_names)

    print('Projects len: ', len(projects_data))

    for n_topics in [50]:  # range(5, 150, 5):
        for max_df in [0.8]:
            for min_df in [0.2]:

                suffix = test + '_' + str(n_topics) + '_' + str(max_df) + '_' + str(min_df)
                print('------suffix:', suffix)

                run_LDA(projects_data, projects_names, test, n_topics, min_df, max_df,
                        n_top_words=n_top_words, n_features=n_features)


    print('Testing is Done............')

# -----------------------------------------------------------

def run_LDA_largeDataset_readme():

    n_features = 50000  # const
    n_top_words = 50 # const

    test = config_files['dataset_largeDataset_readme']

    project_details = pd.read_csv(os.path.join(base_dir, config_files['largeDataset_data']))
    project_details.set_index('name', inplace=True)
    project_details.to_csv(os.path.join(base_dir, 'results', test, config_files['projects_details']))
    projects_names = project_details.index.values

    readme = pd.read_csv(os.path.join(base_dir, 'config', 'readme_files.csv')).readme
    readme.fillna('', inplace=True)
    projects_data = readme.replace({r'\\r': ' ', r'\\n': ' '}, regex=True).values

    print('Projects len: ', len(projects_data))

    for n_topics in [50]:  # range(5, 150, 5):
        for max_df in [0.8]:
            for min_df in [0.2]:

                suffix = test + '_' + str(n_topics) + '_' + str(max_df) + '_' + str(min_df)
                print('------suffix:', suffix)

                run_LDA(projects_data, projects_names, test, n_topics, min_df, max_df,
                        n_top_words=n_top_words, n_features=n_features)


    print('Testing is Done............')


# -----------------------------------------------------------


def run_LDA_showcases_parallel(arg_index):

    test = config_files['dataset_showcase_noStem2']
    # test = config_files['dataset_showcases'] # 'showcase1'


    project_details = load_config(os.path.join(base_dir, config_files['showcases_data']))
    project_details = pd.DataFrame(project_details).T
    project_details.to_csv(os.path.join(base_dir, 'results', test, config_files['projects_details']))
    projects_names = project_details.index.values
    projects_data = get_projects_data(projects_names)


    args = []
    for n_topics in [50, 100]: #[20, 30, 40, 50, 60]:
        for max_df in [0.5]:
            for min_df in [0.4]:

                args.append({"projects_data": projects_data,
                             "projects_names": projects_names,
                             "test": test,
                             "n_topics": n_topics,
                             "min_df": min_df,
                             "max_df": max_df
                })

    # pool = multiprocessing.Pool(16)
    # pool.map(run_LDA_part, args)
    run_LDA(**args[arg_index])

    print('Testing is Done............')

# ------------------------------------------------------------

def run_LDA_largeDataset_parallel(arg_index):

    test = config_files['dataset_largeDataset']

    project_details = pd.read_csv(os.path.join(base_dir, config_files['largeDataset_data']))
    project_details.set_index('name', inplace=True)
    project_details.to_csv(os.path.join(base_dir, 'results', test, config_files['projects_details']))
    projects_names = project_details.index.values
    projects_data = get_projects_data(projects_names)


    args = []
    for n_topics in [100, 50]: #[20, 30, 40, 50, 60]:
        for max_df in [0.8, 0.5]:
            for min_df in [0.05]:

                args.append({"projects_data": projects_data,
                             "projects_names": projects_names,
                             "test": test,
                             "n_topics": n_topics,
                             "min_df": min_df,
                             "max_df": max_df
                })

    # pool = multiprocessing.Pool(16)
    # pool.map(run_LDA_part, args)
    # print("runnign LDA with ars: ", args[arg_index]["n_topics"])
    run_LDA(**args[arg_index])


# ------------------------------------------------------------------------------


def run_LDA_LACT_data(test='LACT41'):

    n_features = 50000
    # n_topics = 25
    n_top_words = 50

    # test = 'LACT41'

    projects_data = pd.read_csv(os.path.join(base_dir, config_files[test+'_data'])).ix[:, 0].tolist()
    print('Num of projects: ', len(projects_data))
    # print(projects_data[0])

    project_details = pd.DataFrame(columns=["index", "group", "language"])
    project_details['index'] = pd.read_csv(os.path.join(base_dir, config_files[test+'_names']), header=None, sep=' ').ix[:,3]
    # print(project_details)
    project_details['language'] = project_details['index'].map(lambda x: x.split('_')[0])
    project_details['group'] = project_details['index'].map(lambda x: x.split('_')[1])

    # replace type then replace group
    project_details['index'] = project_details['index'].map(lambda x: x.replace(x.split('_')[0] + '_', ''))
    project_details['index'] = project_details['index'].map(lambda x: x.replace(x.split('_')[0] + '_', ''))

    project_details.set_index('index', inplace=True)
    # print(project_details)
    project_details.to_csv(os.path.join(base_dir, 'results', test, config_files['projects_details']))
    projects_names = project_details.index.values


    for n_topics in [50]:
        for max_df in [0.9]: #[0.9, 0.8, 0.7, 0.6]:
            for min_df in [.05]:

                suffix = test + '_' + str(n_topics) + '_' + str(max_df) + '_' + str(min_df)
                print('------suffix:', suffix)

                run_LDA(projects_data, projects_names, test, n_topics, min_df, max_df,
                        n_top_words=n_top_words, n_features=n_features)


    print('Testing is Done............')

# ------------------------------------------------------------------------------


if __name__ == "__main__":

    # print('--- python args: ', sys.argv[1])
    # run_LDA_showcases()
    # run_LDA_showcases_example()
    # run_LDA_showcases_parallel(int(sys.argv[1]))
    # run_LDA_LACT_data('LACT41')
    # run_LDA_largeDataset_parallel(int(sys.argv[1]))
    run_LDA_largeDataset_readme()
import sys
from LASCAD.LDA.Clustering import Clustering
from LASCAD.LDA.SimilarSoftwareEngine import SimilarSoftwareEngine
from LASCAD.LDA.SimilarSoftware_ByDescription import SimilarSoftware_ByDescription
import pandas as pd
import numpy as np
import time
from scipy import stats
from os.path import join

# ----------------------------------------------------------

def printSimilar(result_out, app, engine, top=10):
    n = result_out.shape[0]
    res = engine.getSimilarSoftwareApps(app, topSimilar=top)
    c = engine.clustering.projects_details
    queryIndex = np.where(engine.projects == app)[0][0]
    # 'Query/result', 'Name', 'Group', 'language', 'Description'
    result_out.loc[n] = ['Query', c.iloc[queryIndex].name, c.iloc[queryIndex].group,
                         c.iloc[queryIndex].language, c.iloc[queryIndex].description]
    n += 1
    print('query project: ', c.iloc[queryIndex])

    for i in res:
        index = np.where(engine.projects == i[1])[0][0]
        print('Similar app: {}, \t{}, \t({}), Description:{}'.format(i[1], c.iloc[index].group, c.iloc[index].language,
                        c.iloc[index].description))
        result_out.loc[n] = ['Result', c.iloc[index].name, c.iloc[index].group,
                             c.iloc[index].language, c.iloc[index].description]
        n += 1

    return res

# ---------------------------------------------------------

def test_findSimilarApps(engine, top):
    """Find similar apps for all projects"""

    projects_details = engine.clustering.projects_details
    n_tasks = projects_details.shape[0]
    count_top = np.zeros(top)
    tasks_precision = pd.DataFrame(index=projects_details.index, columns=['precision'])
    tasks_precision['precision'] = 0

    found = 0
    for proj_index, proj in enumerate(projects_details.index):

        res = engine.getSimilarSoftwareApps(proj, topSimilar=top)

        curr_cat = projects_details.iloc[proj_index].group
        flag = 0
        for i, p in enumerate(res):
            simProj = projects_details.loc[p[1]]
            if isinstance(simProj, pd.DataFrame):
                simProj = simProj.iloc[0]

            if curr_cat == simProj.group:
                count_top[i] += 1
                tasks_precision.iloc[proj_index] += 1
                flag = 1
        found += flag

    print('Total tasks with hit: ', found)
    precision = (count_top.sum() / n_tasks / top)
    # print('precision: {}'.format(precision))
    print('Count top: {}'.format(count_top / n_tasks))

    return precision, tasks_precision/top


def get_precision_similarApps(engine, querySoftware, similarList, top):

    queryIndex = np.where(engine.projects == querySoftware)[0][0]
    projects_details = engine.clustering.projects_details
    count_top = np.zeros(top)

    curr_cat = projects_details.iloc[queryIndex].group

    for i, p in enumerate(similarList):
        simProj = projects_details.loc[p[1]]
        if isinstance(simProj, pd.DataFrame):
            simProj = simProj.iloc[0]

        if curr_cat == simProj.group:
            count_top[i] += 1

    precision = (count_top.sum() / top)
    # print('precision: {}'.format(precision))

    return precision, count_top

# ---------------------------------------------------------

def test_findSimialrApps_Random(projects_details, top):
    """Finding similar apps by Random suggestions"""

    n_tasks = projects_details.shape[0]
    count_top_R = np.zeros(top)
    tasks_precision = np.zeros(n_tasks)
    found_R = 0
    for proj_index, proj in enumerate(projects_details.index):
        res = np.random.random_integers(0, projects_details.shape[0] - 1, size=top)
        #     print('query project: ', projects_details.loc[proj])

        curr_cat = projects_details.iloc[proj_index].group
        flag = 0
        for i, p in enumerate(res):
            simProj = projects_details.iloc[p]
            if isinstance(simProj, pd.DataFrame):
                simProj = simProj.iloc[0]

            if curr_cat == simProj.group:
                count_top_R[i] += 1
                tasks_precision[proj_index] += 1
                flag = 1

        found_R += flag

    print('Total tasks with hit (Random): ', found_R)
    precision = (count_top_R.sum() / n_tasks / top)
    # print('precision: {}'.format(precision))
    print('Count top: {}'.format(count_top_R / n_tasks))

    return precision, tasks_precision/top

# ----------------------------------------------------------

def test_findSimialrApps_textSearch(engine, top):
    """Text search in project's description"""

    # test all projects
    projects_details = engine.projects_details
    n_tasks = projects_details.shape[0]
    count_top = np.zeros(top)
    tasks_precision = np.zeros(n_tasks)
    found = 0
    for proj_index, proj in enumerate(projects_details.index):

        res = engine.getSimilarSoftwareApps(proj_index, topSimilar=top)

        curr_cat = projects_details.iloc[proj_index].group
        flag = 0
        for i, p in enumerate(res):
            simProj = projects_details.loc[p]
            if isinstance(simProj, pd.DataFrame):
                simProj = simProj.iloc[0]

            if curr_cat == simProj.group:
                count_top[i] += 1
                tasks_precision[proj_index] += 1
                flag = 1
        found += flag

    print('Total tasks with hit in whoosh: ', found)
    precision = (count_top.sum() / n_tasks / top)
    # print('precision: {}'.format(precision))
    print('Count top: {}'.format(count_top / n_tasks))

    return precision, tasks_precision/top

# ----------------------------------------------------------

def find_pvalue_1smaple(x_hat, real_mean):
    pval_2sided = stats.ttest_1samp(x_hat, real_mean)
    return pval_2sided


def find_pvalue_2smaple_paired(a, b):
    """a and b are arrays"""
    pval_2sided = stats.ttest_rel(a, b)
    return pval_2sided


def find_pvale_2smaples_independent(a, b):
    """a and b are arrays"""
    pval_2sided = stats.ttest_ind(a, b)
    return pval_2sided


def print_accuracy(NUM_TOPICS=50, max_df=0.5, min_df=0.2, n_clusters=20,
                   dataset='showcase_noStem2', loadSaved=False):
    top = 5
    t0 = time.time()
    engine = SimilarSoftwareEngine(NUM_TOPICS=NUM_TOPICS, max_df=max_df,
                                   min_df=min_df, n_clusters=n_clusters,
                                   dataset=dataset, loadSaved=loadSaved)
    print("done in %0.2f min." % ((time.time() - t0)/60.))


    t0 = time.time()
    print('Finding similarity for LASCAD:')
    LASCAD_precision, precision_per_app_LASCAD = test_findSimilarApps(engine, top)
    precision_per_app_LASCAD.to_csv(join('..', 'results', 'similarApps', 'similar_app_precision_' + dataset + '.csv'))
    print("done in %0.2f min." % ((time.time() - t0)/60.))

    if dataset == 'largeDataset':
        engine2 = SimilarSoftware_ByDescription(NUM_TOPICS=NUM_TOPICS, max_df=max_df,
                                            min_df=min_df, n_clusters=n_clusters,
                                            dataset=dataset)

        t0 = time.time()
        print('Finding similarity for Whoosh:')
        whoosh_precision, b = test_findSimialrApps_textSearch(engine2, top)
        print("done in %0.2f min." % ((time.time() - t0)/60.))
        print('Whoosh precesion:', whoosh_precision)

    t0 = time.time()
    print('Finding similarity for Random:')
    random_precision, precision_per_app_random = test_findSimialrApps_Random(engine.clustering.projects_details, top)
    print("done in %0.2f min." % ((time.time() - t0)/60.))

    print('LASCAD precession for top {}: {}'.format(top, LASCAD_precision))
    print('Random precesion for top {}: {}'.format(top, random_precision))

    # P-value
    # pval = find_pvalue_2smaple_paired(precision_per_app_LASCAD.values, precision_per_app_random)
    # print('Pval paired: ', pval)
    #
    # pval = find_pvale_2smaples_independent(precision_per_app_LASCAD.values, precision_per_app_random)
    # print('Pval independent: ', pval)

    return engine


def print_accuracy_text_search(NUM_TOPICS=50, max_df=0.8, min_df=0.2, n_clusters=20,
                   dataset='showcase_noStem2', loadSaved=True):
    top = 5
    engine = SimilarSoftware_ByDescription(NUM_TOPICS=NUM_TOPICS, max_df=max_df,
                                            min_df=min_df, n_clusters=n_clusters,
                                            dataset=dataset)
    t0 = time.time()
    print('Finding similarity for Whoosh:')
    whoosh_precision, a = test_findSimialrApps_textSearch(engine, top)
    print("done in %0.2f min." % ((time.time() - t0)/60.))

    t0 = time.time()
    print('Finding similarity for Random:')
    random_precision, b = test_findSimialrApps_Random(engine.projects_details, top)
    print("done in %0.2f min." % ((time.time() - t0)/60.))

    print('Whoosh precesion:', whoosh_precision)
    print('Random precesion:', random_precision)

    pval = find_pvalue_2smaple_paired(a, b)
    print('Pval paired: ', pval)

    pval = find_pvale_2smaples_independent(a, b)
    print('Pval independent: ', pval)

    return engine

# ----------------------------------------------------------

def calc_manualEvaluationFile(filename):

    d = pd.read_csv(filename, encoding='ISO-8859-1')
    d.fillna(0, inplace=True)
    n = d.query('Type=="Result"').shape[0]
    n_queries = int(n / 10)
    print('n_queries: ', n_queries)

    print('Top i-th hit: \n------------')
    x = d.query('Type=="Result"')['isSimilar'].values
    y = pd.DataFrame(x.reshape(int(n / 10), 10))
    y['Name'] = d.query('Type=="Query"')['Name'].values
    y.set_index('Name', inplace=True)

    print(y.sum(axis=0))

    print('Precision per top i-th result: \n---------------')
    z = round(y.sum(axis=0) / n_queries * 100).astype(int)
    print(z)

    precision = round(x.sum() / n_queries / 10 * 100)  # or = z.sum()/10
    print('Total precision: ', precision.astype(int))

    y['Total'] = y.iloc[:, 0:10].sum(axis=1)
    print(y)
    y.to_csv(filename + '_scores.csv')


# ---------------------------------------------------------

# Set loadSaved=False if running the experiment for a specific dat for the first time.
# It will take long time to run for the first time.

### paper results for the manual evaluation of the 38 similar apps experiment
calc_manualEvaluationFile('../results/similarApps/test_queries_largeDataset_0.02_Ayat_manualEval.csv')

## showcases data
print_accuracy(NUM_TOPICS=50, max_df=0.7, min_df=0.1, n_clusters=20, dataset='showcase_noStem2', loadSaved=True)

### Large dataset lower bound test
print_accuracy(NUM_TOPICS=50, max_df=0.5, min_df=0.05, n_clusters=20, dataset='largeDataset', loadSaved=True)

# Text search experiment
print_accuracy_text_search(NUM_TOPICS=50, max_df=0.5, min_df=0.02, n_clusters=20, dataset='largeDataset', loadSaved=True)



# Small test for specific projects
# result_out = pd.DataFrame(columns=['Query/result', 'Name', 'Group', 'language', 'Description'])
#
# queryList = ['express', 'django', 'slap', 'Vim', 'Knife', 'Typewriter', 'JCEditor',
#      'closure-compiler', 'cython', 'TypeScript', 'icecream',
#      'scikit-learn', 'numl', 'Conjecture', 'pylearn2', 'tensorflow',
#      'Rocket.Chat', 'quietnet']
# precision = 0
# count_top = np.zeros(10)
#
# for i in queryList:
#     try:
#         res = printSimilar(result_out, i, engine, top=10)
#         prec, count = get_precision_similarApps(engine, i, res, 10)
#         precision += prec
#         count_top = count_top + count
#         print('Prec: ', prec)
#         print(count)
#         result_out.to_csv('../results/similarApps/' + 'test_queries_largeDataset_0.02.csv', index=False)
#     except:
#         print('Error in ', i)
#
#     print('Final precision: ', precision/len(queryList))
#     print('Final count top:', count_top/len(queryList))

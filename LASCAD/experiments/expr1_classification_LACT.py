import pandas as pd
import numpy as np
import traceback
from LASCAD.LDA.Clustering import testClustering
import os


if __name__ == '__main__':

    results = pd.DataFrame(columns=['Dataset', 'n_clusters', 'NUM_TOPICS', 'max_df', 'min_df',
                                    'precision', 'recall', 'f-score'])
    i = 0
    # dataset = 'LACT41'
    # dataset = 'LACT43'
    dataset = 'showcase_noStem2'
    # dataset = 'largeDataset'

    method = 'LACT'
    for n_clusters in range(5, 120, 5): # [6, 10, 20, 30, 40, 50, 60, 70]:
        print('n_clusters', n_clusters)
        for NUM_TOPICS in range(20, 110, 10):
            for max_df in [.8]:
                for min_df in [.2]:
                    print('{}- Running: NUM_TOPICS={}, max_df={}, min_df={}, test={}'
                          .format(i, NUM_TOPICS, max_df, min_df, dataset))
                    try:
                        score, n_clusters = testClustering(NUM_TOPICS=NUM_TOPICS, max_df=max_df,
                                                   min_df=min_df, dataset=dataset,
                                                   verbose=False,
                                                   plot_heatmap=False,
                                                   categ_method=method,
                                                   n_clusters=n_clusters
                                               )
                        score = np.round(np.array(score)*100., 2)
                        results.loc[i] = [dataset, n_clusters, NUM_TOPICS, max_df, min_df,
                                          score[0], score[1], score[2]]
                        results.to_csv(os.path.join('..', 'results', 'categorization_accuracy',
                                                    method + '_accuracy_scores_' + dataset + '.csv'))
                        i += 1
                    except:
                        print('n_clusters={}, NUM_TOPICS={}, max_df={}, min_df={}, test={} ..... failed'
                                    .format(n_clusters, NUM_TOPICS, max_df, min_df, dataset))
                        traceback.print_exc()

                    print('Done......')


    print(results)
    results.to_csv(os.path.join('..', 'results', 'categorization_accuracy',
                                'LACT_topics_fscore' + dataset + '.csv', index=False))

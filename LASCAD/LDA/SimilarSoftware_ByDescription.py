"""
    Given a software, find similar software using project Description
    Currently based on software name that exist in the dataset

"""


from LASCAD.LDA.Clustering import Clustering
import pandas as pd
import numpy as np
from whoosh.index import create_in
from whoosh.fields import Schema, ID, TEXT
from whoosh.qparser import QueryParser
from whoosh.analysis import StemmingAnalyzer
from time import time
import os
import json
from LASCAD.LDA.Clustering import base_dir
import urllib.request
import requests


class SimilarSoftware_ByDescription:

    def __init__(self, NUM_TOPICS=50, max_df=0.5, min_df=0.02, n_clusters=20, maxTopSimilar=100,
                                 dataset='showcase1', creat_schema=True):

        self.projects_details = pd.read_csv(os.path.join(base_dir, 'results/'
                                            + dataset + '/projects_details.csv'), index_col=None)
        self.maxTopSimilar = maxTopSimilar
        self.suffix = dataset + '_' + str(NUM_TOPICS) + '_' + str(max_df) + '_' + str(min_df)

        self.schema = Schema(name=TEXT(stored=True), description=TEXT(analyzer=StemmingAnalyzer()))
        self.ix = create_in("../results/whoosh", self.schema)
        writer = self.ix.writer()

        if creat_schema:
            print('Create woosh schema..')
            for i in self.projects_details.index:
                descript = self.projects_details.loc[i].description
                if not isinstance(descript, str):
                    descript = ''
                writer.add_document(name=self.projects_details.loc[i, 'name'],
                                    description=descript)

            writer.commit()
            print('Whoosh schema committed')

    # ------------------------------------------------------------------

    def getSimilarAppsForOneApp(self, i):

        return Exception

    # --------------------------------------------------------------------

    def getSimilarSoftwareApps(self, querySoftware, topSimilar=5):
        """
        :param querySoftware: can be project name or project index
        :param topSimilar: number of returned similar apps
        :return: list of tuples: [(similarity_score, similar_app_name), ...]
        """

        with self.ix.searcher() as searcher:
            queryText = querySoftware
            if not isinstance(queryText, str):
                queryText = self.projects_details.loc[queryText, 'name']
                if not isinstance(queryText, str):
                    queryText = ''

            query = QueryParser("description", self.ix.schema).parse(queryText)
            results = searcher.search(query, limit=topSimilar+1)
            results_list = [self.projects_details[self.projects_details.name == result.values()[0]].index[0]
                            for result in results]
            if querySoftware in results_list:
                results_list.remove(querySoftware)

        return results_list[:topSimilar]


    def github_login(self):

        api_token = '3278f073f78eccdef6e0aefa66bb52e673773d35'
        headers = {'Authorization': 'token {0}'.format(api_token)}

        res = requests.get(
            url='https://api.github.com/users/doaa-altarawy/repos',
            headers=headers,
        )
        print('Auth: ', res.ok)


    def get_encoded_readme(self):
        url_1 = 'https://api.github.com/repos/'
        project = 'pallets/flask'
        url_2 = '/readme'
        size_gt_0 = 0
        not_found = 0
        readme = pd.DataFrame(columns=['size', 'reame_encoded'], index=self.projects_details.index)
        self.github_login()

        for i in self.projects_details.index:
            project = self.projects_details.loc[i].full_name
            print(project)
            url = url_1 + project + url_2
            print(url)
            try:
                with urllib.request.urlopen(url) as response:
                    data = response.read()
                    print(data.size)
                    readme.iloc[i] = [data.size, data.content]
                    if data.size > 0:
                        size_gt_0 += 1
            except:
                print('exception...')
                print(data)
                not_found += 1
                readme.iloc[i] = [0, '']

        print('Not found ', not_found)
        print('Size > 0: ', size_gt_0)

        return size_gt_0

    def get_readme(self):

        url_1 = 'https://raw.githubusercontent.com/'
        project = 'pallets/flask'
        url_2 = ['/master/README', '/master/readme', '/master/Readme']
        exts = ['', '.md', '.txt', '.rst']
        readme = pd.DataFrame(columns=['full_name', 'size', 'readme'], index=self.projects_details.index)
        not_found = 0
        size_gt_0 = 0

        for i in self.projects_details.index:
            project = self.projects_details.loc[i].full_name
            print('Curr project: ', project)

            found = False
            for url in url_2:
                for ext in exts:
                    full_url = url_1 + project + url + ext
                    print(full_url)
                    try:
                        with urllib.request.urlopen(full_url) as response:
                            html = response.read()
                            print('found with length: ', len(html))
                            readme.iloc[i] = [project, len(html), html]
                            if len(html) > 0:
                                size_gt_0 += 1
                            found = True
                            break
                    except:
                        print('exception.')
                if found:
                    break

            if not found:
                print('Not found.')
                not_found += 1
                readme.iloc[i] = [project, 0, '']

        print(os.getcwd())
        readme.to_csv('readme_files.csv')

        print('Not found ', not_found)
        print('size > 0: ', size_gt_0)


    # -------------------------------------------------------------------


# print('Init engine...')
# engine = SimilarSoftware_ByDescription(NUM_TOPICS=50, max_df=0.5,
#                                        min_df=0.02, n_clusters=20,
#                                        dataset='largeDataset')

# engine.get_readme()


# df = pd.read_csv('readme_files.csv', index_col=0)
#
# print(df.shape)
#
# print(df[df['size'] > 0].shape)
# print('Size=0', df[df['size'] == 0].shape)



# results = engine.getSimilarSoftwareApps('flask', 20)
# print(results)
# cols = ['name', 'language', 'group', 'description']
# print(engine.projects_details.loc[results, cols])

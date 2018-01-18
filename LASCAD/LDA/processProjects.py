"""
    For Times-series projects:
    - Checkout a list of projects ()
    - Create tages and checkout tags
    - Process source code files into one .out file for each project

    For categories projects:
    - Checkout a list of projects (showcases_config.json)
    - Process source code files into one .out file for each project

    Processing runs in parallel.

"""

from __future__ import print_function
from datetime import datetime
# from git import Repo, Git
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
import os.path
import re, string, ntpath, json, codecs
import threading
import shutil, errno
import numpy as np
import pandas as pd
from LASCAD.config.languageConst import *
import multiprocessing, traceback
import logging, sys


base_dir = '/home/hshahin/workspaces/seclassification/'


def load_config(config_file):
    with open(os.path.join(base_dir, config_file)) as data_file:
        config = json.load(data_file)
    return config


config_files = load_config('config/const.json')

base_dir = config_files['base_dir']
source_dir = os.path.join(base_dir, config_files['source_dir'])
out_dir = os.path.join(base_dir, config_files['out_dir'])

# ----------- Logging --------------------------

fh = logging.FileHandler(filename = os.path.join(base_dir , config_files['log_file']))
fh.setLevel(logging.INFO)
logger = logging.getLogger()
logger.addHandler(fh)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------


# --------------------------------------------------------------------

config_timeseries_data = load_config(os.path.join(base_dir , config_files['dataset_timeseries']))
EXTENTIONS = load_config(os.path.join(base_dir , config_files['lang_extensions']))

# split camel case tokens
_underscorer1 = re.compile(r'(.)([A-Z][a-z]+)')
_underscorer2 = re.compile('([a-z0-9])([A-Z])')

# -------------------------------------------------------------------

def copy_folder(src, dst):
    try:
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

# --------------------------------------------------------------------

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


# ********************** Creating tags functions ***************************
# **************************************************************************

def get_date_time(epoch):
    '''
    convert epoch to date_time
    '''
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch))

# --------------------------------------------------------------------

def tag_exists(path, tag_name):
    repo = Repo(path)

    return True if tag_name in repo.tags else False

# --------------------------------------------------------------------

def get_epoch(year, month='01'):
    """
    calculate the epoch of first day of a year-month
    """
    pattern = '%Y.%m.%d %H:%M:%S'
    return int(time.mktime(time.strptime(str(year) + '.' + str(month) + '.01 00:00:00', pattern)))

# --------------------------------------------------------------------

def create_tags(path):
    '''
    takes repo path and creates tags for first commit in Jan and Jun. for every year
    # get the list of commits
    # get the latest commit date
    # current_year is the year from that date
    # loop through the list of commit to find the commit having a date equal or just after 1/1/current_year
    # once found create a tage with the current_year name on it AND
    # subtract 1 from the year and continue.

    '''
    repo = Repo(path)

    # get the list of commits
    commits = list(repo.iter_commits())

    # get the latest commit date, current_year is the year from that date
    current_year = datetime.fromtimestamp(commits[0].committed_date).year


    for idx, commit in enumerate(commits):
        # time.sleep(2)
        # print(commits[idx].hexsha)

        current_year_01 = str(current_year)+'-01'
        current_year_06 = str(current_year)+'-06'

        try:
            if get_epoch(current_year, '01') > commit.committed_date and \
                    int(time.time()) > get_epoch(current_year, '01')  and \
                    idx !=0:
                if str(current_year_01) not in repo.tags and idx != 0:
                    print(commits[idx-1].hexsha+' '+get_date_time(commits[idx-1].committed_date)+' '+current_year_01)
                    past = repo.create_tag(current_year_01, ref=commits[idx-1],
                                      message="This is a tag to mark the first commit in year %s" % current_year_01)
                current_year = datetime.fromtimestamp(commit.committed_date).year

            if get_epoch(current_year, '06') > commit.committed_date and \
                    int(time.time()) > get_epoch(current_year, '06') and \
                idx != 0:
                if str(current_year_06) not in repo.tags:
                    print(commits[idx-1].hexsha+' '+get_date_time(commits[idx-1].committed_date)+' '+current_year_06)
                    past = repo.create_tag(current_year_06, ref=commits[idx-1],
                                      message="This is a tag to mark the first commit in year %s" % current_year_06)
        except AttributeError:
            pass

# --------------------------------------------------------------------

def checkout_tag(path, tag_name):
    '''
    checks out a tag if it exists
    '''
    repo = Repo(path)
    git = Git(path)
    if tag_name in repo.tags:
        git.checkout(tag_name)

# --------------------------------------------------------------------

def delete_tags(path):
    '''
    remove all tags in a given repo
    '''

    repo = Repo(path)
    for tag in repo.tags:
        repo.delete_tag(tag)


# *************  Create tags every 6 months for each repo ******************
# **************************************************************************

def create6mothTags():
    for project_name, project_type in config_timeseries_data.items():
        print("Processing project: " + project_name )
        t0 = time.time()
        delete_tags(os.path.join(source_dir, project_name))
        create_tags(os.path.join(source_dir, project_name))
        print("Project: " + project_name + " taged in %0.3fs." % (time.time() - t0))

# create6mothTags()


# *************************  Preprocesing functions  ***********************
# **************************************************************************

def path_leaf(path):
    head, tail = ntpath.split(path)
    return head, tail

# --------------------------------------------------------------------

def camel_to_spaces(s):
    """
    convert camel case into spaces seperated
    """
    subbed = _underscorer1.sub(r'\1 \2', s)
    return _underscorer2.sub(r'\1 \2', subbed).lower()

# --------------------------------------------------------------------

def snake_to_spaces(snake_cased_str):
    """
    convert snake case into spaces seperated
    """
    separator = "_"
    components = snake_cased_str.split(separator)
    if components[0] == "":
        components = components[1:]
    if components[-1] == "":
        components = components[:-1]
    if len(components) > 1:
        spaced_str = components[0].lower()
        for x in components[1:]:
            spaced_str += " " + x.lower()
    else:
        spaced_str = components[0]
    return spaced_str

# --------------------------------------------------------------------

def file_preprocessing(input_file, output_file):
    """
    - replace punctuations with spaces
    - stemming
    - camel to spaces and snake to spaces
    - remove language spesific keywords
    - write the entire project snapshot into one file under project root folder
    """
    # print("processing file " + input_file)
    # replace the punctuations with space
    replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    # stemming
    stemmer = PorterStemmer()

    with open(input_file, 'r', encoding='utf-8', errors='replace') as inFile, open(output_file,'w') as outFile:
        for line in inFile:
            # replace punctuations
            # convert camel case into space separated
            # convert snake case into space separated
            # remove language keywords
            line_witout_puncs = ' '.join([snake_to_spaces(camel_to_spaces(word))
                                          for word in line.translate(replace_punctuation).split()
                                          if len(word) >=4 and word not in stopwords.words('english')
                                          and word not in all_keywords])

            # stemming
            # singles = []
            # for plural in line_witout_puncs.split():
            #     try:
            #         singles.append(stemmer.stem(plural))
            #     except UnicodeDecodeError:
            #         print(plural)

            # line_stemmed = ' '.join(singles)
            # print(line_stemmed, file=outFile)
            print(line_witout_puncs, file=outFile)

# --------------------------------------------------------------------

def return_file_type(project_path, file_type):
    if '.proc' in file_type:
        exten = EXTENTIONS[file_type.split('.')[0]]
        if type(exten) is list:
            extenstion = tuple(i+'.proc' for i in exten)
        else:
            extenstion = exten + '.proc'
    elif type(EXTENTIONS[file_type]) is list:
        extenstion = tuple(EXTENTIONS[file_type])
    else:
        extenstion = EXTENTIONS[file_type]

    project_files = [os.path.join(root, name)
             for root, dirs, files in os.walk(project_path)
             for name in files
             if name.endswith(extenstion)]
    return project_files

# --------------------------------------------------------------------

def project_preprocessing(project_path, file_type, project_name):
    # print ("processing project "+ project_path)
    # process project source code files and save each file as *.proc
    project_files = return_file_type(project_path, file_type)
    for source_file in project_files:
        head, tail = path_leaf(source_file)
        proc_file = os.path.join(head , tail + '.proc')
        file_preprocessing(source_file, proc_file)

    if not os.path.exists(os.path.join(out_dir, project_name.lower())):
        os.makedirs(os.path.join(out_dir, project_name.lower()))

    # concatenate all processed project files into one file under out directory
    project_proc_files = return_file_type(project_path, file_type + '.proc')
    with open(os.path.join(out_dir, project_name.lower(), config_files['final_processed']), 'w') as outfile:
        for fname in project_proc_files:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

# ******************* checkout tags in separate folders *******************
# *************************************************************************


def checkout_projects_tags():
    """ For times-series data:

        - create folder project_tags
        - for each tag if tag exists
        - copy the project into project_tag/tag_name
        - checkout ptoject to tag_name
        - delete .git folder
    :return:
    """
    for project_name, project_type in config_timeseries_data.items():
        project_name = project_name.lower()
        project_path = os.path.join(source_dir, project_name)
        project_tags_path = project_path + '-tags'


        if not os.path.exists(project_tags_path):
            os.makedirs(project_tags_path)

        repo = Repo(project_path)
        for tag_name in tag_names:
            if tag_exists(project_path, tag_name):
                print("Copying "+project_name+' '+tag_name)
                current_tag_path = os.path.join(project_tags_path, tag_name)
                copy_folder(project_path, current_tag_path)

        for tag_name in tag_names:
            if tag_exists(project_path, tag_name):
                print("Checkout "+project_name+' '+tag_name)
                current_tag_path = os.path.join(project_tags_path, tag_name)
                checkout_tag(current_tag_path, tag_name)

        for tag_name in tag_names:
            if tag_exists(project_path, tag_name):
                print("deleting .git "+project_name+' '+tag_name)
                current_tag_path = os.path.join(project_tags_path, tag_name)
                os.chdir(current_tag_path)
                shutil.rmtree(os.path.join(current_tag_path, '.git'))


# ******************* checkout Recent projects (no tags) *******************
# *************************************************************************

def get_projectsList(dataset):
    '''
    read projects name/type/group
    :param dataset: 'showcases' or 'largeDataset'
    :return: projects_list
    '''

    if dataset == 'showcases':
        # For categories
        projects_list = load_config(os.path.join(base_dir, config_files[dataset + '_data']))
        projects_list = pd.DataFrame(projects_list).T
        projects_list.reset_index(inplace=True)
        projects_list.rename(columns={'index': 'name'}, inplace=True)

    elif dataset == 'largeDataset':
        projects_list = pd.read_csv(os.path.join(base_dir, config_files[dataset + '_data']))


    return projects_list

# ------------------------------------------------------------------------------

def clone_project(git_fullName, project_path):
    # TODO: clone project from git_fullName to the project_path folder

    pass


def clone_projects(dataset='showcases'):
    """
    - for each project_name:
        - create folder project_name
        - clone the project into project_name/
        - delete .git folder
    :return:
    """

    projects_list = get_projectsList(dataset)

    for index in projects_list.index:
        project_name = projects_list.iloc[index]['name']
        project_path = os.path.join(source_dir,
                                    projects_list.iloc[index]['name'].lower())
        git_fullName = projects_list.iloc[index]['full_name']

        if not os.path.exists(project_path):
            os.makedirs(project_path)

        # TODO: write the clone function
        try:
            print("Cloning project: ", project_name)
            clone_project(git_fullName, project_path)
        except:
            print('ERROR: cloning project {} failed..'.format(project_name))
            logging.error(traceback.print_exc())

        try:
            print("deleting .git of ", project_name)
            os.chdir(project_path)
            shutil.rmtree(os.path.join(project_path, '.git'))
        except:
            print('ERROR: deleting .git of project {} failed..'.format(project_name))
            logging.error(traceback.print_exc())


# ********************* Run preprocessing **********************
# **************************************************************

def run_preprocessing_tags(project_source_dir):
    project_tags_path = os.path.join(source_dir , project_source_dir)
    logger.info('--------------- %', project_tags_path)

    for project_tag in get_immediate_subdirectories(project_tags_path):
        project_tag_path = os.path.join(project_tags_path , project_tag)
        t0 = time.time()
        project_preprocessing(project_tag_path, project_type_map[project_source_dir], project_tag)
        logger.info("processing project: {} \t tag {} done in {} mins".format(project_tags_path,
                                    project_tag, (time.time() - t0)/60.))
    logger.info('****This thread is done: {}'.format(os.getpid()))

# --------------------------------------------------------------------

def run_preprocessing_project(project):
    ''' project = [name, group, language]
    '''
    if project[2] not in EXTENTIONS.keys():
        logger.info('skipping project {} of language {}.'.format(project[0], project[2]))

    logger.info(project)
    project_path = os.path.join(source_dir , project[0].lower())

    t0 = time.time()
    try:
        # process all possible files that could have a code (any code type)
        project_preprocessing(project_path, "ALL", project[0])
        # project_preprocessing(project_path, project[2], project[0])
    except:
        logger.info("Error in project {}".format(project))
        logging.error(traceback.print_exc())

    logger.info("***Processing project:{} done in {}min.".format(project_path, (time.time() - t0)/60.0))

# --------------------------------------------------------------------

def run_preprocessing_categories(dataset):
    # For categories
    projects_list = get_projectsList(dataset)
    projects_list = projects_list[['name', 'group', 'language']].as_matrix()
    logger.info('Start time: {}'.format(time.time()))
    logger.info('all projects: {}'.format(projects_list.shape))

    print(projects_list)
    pool = multiprocessing.Pool(8)
    pool.map(run_preprocessing_project, projects_list)

    logger.info('Main process Done..........................')

# ---------------------------------------------------------

def run_preprocessing_time_series():

    logger.info('Start time: {}'.format(time.time()))

    #----------------- For tags
    project_tags_paths = get_immediate_subdirectories(source_dir)
    pool = multiprocessing.Pool(16)
    pool.map(run_preprocessing_tags, project_tags_paths)


    logger.info('Main process Done..........................')

# --------------------------------------------------------------------

if __name__=="__main__":
    # For categories
    dataset = 'showcases'
    # dataset = 'largeDataset'
    # clone_projects(dataset)
    run_preprocessing_categories(dataset)

    # # For time-series  <---- Not used
    # create6mothTags()
    # checkout_projects()
    # run_preprocessing_time_series()

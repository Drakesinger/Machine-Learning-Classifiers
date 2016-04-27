# -*- encoding: utf-8 -*-

import os
import codecs

from threading import Thread
from random import shuffle
from random import randint

from shutil import rmtree
from async import print_progress

__author__ = 'Horia Mut'

unprocessed_assets_directory = os.getcwd() + "/Assets/"
processed_files_directory = os.getcwd() + "/Processed/"


# The categories available.
categories = ['pos', 'neg']

default_part_of_speech = ['NOM', 'ADV', 'VER', 'ADJ', 'PRP', 'KON', 'PRO', 'ABR']


def start_preprocessing(part_of_speech=None):
    '''
    Start the preprocessing and generate the required folders.
    :return: does not return anything.
    '''

    # Delete processed folder if it exists
    print "Deleting previous processed data folder."
    if os.path.exists(processed_files_directory):
        rmtree(processed_files_directory, True)

    print "Building pre-processed data."
    if part_of_speech:
        global default_part_of_speech
        default_part_of_speech = part_of_speech
    print "Part of Speech used:" + str(default_part_of_speech)

    for text_type in categories:
        assets_folder_name = text_type + "/"
        process_folder(assets_folder_name, text_type, part_of_speech=default_part_of_speech)

    global is_stop_requested
    is_stop_requested = True

    print "Pre-processing is done. Data is ready to work with."


def define_part_of_speech(pos):
    # The word types we want to extract and study.
    global default_part_of_speech
    default_part_of_speech = pos


def process_folder(folder_name, text_type, part_of_speech=None):
    '''
    Here we will reprocess our data before working with it.
    :param folder_name:
    :return:
    '''
    lower_bound = 0
    upper_bound = 1000
    limit = 200

    index = 0
    number_of_files = len(os.listdir(unprocessed_assets_directory + folder_name))
    print_progress(index, number_of_files, prefix='Progress:', suffix='Complete', barLength=50)

    for filename in os.listdir(unprocessed_assets_directory + folder_name):
        words_in_the_file = process_file(folder_name, filename, part_of_speech=part_of_speech)
        shuffle(words_in_the_file)

        index += 1
        print_progress(index, number_of_files, prefix='Progress:', suffix='Complete', barLength=50)

        # Decide if the file will be used for testing or training.
        choice = randint(lower_bound, upper_bound)
        if choice <= limit:
            # Test file
            make_processed_file(filename, text_type, words_in_the_file, type='test')
        else:
            # Training file
            make_processed_file(filename, text_type, words_in_the_file, type='train')


def process_file(assets_folder_name, filename, part_of_speech=None):
    '''

    :param filename:
    :return:
    '''
    working_file = codecs.open(unprocessed_assets_directory + assets_folder_name + filename, 'r', 'utf-8', buffering=1)
    file_words = []
    for line in working_file:
        words = line.split("\t")
        word = None

        try:
            if part_of_speech:
                if words[1] in part_of_speech:
                    word = (words[2]).rstrip()
            else:
                word = (words[2]).rstrip()
        except UnicodeDecodeError:
            word = words[2].rstrip()
        except IndexError:
            pass

        if word:
            file_words.append(word)

    working_file.close()
    return file_words


def make_processed_file(filename, text_type, words_in_the_file, type='', part_of_speech=''):
    type_path = '/' + type + '/'
    part_of_speech = '/' + part_of_speech + '/'
    text_sentiment = '/' + text_type + '/'

    if not os.path.exists(processed_files_directory + type_path + text_sentiment + part_of_speech):
        os.makedirs(processed_files_directory + type_path + text_sentiment + part_of_speech)

    # Disgusting code.
    processed_file = codecs.open(processed_files_directory + type_path
                                 + text_sentiment
                                 + part_of_speech
                                 + filename, 'w+', 'utf-8',
                                 buffering=1)

    for ustring in words_in_the_file:
        processed_file.write(ustring)
        processed_file.write('\n')

    processed_file.close()

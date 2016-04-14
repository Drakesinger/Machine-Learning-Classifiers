# -*- encoding: utf-8 -*-

__author__ = 'horia_000'

import os
import codecs
from random import shuffle

current_working_directory = os.getcwd() + "/Assets/"
processed_files_directory = os.getcwd() + "/Processed/"

def process_folder(folder_name):
    '''
    Here we will reprocess our data before working with it.
    :param folder_name:
    :return:
    '''
    for filename in os.listdir(current_working_directory + folder_name):
        words_in_the_file = process_file(filename)
        shuffle(words_in_the_file)

        make_processed_file(filename, words_in_the_file)


def make_processed_file(filename, words_in_the_file):
    if not os.path.exists(processed_files_directory):
        os.mkdir(processed_files_directory)
    processed_file = codecs.open(processed_files_directory + filename, 'w+', 'utf-8', buffering=1)
    for ustring in words_in_the_file:
        # print ustring.encode('utf-8'),ustring
        processed_file.write(ustring)
        processed_file.write('\n')
    processed_file.close()


def process_file(filename):
    '''

    :param filename:
    :return:
    '''
    working_file = codecs.open(current_working_directory + assets_folder_name + filename, 'r', 'utf-8',buffering=1)
    file_words = []
    for line in working_file:
        words = line.split("\t")
        word = ''

        try:
            word = (words[2]).rstrip()
            # print "unicode:",word
        except UnicodeDecodeError:
            word = words[2].rstrip()
            #print "word:", word
        except IndexError:
            pass #print line, filename
        file_words.append(word)
    working_file.close()
    return file_words



if __name__ == '__main__':
    folder_name_neg = "neg"
    folder_name_pos = "pos"

    assets_folder_name = folder_name_neg + "/"
    process_folder(assets_folder_name)

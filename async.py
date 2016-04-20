from multiprocessing import Process, Queue

__author__ = 'Horia Mut'


def print_progress(iteration, total, prefix='', suffix='', decimals=2, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
    @source:
        http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    filledLength = int(round(barLength * iteration / float(total)))
    percents = round(100.00 * (iteration / float(total)), decimals)
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        print("\n")


import time
import random

import sys


# a downloading thread
def worker(path, total, queue):
    size = 0
    while size < total:
        dt = random.randint(1, 3)
        time.sleep(dt)
        ds = random.randint(1, 5)
        size = size + ds
        if size > total: size = total
        queue.put(("update", path, total, size))
    queue.put(("done", path))


# the reporting thread
def reporter(queue, number_of_worker_threads):
    status = {}
    while number_of_worker_threads > 0:
        message = queue.get()
        if message[0] == "update":
            path, total, size = message[1:]
            status[path] = (total, size)
            # update the screen here
            show_progress(status)
        elif message[0] == "done":
            number_of_worker_threads = number_of_worker_threads - 1
    print ""


def show_progress(status):
    line = ""
    for path in status:
        (total, size) = status[path]
        line = line + "%s: %3d/%d   " % (path, size, total)
    sys.stdout.write("\r" + line)
    sys.stdout.flush()


def example_usage():
    '''
    http://stackoverflow.com/questions/28057445/python-threading-multiline-progress-report
    :return:
    '''
    q = Queue()
    w1 = Process(target=worker, args=("abc", 30, q))
    w2 = Process(target=worker, args=("foobar", 25, q))
    w3 = Process(target=worker, args=("bazquux", 16, q))
    r = Process(target=reporter, args=(q, 3))
    for t in [w1, w2, w3, r]:
        t.start()
    for t in [w1, w2, w3, r]:
        t.join()


if __name__ == "__main__":
    example_usage()

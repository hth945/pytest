import os
import functools
import time
from glob import glob
from fire import Fire
from tqdm import tqdm
from multiprocessing import Pool

def worker(s):
    time.sleep(0.1)
    # print(s)
    return s

def create_lmdb( num_threads):
    video_names=[1,2,3]
    with Pool(processes=num_threads) as pool:
        # for ret in tqdm(pool.imap_unordered(functools.partial(worker), video_names), total=len(video_names)):
        #     print(ret)

        for ret in pool.imap_unordered(functools.partial(worker), video_names):
            time.sleep(1)
            print('ret')
            print(ret)

        # for ret in pool.imap_unordered(worker, (video_names,)):
        #     print('ret')
        #     print(ret)

if __name__=="__main__":  # 必须使用(在多进程中)
    create_lmdb(2)
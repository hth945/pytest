

from multiprocessing import Pool
import time

def worker(s):
    time.sleep(0.1)
    print(s)
    return s


# def test(p):
#        print(p)
#        time.sleep(1)
# if __name__=="__main__":
#     pool = Pool(processes=2)
#     for i  in range(50):
#         pool.apply_async(worker, args=(i,))   #维持执行的进程总数为10，当一个进程执行完后启动一个新进程.
#     print('test')
#     pool.close()
#     pool.join()


#
# exit()

import functools

from tqdm import tqdm
from multiprocessing import Pool
import time

def worker(s):
    time.sleep(1)
    print(s)
    return s

# def create_lmdb( num_threads):
#     video_names=['1','2','3']
#     with Pool(processes=num_threads) as pool:
#         for ret in tqdm(pool.imap_unordered(functools.partial(worker), video_names), total=len(video_names)):
#             print(ret)
def create_lmdb( num_threads):
    video_names=[1,2,3]
    with Pool(processes=num_threads) as pool:
        for item in video_names:
            ret = pool.apply(worker,args=(item,))
            print(ret)
if __name__=="__main__":  # 必须使用(在多进程中)
    create_lmdb(1)
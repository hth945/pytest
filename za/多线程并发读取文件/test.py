#%%

import concurrent.futures
import glob
import time
n = 0
def load_and_resize(image_filename):
    global n
    n += 1
    print(image_filename)
    time.sleep(1)
    pass
    # img = cv2.imread(image_filename)
    # img = cv2.resize(img, (600, 600))

start_time1 = time.time()
# with concurrent.futures.ProcessPoolExecutor() as executor: ## 默认为1
#     path = r'..\..\dataAndModel\data\酒瓶瑕疵检测\datalab2\images'
#     image_files = glob.glob(path)
#     executor.map(load_and_resize, image_files)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor: ## 默认为1
    path = r'..\..\dataAndModel\data\酒瓶瑕疵检测\datalab2\images\*.jpg'
    
    image_files = glob.glob(path)
    print(len(image_files))
    executor.map(load_and_resize, image_files)


print('多核并行加速后运行 time:', round(time.time() - start_time1, 2), " 秒")
print(n)


# %%

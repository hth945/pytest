#%%
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn Training')
    parser.add_argument('--data_path', default='/public/yzy/coco/2017/', help='dataset path')
    parser.add_argument("--bool",help = "Whether to pirnt sth.")
    args = parser.parse_args(['--data_path', '123'])
    return args


def main():
    # print('1')
    args = get_args()
    print('2')
    print(args.bool)
    print(args.data_path)
main()



# %%

jupyter notebook --generate-config
#  /root/.jupyter/jupyter_notebook_config.py



c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 65500
c.NotebookApp.allow_root = True
c.NotebookApp.open_browser = False
c.NotebookApp.password = 'sha1:8e4f16cf0aa6:30644b99b9930e0179359266c359461493c2cec5' #填入刚刚复制的字符
c.NotebookApp.notebook_dir = '/'

conda activate tensorflow_2.1_cu10.1_py36
nohup  jupyter lab > /dev/null 2>&1 &
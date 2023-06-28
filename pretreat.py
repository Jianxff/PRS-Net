from src.polygon import Polygon
import threading
import time


class Pretreator:
  r""" Pretreator for Polygon

  Polygon预处理器
  多线程处理 ShapeNetCore.v2 模型数据集
  """

  models = []
  def __init__(self, index_files: list):
    r""" Init Pretreator

    载入索引 
    """
    self.index_files = index_files
    for f in index_files:
      with open(f, 'r') as fs:
        for line in fs.readlines():
          id, obj, out = line.split()
          self.models.append((id, obj, out))

  def process(self, thread_num = 1):
    r"""
    多线程处理
    .. note:
      - step: 每个线程处理的模型数量 
    """
    step = len(self.models) // thread_num
    for i in range(thread_num):
      if i == thread_num - 1:
        threading.Thread(target=self.__process_thread__, args=(i, self.models[i*step:])).start()
      else:
        threading.Thread(target=self.__process_thread__, args=(i, self.models[i*step:(i+1)*step])).start()

  @staticmethod
  def __process_thread__(label, models):
    r""" Process thread
    处理线程
    """
    def log_failed(id,path,out):
      r""" Log failed models
      记录处理失败的模型
      """
      with open(f'/root/autodl-tmp/shapenet.{label}.failed', 'a') as f:
        f.write(f'{id} {path} {out}\n')

    cnt = 0
    all = len(models)
    start_tm = time.time()
    for (id, path, out) in models:
      print(f'thread [{label}]: {all - cnt} models left')
      cnt += 1
      try:
        p = Polygon(id)
        if not p.process(path, rand_rotate=0.2): # 预处理
          log_failed(id, path, out)
        p.dump(out) # 保存
      except Exception as e:  #
        print(f'thread [{label}]: {id} failed, {str(e)}') 
        log_failed(id, path, out)
        continue
    print(f'thread [{label}]: process {cnt} models in {time.time() - start_tm} seconds')


indexs = ['/root/autodl-tmp/shapenet.train', '/root/autodl-tmp/shapenet.test']
pretreator = Pretreator(indexs)
pretreator.process(thread_num=3)
from polygon import Polygon
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
          id, obj, res = line.split()
          self.models.append((id, obj))

  def process(self, thread_num = 1):
    step = len(self.models) // thread_num
    for i in range(thread_num):
      if i == thread_num - 1:
        threading.Thread(target=self.__process_thread__, args=(i, self.models[i*step:])).start()
      else:
        threading.Thread(target=self.__process_thread__, args=(i, self.models[i*step:(i+1)*step])).start()

  @staticmethod
  def __process_thread__(label, models):
    cnt = 0
    all = len(models)
    start_tm = time.time()
    for (id, path) in models:
      print(f'thread [{label}]: {all - cnt} models left')
      cnt += 1
      try:
        p = Polygon(id)
        p.process(path, rand_rotate = True) # 预处理
        p.dump(output_path + '/' + id + '.mat') # 保存
      except Exception as e:
        print(e)
        continue
    end_tm = time.time()
    print(f'thread [{label}]: process {cnt} models in {end_tm - start_tm} seconds')


indexs = ['/root/autodl-tmp/ShapeNetCore.v2.train', '/root/autodl-tmp/ShapeNetCore.v2.test']
pretreator = Pretreator(indexs)
pretreator.process(thread_num=6)
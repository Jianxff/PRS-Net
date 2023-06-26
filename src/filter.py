import os
import random

class ShapeNetIndex:
  index_dir: os.path
  result_dir: os.path
  dataset: list
  data_train: list
  data_test: list

  def __init__(self, index_dir, result_dir):
    self.index_dir = index_dir
    self.result_dir = result_dir
    self.dataset = []
    self.data_test = []
    self.data_train = []

  def filter(self, dataset_path):
    cnt = 0
    for root, _, files in os.walk(dataset_path):
      for f in files:
        if f.endswith('.obj'):
          path = os.path.join(root, f)
          id = path.split('/')[-3]
          self.dataset.append((id, path))
          cnt += 1
    print(f'filter {cnt} models')

  def split(self,test_proportion = 0.1):
    random.shuffle(self.dataset)
    test_num = int(len(self.dataset) * test_proportion)
    self.data_test = self.dataset[:test_num]
    self.data_train = self.dataset[test_num:]
    print(f'split {len(self.data_train)} train models and {len(self.data_test)} test models')
  
  def dump(self):
    train_path = os.path.join(self.index_dir, 'ShapeNetCore.v2.train')
    test_path = os.path.join(self.index_dir, 'ShapeNetCore.v2.test')
    
    with open(train_path, 'w') as f:
      for id, path in self.data_train:
        f.write(f'{id} {path} {self.result_dir}/{id}.mat\n')
    with open(test_path, 'w') as f:
      for id, path in self.data_test:
        f.write(f'{id} {path} {self.result_dir}/{id}.mat\n')


# # 武器, 乐器, 汽车
# exclude_label = ['04090263','03467517', '02958343']

dataset_path = '/root/autodl-tmp/ShapeNetCore.v2'
shapenet = ShapeNetIndex(index_dir='/root/autodl-tmp',
                         result_dir='/root/autodl-tmp/ShapeNetCore.v2-RS')
shapenet.filter(dataset_path)
shapenet.split(test_proportion=0.1)
shapenet.dump()
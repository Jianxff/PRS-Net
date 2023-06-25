from polygon import Polygon
import threading
import os

exclude_label = ['04090263','03467517']
dataset_path = '/root/autodl-tmp/ShapeNetCore.v2'
output_path = '/root/autodl-tmp/ShapeNetCore.v2-PT'
index_path = '/root/autodl-tmp/ShapeNetCore.v2-index.txt'

def filter_files(dataset_path, index_path):
  os.system(f'rm -rf {output_path}/*')

  fs = open(index_path, 'w')
  cnt = 0
  for root, _, files in os.walk(dataset_path): 
    for f in files:
      if f.endswith(".obj"):
        path = os.path.join(root, f)
        info = path.split('/')
        try:
          id = info[-3]
          label = info[-4]
          if label in exclude_label:
            continue
        except:
          continue
        fs.write(f'{id} {path}\n')
        cnt += 1
  fs.close()    
  print(f'extracted {cnt} models')


def pre_process(index_path,limit = -1, thread_num = 1):
  print(f'pre-processing {str(limit) if limit > 0 else "all"} models in {thread_num} threads')
  models = []
  with open(index_path, 'r') as f:
    for line in f.readlines():
      id, path = line.split()
      models.append((id,path))
      if len(models) >= limit:
        break
  
  step = len(models) // thread_num
  for i in range(thread_num):
    if i == thread_num - 1:
      threading.Thread(target=process_thread, args=(i, models[i*step:].copy())).start()
    else:
      threading.Thread(target=process_thread, args=(i, models[i*step:(i+1)*step].copy())).start()

def process_thread(label:int, models):
  for (id, path) in models:
    print(f'thread [{label}]: pre-processing {path}')
    p = Polygon()
    if not p.load_model(path):
      continue
    p.voxelize()
    p.compute_closests()
    p.dump(output_path + '/' + id + '.mat')
    print(f'thread [{label}]: finish on processing {id}')


if __name__ == '__main__':
  # filter_files(dataset_path, index_path)
  pre_process(index_path, limit=16, thread_num=2)
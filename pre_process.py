from polygon import Polygon

def filter_files(dataset_path):
  output = []
  temp = []
  for root, dirs, files in os.walk(dataset_path): 
    temp += (d for d in dirs)
    for f in files:
      if f.endswith(".obj"):
        output.append({
          'path':os.path.join(root, f),
          'id':str(temp[-2] if len(temp) > 0 else '')
        })
        temp.clear()
  return output

def pre_process(limit = -1):
  models = filter_files('/root/autodo-tmp/ShapeNetCore.v2')
  cnt = 0
  for m in models:
    if cnt >= limit and limit > 0:
      break
    p = Polygon(m['path'])
    p.voxelize()
    p.compute_closests()
    p.dump('/root/autodo-tmp/ShapeNetCore.v2-PT/' + m['id'] + '.mat')
    cnt += 1


pre_process(1)


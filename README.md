# IMPLEMENT for PRS-NET

Implementation for PRS-Net: Planar Reflective Symmetry Detection Net for 3D Models

### Usage
**NOTICE:** All `parameters` and `file path` should be set on source files. Setting with argument parser is on working.

- filter dataset and set proportion for spliting train and test
  ``` bash
  python utils/filter.py
  ```

- pre-processing original `ShapeNetCore.v2` dataset, including voxelization and compute closest points
  ``` bash
  python utils/pretreat.py
  ```

- train network
  ``` bash
  python train.py
  ```

- test network
  ``` bash
  python test.py
  ```


### Network Architect

![network](static/network.png)

### Pretreated DataSet
- compute closest points and run voxelization
![dataset](static/pretreated.png)

### Evaluation
```
So bad, working on it now... [Sad]
```
![ev1](static/evaluate_1.png)
![ev2](static/evaluate_2.png)
![ev3](static/evaluate_3.png)
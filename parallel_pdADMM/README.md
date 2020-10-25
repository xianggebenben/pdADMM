# Parallel pdADMM
## Packages required
- pyarrow
- tornado
- asyncio
- tensorflow
- keras

## How to run the code
*N+1* nodes is needed to run a *N-layer* model. We use *layer0*, *layer1*...*layerN* in *config.ini* to denote the *N+1* nodes. Before the training starts, the extra node *layer0* is used to generate and distribute the parameters for each layer, during the training process it is responsible for collecting required parameters from each layer and computing accuracy.
1. Modify *config.ini* on each node.
  - below is a example of *config.ini* for layer1
  ```
  [currentLayer]
  layer = 1 # which layer you want to run on current machine

  [common]
  total_layers = 10 # total number of layers
  iteration = 5     # number of iterations
  rho = 0.0001      
  seed_num = 100    # seed number
  neurons = 2000    # number of neurons
  plasma_path = /home/ec2-user/plasma  # modify ‘/home/ec2-user’ to an existing path
  platform = cpu   # cpu or gpu
  chunks = 10      # how many chunks do you want to split the parameters. We split the parameters into small chunks to speedup the communication between layers.

  [layer0]
  server = 172.31.94.188  # ip of layer0

  [layer1]
  server = 172.31.7.13  # ip of layer1

  [layer2]
  server = 172.31.13.223 # ip of layer2
  ```
2. On each node, run the following code. For the detailed use of Plasma, please visit https://arrow.apache.org/docs/python/plasma.html
```
plasma_store -m 30000000000 -s /home/ec2-user/plasma
```
3. On each node, run the following code.
```
python3 server_pdadmm.py
```
4. On *layer0*, run
```
python3 client_pdadmm.py
```


import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
# import matplotlib
# matplotlib.use("Agg")

def animate_gradient(Wnew, n, m, gradImgs):
  for i in range(n):
    m[:,:,i] = (Wnew[i,:]).reshape((SOMsize, SOMsize))
  img = plt.imshow(m, animated=True)
  gradImgs.append([img])
  return gradImgs

def animate_nodes(winningNodes, m, nodeImgs):
  print('number of winning nodes: ' + str(len(winningNodes)))
  winningPlot = np.zeros((m.shape))
  count = 0
  for (row, col) in winningNodes:
    count += 1
    winningPlot[row,col] = m[row,col]
    img = plt.imshow(winningPlot, animated=True)
    if (count % 30) == 0:
      nodeImgs.append([img])
  img = plt.imshow(winningPlot, animated=True)
  nodeImgs.append([img])
  return nodeImgs

def gen_som(X, Y, SOMsize, saveAnimation):
  fig = plt.figure()
  plt.axis('off')
  imgs = []
  (N, d) = X.shape
  n = d
  m = SOMsize ** 2
  Wnew = np.random.rand(n, m) * 255
  for i in range(m):
    Wnew[:,i] = Wnew[:,i] / np.linalg.norm(Wnew[:,i])
  winningNodes = set()
  error = np.Inf
  tolerance = 10e-8
  eta = 0.9
  sigma = SOMsize / 5
  alpha = 0.9999
  iter = 0
  a = np.zeros((SOMsize, SOMsize))
  Wold = np.zeros((n, m))
  p = np.zeros((SOMsize, SOMsize, 2))
  for i in range(SOMsize):
    for j in range(SOMsize):
      p[i,j] = [i, j]
  d = np.zeros((SOMsize, SOMsize, 2))
  u = np.zeros((SOMsize, SOMsize, 2))
  m = np.zeros((SOMsize, SOMsize, n))
  # while error>tolerance and eta>0.1 and sigma>1:
  for i in range(20000):
    iter = iter+1
    Wold = np.copy(Wnew)
    i = random.randint(0, N-1)
    xtrain = X[i,:] / np.linalg.norm(X[i,:])
    a = (np.matmul(xtrain, Wold)).reshape((SOMsize, SOMsize))
    [row, col] = np.unravel_index(np.argmax(a), a.shape)
    d = np.linalg.norm((p - np.asarray([row, col])), axis=2)
    u = np.exp((-1) * (d**2) / (2*(sigma**2)))
    Wnew = Wold + (eta*(u.flatten())) * (xtrain.reshape(4,1) - Wold)
    Wnew = Wnew / np.linalg.norm(Wnew, axis=0)
    eta = eta*alpha
    sigma = sigma*alpha 
    error = np.linalg.norm(Wnew[:,(SOMsize*(row)+col)] - Wold[:,(SOMsize*(row)+col)])
    if saveAnimation:
      imgs = animate_gradient(Wnew, n, m, imgs)
  print("error: %f" % error)
  print("eta: %f" % eta)
  print("sigma: %f" % sigma)
  for i in range(n):
    m[:,:,i] = (Wnew[i,:]).reshape((SOMsize, SOMsize))
  for i in range(N):
    xtrain = X[i,:] / np.linalg.norm(X[i,:])
    label = Y[i]
    a = (np.matmul(xtrain, Wnew)).reshape((SOMsize, SOMsize))
    [row, col] = np.unravel_index(np.argmax(a), a.shape)
    winningNodes.add((row, col, label))
  print('starting animation')
  if saveAnimation:
    imgs = animate_nodes(winningNodes, m, imgs)
  print('that took ' + str(iter) + ' iterations')
  return m, winningNodes, fig, imgs

def get_som_input(N):
  rows = N**3
  X = np.zeros((rows, 3))
  intensities = np.linspace(0,255,N)
  for r in range(N):
    for g in range(N):
      for b in range(N):
        X[(N**2)*r+N*g+b,:] = [intensities[r], intensities[g], intensities[b]]
  filtered = []
  for i in range(N**3):
    if (X[i,0] != X[i,1] or X[i,0] != X[i,2] or X[i,1] != X[i,2]):
      filtered.append(X[i])
  filtered = np.asarray(filtered)
  return filtered

def plot_som(som, winningNodes):
  rows, cols, features = som.shape
  # winningPlot = np.zeros((rows, cols, 3))
  print('len = ' + str(len(winningNodes)))
  for (row, col, label) in winningNodes:
    # winningPlot[row,col] = som[row,col]
    if label == 'Iris-setosa':
      training_sample = [1.0, 0, 0]
    elif label == 'Iris-virginica':
      training_sample = [0, 1.0, 0]
    elif label == 'Iris-versicolor':
      training_sample = [0, 0, 1.0]
    # winningPlot[row,col] = training_sample
    som[row,col] = training_sample
  # plot1 = plt.figure(1)
  # plt.imshow(winningPlot)

  plot2 = plt.figure(1)
  plt.imshow(som)
  plt.show()

def get_input():
  filename = "iris.data"
  with open(filename, "r") as f:
    content = f.readlines()
  content = [x.strip().split(",") for x in content] 
  X_train = []
  Y_train = []
  for input in content:
    if len(input) != 5:
      continue
    label = input.pop()
    features = list(map(float, input))
    X_train.append(np.asarray(features))
    Y_train.append(label)
  X_train = np.asarray(X_train)
  Y_train = np.asarray(Y_train)
  return X_train, Y_train

def get_euclidian_sum(curNode, som, row, col):
  euclidian_sum = 0
  total_adjacent_nodes = 0
  try:
    euclidian_sum += np.linalg.norm(som[row+1][col]-curNode)
    total_adjacent_nodes += 1
  except IndexError:
    pass
  try:
    euclidian_sum += np.linalg.norm(som[row-1][col]-curNode)
    total_adjacent_nodes += 1
  except IndexError:
    pass
  try:
    euclidian_sum += np.linalg.norm(som[row][col+1]-curNode)
    total_adjacent_nodes += 1
  except IndexError:
    pass
  try:
    euclidian_sum += np.linalg.norm(som[row][col-1]-curNode)
    total_adjacent_nodes += 1
  except IndexError:
    pass
  try:
    euclidian_sum += np.linalg.norm(som[row+1][col-1]-curNode)
    total_adjacent_nodes += 1
  except IndexError:
    pass
  try:
    euclidian_sum += np.linalg.norm(som[row-1][col+1]-curNode)
    total_adjacent_nodes += 1
  except IndexError:
    pass
  try:
    euclidian_sum += np.linalg.norm(som[row+1][col+1]-curNode)
    total_adjacent_nodes += 1
  except IndexError:
    pass
  try:
    euclidian_sum += np.linalg.norm(som[row-1][col-1]-curNode)
    total_adjacent_nodes += 1
  except IndexError:
    pass
  return euclidian_sum / total_adjacent_nodes

def get_euclidian_som(som):
  rows, cols, features = som.shape
  euclidian_som = np.zeros((rows, cols))
  for i in range(len(euclidian_som)):
    for j in range(len(euclidian_som[i])):
      euclidian_sum = get_euclidian_sum(som[i][j], som, i, j)
      euclidian_som[i][j] = euclidian_sum
  euclidian_som *= (1.0/euclidian_som.max())
  return np.stack((np.sqrt(euclidian_som),)*3, axis=-1)

start = time.time()

X_train, Y_train = get_input()

ROWS = 5
SOMsize = 150
vidName = 'animated_som_20.mp4'
saveAnimation = False

# input = get_som_input(ROWS)
# print(input)

som, winningNodes, fig, imgs = gen_som(X_train, Y_train, SOMsize, saveAnimation)
euclidian_som = get_euclidian_som(som)
# print(len(winningNodes))

end = time.time()
print("runtime: %f" % ((end - start)/60))

if saveAnimation:
  somAnimation = animation.ArtistAnimation(fig, imgs)
  #Writer = animation.writers['ffmpeg']
  writer = animation.FFMpegWriter(fps=15, bitrate=50)
  #writer = Writer(fps=15, bitrate=300)
  print("saving animation...")
  somAnimation.save(vidName, writer=writer)
  print("saved")

# plot_som(som, winningNodes)
plot_som(euclidian_som, winningNodes)
# print(euclidian_som)

# print(som)

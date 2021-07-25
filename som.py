import random
import numpy as np
import matplotlib.pyplot as plt
import time

def gen_som(X, SOMsize):
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
  alpha = 0.995
  iter = 0
  a = np.zeros((SOMsize, SOMsize))
  Wold = np.zeros((n, m))
  p = np.zeros((SOMsize, SOMsize, 2))
  for i in range(SOMsize):
    for j in range(SOMsize):
      p[i,j] = [i, j]
  d = np.zeros((SOMsize, SOMsize, 2))
  u = np.zeros((SOMsize, SOMsize, 2))
  m = np.zeros((SOMsize,SOMsize,3))
  while error>tolerance and eta>0.1 and sigma>1:
    iter = iter+1
    Wold = np.copy(Wnew)
    i = random.randint(0, N-1)
    xtrain = X[i,:] / np.linalg.norm(X[i,:])
    a = (np.matmul(xtrain, Wold)).reshape((SOMsize, SOMsize))
    [row, col] = np.unravel_index(np.argmax(a), a.shape)
    d = np.linalg.norm((p - np.asarray([row, col])), axis=2)
    u = np.exp((-1) * (d**2) / (2*(sigma**2)))
    Wnew = Wold + (eta*(u.flatten())) * (xtrain.reshape(3,1) - Wold)
    Wnew = Wnew / np.linalg.norm(Wnew, axis=0)
    eta = eta*alpha
    sigma = sigma*alpha 
    error = np.linalg.norm(Wnew[:,(SOMsize*(row)+col)] - Wold[:,(SOMsize*(row)+col)])
  print("error: %f" % error)
  print("eta: %f" % eta)
  print("sigma: %f" % sigma)
  for i in range(N):
    xtrain = X[i,:] / np.linalg.norm(X[i,:])
    a = (np.matmul(xtrain, Wnew)).reshape((SOMsize, SOMsize))
    [row, col] = np.unravel_index(np.argmax(a), a.shape)
    winningNodes.add((row, col))
  for i in range(n):
    m[:,:,i] = (Wnew[i,:]).reshape((SOMsize, SOMsize))
  print('that took ' + str(iter) + ' iterations')
  return m, winningNodes

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
  winningPlot = np.zeros((som.shape))
  for (row, col) in winningNodes:
    winningPlot[row,col] = som[row,col]
  plot1 = plt.figure(1)
  plt.imshow(winningPlot)

  plot2 = plt.figure(2)
  plt.imshow(som)
  plt.show()

start = time.time()

ROWS = 40
SOMsize = 200

input = get_som_input(ROWS)
# print(input)

som, winningNodes = gen_som(input, SOMsize)
# print(len(winningNodes))

end = time.time()
print("runtime: %f" % (end - start))

plot_som(som, winningNodes)
# print(som)

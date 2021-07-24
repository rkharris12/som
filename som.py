import random
import numpy as np
import matplotlib.pyplot as plt

def gen_som(X, SOMsize):
  (N, d) = X.shape
  n = d
  m = SOMsize ** 2
  Wnew = np.random.rand(n, m) * 255
  for i in range(m):
    Wnew[:,i] = Wnew[:,i] / np.linalg.norm(Wnew[:,i])
  winningNodes = []
  error = np.Inf
  tolerance = 10e-8
  eta = 0.9
  sigma = SOMsize / 5
  alpha = 0.99
  iter = 0

  while error>tolerance and eta>0.1 and sigma>1:
    iter = iter+1
    Wold = np.copy(Wnew)
    i = random.randint(0, N-1)
    xtrain = np.transpose(X[i,:])
    xtrain = xtrain / np.linalg.norm(xtrain)
    a = np.zeros((SOMsize, SOMsize))
    for i in range(SOMsize):
      for j in range(SOMsize):
        a[i,j] = np.matmul(np.transpose(Wold[:,(SOMsize*i+j)]), xtrain)
    [row, col] = np.unravel_index(np.argmax(a), a.shape)
    winningNodes.append((row, col))
    for i in range(SOMsize):
      for j in range(SOMsize):
        d = np.linalg.norm(np.asarray([i, j]) - np.asarray([row, col]))
        u = np.exp((-1) * (d**2) / (2*(sigma**2)))
        Wnew[:,(SOMsize*(i)+j)] = Wold[:,(SOMsize*(i)+j)]+u*eta*(xtrain-Wold[:,(SOMsize*(i)+j)])
        Wnew[:,SOMsize*(i)+j] = Wnew[:,SOMsize*(i)+j] / np.linalg.norm(Wnew[:,SOMsize*(i)+j])
    eta = eta*alpha
    sigma = sigma*alpha 
    error = np.linalg.norm(Wnew[:,(SOMsize*(row)+col)] - Wold[:,(SOMsize*(row)+col)])
  m = np.zeros((SOMsize,SOMsize,3))
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

ROWS = 8
SOMsize = 1000

input = get_som_input(ROWS)
# print(input)

som, winningNodes = gen_som(input, SOMsize)

plot_som(som, winningNodes)

# print(som)



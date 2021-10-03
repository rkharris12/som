import random
import numpy as np
import matplotlib.pyplot as plt
import time

def gen_som(X, SOMsize, sigma_factor):
  (N, d) = X.shape
  n = d
  m = SOMsize ** 2
  Wnew = np.random.rand(n, m) * 255
  for i in range(m):
    Wnew[:,i] = Wnew[:,i] / np.linalg.norm(Wnew[:,i])
  #winningNodes = set()
  error = np.Inf
  tolerance = 10e-8
  eta = 0.9
  sigma = SOMsize / sigma_factor
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
  start_alg_time = time.time()
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
  end_alg_time = time.time()
  print("alg time: %f" % (end_alg_time - start_alg_time))
  #for i in range(N):
  #  xtrain = X[i,:] / np.linalg.norm(X[i,:])
  #  a = (np.matmul(xtrain, Wnew)).reshape((SOMsize, SOMsize))
  #  [row, col] = np.unravel_index(np.argmax(a), a.shape)
  #  winningNodes.add((row, col))
  #test = np.zeros((X.shape[0], SOMsize**2))
  #test = np.matmul(X, Wnew)
  #winningNodes = np.transpose(np.asarray(np.unravel_index(np.argmax((np.matmul(X, Wnew)), axis=1), (SOMsize, SOMsize))))
  #winningNodes = np.transpose(np.asarray(np.unravel_index(np.argmax(test, axis=1), (SOMsize, SOMsize))))
  test = np.zeros((1000, SOMsize**2))
  num_iter = X.shape[0] // 1000
  winningNodes = np.zeros((X.shape[0], 2), dtype="int64")
  if X.shape[0] % 1000 != 0:
    num_iter += 1
  for i in range(num_iter):
    #if i % 10 == 0:
    #  print(str(100*(1000*i/X.shape[0])) + " % done")
    print(str(100*(1000*i/X.shape[0])) + "% done")
    # account for the last iteration where there might not be 1000 more in X
    if i*1000 == X.shape[0] // 1000:
      x_range = X.shape[0] - i*1000
    else:
      x_range = 1000
    test = np.matmul(X[(i*x_range):((i+1)*x_range),:], Wnew)
    winningNodes[(i*x_range):((i+1)*x_range),:] = np.transpose(np.asarray(np.unravel_index(np.argmax(test, axis=1), (SOMsize, SOMsize))))
  end_winning_nodes_time = time.time()
  print("winning nodes time: %f" % (end_winning_nodes_time - end_alg_time))
  for i in range(n):
    m[:,:,i] = (Wnew[i,:]).reshape((SOMsize, SOMsize))
  print("iterations: " + str(iter))
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

def plot_som(som, winningNodes, numrows, somsize, sigmafactor):
  winningPlot = np.zeros((som.shape))
  for pos in winningNodes:
    winningPlot[pos[0],pos[1]] = som[pos[0],pos[1]]
  plot1 = plt.figure(1)
  plt.imshow(winningPlot)
  plt.savefig("winners_" + str(numrows) + "_" + str(somsize) + "_" + str(sigmafactor) + ".png")

  plot2 = plt.figure(2)
  plt.imshow(som)
  plt.savefig("colors_" + str(numrows) + "_" + str(somsize) + "_" + str(sigmafactor) + ".png")
  #plt.show()


start_time = time.time()

ROWS = [50]
SOMSIZES= [200]
SIGMA_FACTORS = [5]

for row in ROWS:
  for somsize in SOMSIZES:
    for sigma_factor in SIGMA_FACTORS:
      print("row: %d, somsize: %d, sigma_factor: %d" % (row, somsize, sigma_factor))
      start_som_time = time.time()

      input = get_som_input(row)
      gen_input_time = time.time()
      print("gen input runtime: %f" % (gen_input_time - start_som_time))

      som, winningNodes = gen_som(input, somsize, sigma_factor)
      gen_som_time = time.time()
      print("gen som runtime: %f" % (gen_som_time - gen_input_time))

      plot_som(som, winningNodes, row, somsize, sigma_factor)
      plot_som_time = time.time()
      print("plot som runtime: %f" % (plot_som_time - gen_som_time))

      print("total som runtime: %f" % (plot_som_time - start_som_time))      

end_time = time.time()
print("runtime: %f" % (end_time - start_time))

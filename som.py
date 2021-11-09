import random
import numpy as np
import matplotlib.pyplot as plt
import time


# generate the SOM and find winning nodes
def gen_som(X, som_size, sigma_factor):
  (num_training_samples, n) = X.shape
  m = som_size ** 2
  # initialize normalized weight vectors
  Wnew = np.random.rand(n, m) * 255
  for i in range(m):
    Wnew[:,i] = Wnew[:,i] / np.linalg.norm(Wnew[:,i])
  # initialize parameters and matrices
  error = np.Inf
  tolerance = 10e-8
  eta = 0.9
  sigma = som_size / sigma_factor
  alpha = 0.995
  iter = 0
  a = np.zeros((som_size, som_size))
  Wold = np.zeros((n, m))
  p = np.zeros((som_size, som_size, 2))
  for i in range(som_size):
    for j in range(som_size):
      p[i,j] = [i, j]
  d = np.zeros((som_size, som_size, 2))
  u = np.zeros((som_size, som_size, 2))
  som = np.zeros((som_size,som_size,3))
  start_alg_time = time.time()
  # iterate the SOM algorithm until convergence
  while error>tolerance and eta>0.1 and sigma>1:
    Wold = np.copy(Wnew)
    i = random.randint(0, num_training_samples-1)
    xtrain = X[i,:] / np.linalg.norm(X[i,:])
    a = (np.matmul(xtrain, Wold)).reshape((som_size, som_size))
    [row, col] = np.unravel_index(np.argmax(a), a.shape)
    d = np.linalg.norm((p - np.asarray([row, col])), axis=2)
    u = np.exp((-1) * (d**2) / (2*(sigma**2)))
    Wnew = Wold + (eta*(u.flatten())) * (xtrain.reshape(3,1) - Wold)
    Wnew = Wnew / np.linalg.norm(Wnew, axis=0)
    eta = eta*alpha
    sigma = sigma*alpha 
    error = np.linalg.norm(Wnew[:,(som_size*(row)+col)] - Wold[:,(som_size*(row)+col)])
    iter = iter+1
  end_alg_time = time.time()
  print("alg time: %f" % (end_alg_time - start_alg_time))
  # find the winning nodes
  # go through 1000 training samples at at time.  Memory problems if we try to do it all at once for a big SOM with lots of training samples
  temp_activation = np.zeros((1000, som_size**2))
  num_iter = X.shape[0] // 1000
  winning_nodes = np.zeros((X.shape[0], 2), dtype="int64")
  if X.shape[0] % 1000 != 0:
    num_iter += 1
  for i in range(num_iter):
    print(str(100*(1000*i/X.shape[0])) + "% done")
    # account for the last iteration where there might not be 1000 more in X
    if i*1000 == X.shape[0] // 1000:
      x_range = X.shape[0] - i*1000
    else:
      x_range = 1000
    temp_activation = np.matmul(X[(i*x_range):((i+1)*x_range),:], Wnew)
    winning_nodes[(i*x_range):((i+1)*x_range),:] = np.transpose(np.asarray(np.unravel_index(np.argmax(temp_activation, axis=1), (som_size, som_size))))
  end_winning_nodes_time = time.time()
  print("winning nodes time: %f" % (end_winning_nodes_time - end_alg_time))
  for i in range(n):
    som[:,:,i] = (Wnew[i,:]).reshape((som_size, som_size))
  print("iterations: " + str(iter))
  return som, winning_nodes

# generate color spectrum training data
def gen_training_data(N):
  rows = N**3
  X = np.zeros((rows, 3))
  intensities = np.linspace(0,255,N)
  # generate all color combos
  for r in range(N):
    for g in range(N):
      for b in range(N):
        X[(N**2)*r+N*g+b,:] = [intensities[r], intensities[g], intensities[b]]
  filtered = []
  # filter out greys, blacks, and whites
  for i in range(N**3):
    if (X[i,0] != X[i,1] or X[i,0] != X[i,2] or X[i,1] != X[i,2]):
      filtered.append(X[i])
  filtered = np.asarray(filtered)
  return filtered

# save plots of the som and the som with just the winning nodes
def plot_som(som, winning_nodes, row, som_size, sigma_factor):
  winning_plot = np.zeros((som.shape))
  for pos in winning_nodes:
    winning_plot[pos[0],pos[1]] = som[pos[0],pos[1]]
  plot1 = plt.figure(1)
  plt.imshow(winning_plot)
  #plt.savefig("winners_" + str(row) + "_" + str(som_size) + "_" + str(sigma_factor) + ".png")

  plot2 = plt.figure(2)
  plt.imshow(som)
  #plt.savefig("som_" + str(row) + "_" + str(som_size) + "_" + str(sigma_factor) + ".png")
  plt.show()


# main code
start_time = time.time()

ROWS = [25]
SOM_SIZES= [200]
SIGMA_FACTORS = [3]

for row in ROWS:
  for som_size in SOM_SIZES:
    for sigma_factor in SIGMA_FACTORS:
      print("row: %d, som_size: %d, sigma_factor: %d" % (row, som_size, sigma_factor))
      start_som_time = time.time()

      training_data = gen_training_data(row)
      gen_input_time = time.time()
      print("gen training data runtime: %f" % (gen_input_time - start_som_time))

      som, winning_nodes = gen_som(training_data, som_size, sigma_factor)
      gen_som_time = time.time()
      print("gen som runtime: %f" % (gen_som_time - gen_input_time))

      plot_som(som, winning_nodes, row, som_size, sigma_factor)
      plot_som_time = time.time()
      print("plot som runtime: %f" % (plot_som_time - gen_som_time))

      print("total som runtime: %f" % (plot_som_time - start_som_time))      

end_time = time.time()
print("runtime: %f" % (end_time - start_time))

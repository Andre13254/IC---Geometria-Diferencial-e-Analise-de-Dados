import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt



#Plot da Esfera
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
 
 #Malha do domínio
u = np.linspace(0,np.pi,100)
v = np.linspace(0,2*np.pi,100)

 #passando pela parametrização
x = np.outer(np.sin(u),np.cos(v))
y = np.outer(np.sin(u),np.sin(v))
z = np.outer(np.cos(u),np.ones(np.size(v)))

ax.plot_surface(x,y,z, alpha=0.3)


#amostra do domínio por distribuiço uniforme
rng = np.random.default_rng(12345)
n_pontos=500

pontos_u = rng.uniform(0.1,np.pi-0.1,n_pontos)
pontos_v = rng.uniform(-np.pi/2,np.pi/2,n_pontos)

#passando pontos pela parametrização e colocando-os numa matriz
pontos_R3=np.zeros((n_pontos,3))

for i in range(len(pontos_u)):
  x = np.sin(pontos_u[i])*np.cos(pontos_v[i])
  y = np.sin(pontos_u[i])*np.sin(pontos_v[i])
  z = np.cos(pontos_u[i])

  pontos_R3[i,:]=np.array([x,y,z])
  
  #if i%5==0:
  ax.scatter(x,y,z, color='pink')


#Índices dos pontos inicial e final no grafo
indice_inicial=6
indice_final=10


#aplicando knn nos pontos em R3
k_vizinhos=8

nbrs = NearestNeighbors(n_neighbors=k_vizinhos,algorithm='auto',metric='euclidean').fit(pontos_R3)
dist_vizinhos,vizinhos = nbrs.kneighbors(pontos_R3)

#fazendo as matrizes de adj e distancias
matriz_adjacencias = nbrs.kneighbors_graph(pontos_R3).toarray()

matriz_distancias = matriz_adjacencias.copy()
'''iterando sobre o grafo de adjacencias para ver se 2 pontos sao adjacentes'''
for linha in range(n_pontos):
  for coluna in range(n_pontos):

    if matriz_distancias[linha,coluna]==1:
      '''iterando sobre a matriz de vizinhos_nn para achar em que indice na matriz de distancias_nn está a distancia entre os pontos do grafo'''
      for linha_vizinho in range(n_pontos):
        for coluna_vizinho in range(k_vizinhos):
          if vizinhos[linha_vizinho,0]==linha and vizinhos[linha_vizinho,coluna_vizinho]==coluna:
            
            matriz_distancias[linha,coluna]*=dist_vizinhos[linha_vizinho,coluna_vizinho]
    else:
      matriz_distancias[linha,coluna]=0


#transformando a matriz de distancias num grafo
G = nx.from_numpy_array(matriz_distancias)

#achando o caminho mínimo entre dois nós por dijkstra
min_path = nx.dijkstra_path(G,indice_inicial,indice_final)

#plotando o caminho mínimo
for i in range(len(min_path)-1):
  p1 = pontos_R3[min_path[i],:]
  p2 = pontos_R3[min_path[i+1],:]

  reta_x=np.array([p1[0],p2[0]])
  reta_y=np.array([p1[1],p2[1]])
  reta_z=np.array([p1[2],p2[2]])

  ax.plot(reta_x,reta_y,reta_z, color='orange') 

#colocando vertices do caminho minimo numa matriz
vertices_min_path=np.array([[0,0,0]])
for i in min_path:
  vertices_min_path = np.append(vertices_min_path,[pontos_R3[i]],axis=0)
vertices_min_path = np.delete(vertices_min_path,0,0)

#Plotando com outra cor os pontos do caminho minimo
for k in min_path:
  x=pontos_R3[k,0]
  y=pontos_R3[k,1]
  z=pontos_R3[k,2]
  ax.scatter(x,y,z,color='yellow')

#Comprimento do caminho mínimo
comp_cm=0
for ponto in range(len(vertices_min_path)-1):
  comp_cm+=np.linalg.norm(vertices_min_path[ponto+1]-vertices_min_path[ponto])




print('Pontos da amostra:',n_pontos)
print('Viznhos:',k_vizinhos)
print('Número de componentes conexas:',nx.number_connected_components(G))
print('comprimento do cm:',comp_cm)

#Vizualização(não funciona no linux)
#ax.set_aspect('equal')

plt.show()
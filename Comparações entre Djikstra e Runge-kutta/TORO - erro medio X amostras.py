import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot()
ax2 = plt.figure(2).add_subplot(projection='3d')


a=3
r=1
def t(u):
  return (a+r*np.cos(u))

#Plot do toro
u = np.linspace(0,2*np.pi,100)
v = np.linspace(0,2*np.pi,100)

x = np.outer(t(u),np.cos(v))
y = np.outer(t(u),np.sin(v))
z = np.outer(np.sin(u),r*np.ones(np.size(v)))

ax2.plot_surface(x,y,z, alpha=0.3)



#pontos do domínio por distribuiço uniforme para pegar os pontos iniciais e finais das geodesicas
rng = np.random.default_rng(12345)
n_pontos=150

pontos_u = rng.uniform(-np.pi/6,np.pi/6,n_pontos)
pontos_v = rng.uniform(-np.pi/6,np.pi/6,n_pontos)



pontos_u_iniciais = np.array([0])
pontos_u_finais = np.array([0])
pontos_v_iniciais = np.array([0])
pontos_v_finais = np.array([0])
for i in [10,56,34,17,98,2,78,90,77,9]:
  pontos_u_iniciais = np.concatenate((pontos_u_iniciais,np.array([pontos_u[i]])))
pontos_u_iniciais = np.delete(pontos_u_iniciais,0,0)

for i in [66,84,23,75,26,97,3,75,15,7]:
  pontos_u_finais = np.concatenate((pontos_u_finais,np.array([pontos_u[i]])))
pontos_u_finais = np.delete(pontos_u_finais,0,0)

for i in [50,36,9,149,1,86,3,5,90,54]:
  pontos_v_iniciais = np.concatenate((pontos_v_iniciais,np.array([pontos_v[i]])))
pontos_v_iniciais = np.delete(pontos_v_iniciais,0,0)

for i in [12,3,14,6,99,43,138,133,89,8]:
  pontos_v_finais = np.concatenate((pontos_v_finais,np.array([pontos_v[i]])))
pontos_v_finais = np.delete(pontos_v_finais,0,0)

#Pontos iniciais e finais passados pela parametrização

pontos_R3_iniciais=np.zeros((len(pontos_u_iniciais),3))
for i in range(len(pontos_u_iniciais)):
    x = t(pontos_u_iniciais[i])*np.cos(pontos_v_iniciais[i])
    y = t(pontos_u_iniciais[i])*np.sin(pontos_v_iniciais[i])
    z = np.sin(pontos_u_iniciais[i])
    pontos_R3_iniciais[i,:]=np.array([x,y,z])
pontos_R3_finais=np.zeros((len(pontos_u_iniciais),3))
for i in range(len(pontos_u_finais)):
    x = t(pontos_u_finais[i])*np.cos(pontos_v_finais[i])
    y = t(pontos_u_finais[i])*np.sin(pontos_v_finais[i])
    z = np.sin(pontos_u_finais[i])
    pontos_R3_finais[i,:]=np.array([x,y,z])




def runge_kutta_toro(indice_inicial,indice_final,pontos_u_iniciais,pontos_u_finais,pontos_v_iniciais,pontos_v_finais):
  def t(u):
    return (a+r*np.cos(u))

  pontos_R3_iniciais=np.zeros((n_pontos,3))
  for i in range(len(pontos_u_iniciais)):
      x = t(pontos_u_iniciais[i])*np.cos(pontos_v_iniciais[i])
      y = t(pontos_u_iniciais[i])*np.sin(pontos_v_iniciais[i])
      z = np.sin(pontos_u_iniciais[i])

      pontos_R3_iniciais[i,:]=np.array([x,y,z])

  pontos_R3_finais=np.zeros((n_pontos,3))
  for i in range(len(pontos_u_finais)):
      x = t(pontos_u_finais[i])*np.cos(pontos_v_finais[i])
      y = t(pontos_u_finais[i])*np.sin(pontos_v_finais[i])
      z = np.sin(pontos_u_finais[i])

      pontos_R3_finais[i,:]=np.array([x,y,z])
  
 

  def df(u,v,f,g):
   return -(g**2)*(t(u)*np.sin(u))*(1/r)

  def dg(u,v,f,g):
    return 2*f*g*(r*np.sin(u)/t(u))

    #Ponto inicial e direções no domínio
  a1 = pontos_u_iniciais[indice_inicial]     
  b1 = pontos_v_iniciais[indice_inicial]     

  a2 = pontos_u_finais[indice_final]-pontos_u_iniciais[indice_inicial] 
  b2 = pontos_v_finais[indice_final]-pontos_v_iniciais[indice_inicial] 

  direcao = np.array([a2,b2]) / np.linalg.norm(np.array([a2,b2]))
  angulo_inicial = np.arctan2(direcao[1],direcao[0])
  

  ponto_inicial= pontos_R3_iniciais[indice_inicial,:]
  ponto_final = pontos_R3_finais[indice_final,:]
  dir_R3 = (ponto_final - ponto_inicial) / np.linalg.norm(ponto_final - ponto_inicial)

  #Primeira iteração do runge kutta para determinar o primeiro ajuste para o angulo
  X=np.array([a1,b1,np.cos(angulo_inicial),np.sin(angulo_inicial)])
  cos_parada=2
  h=0.001
  pf_ate_geod=1

  listax=np.array([])
  listay=np.array([])
  listaz=np.array([])

  while (cos_parada==2 or abs(cos_parada)>0.1):
    #print(pf_ate_geod)
    xi=np.array([t(X[0])*np.cos(X[1])])
    yi=np.array([t(X[0])*np.sin(X[1])])
    zi=np.array([r*np.sin(X[0])])                      

    listax=np.append(listax,xi)                  
    listay=np.append(listay,yi)
    listaz=np.append(listaz,zi)

    ponto_geod=np.array([xi[0],yi[0],zi[0]])

    v2 = ponto_geod-ponto_final

    pf_ate_geod=np.linalg.norm(v2)

    cos_parada = np.inner(dir_R3,v2) / pf_ate_geod


    k1=h*np.array([X[2],X[3],df(X[0],X[1],X[2],X[3]),dg(X[0],X[1],X[2],X[3])])
    Y=X+0.5*k1
    k2=h*np.array([Y[2],Y[3],df(Y[0],Y[1],Y[2],Y[3]),dg(Y[0],Y[1],Y[2],Y[3])])
    Y=X+0.5*k2
    k3=h*np.array([Y[2],Y[3],df(Y[0],Y[1],Y[2],Y[3]),dg(Y[0],Y[1],Y[2],Y[3])])
    Y=X+k3
    k4=h*np.array([Y[2],Y[3],df(Y[0],Y[1],Y[2],Y[3]),dg(Y[0],Y[1],Y[2],Y[3])])
    X=X+(1/6)*(k1+2*k2+2*k3+k4)

  v1 = ponto_geod - ponto_inicial
  norma_v1 = np.linalg.norm(v1)
  vetor_referencia = np.cross(dir_R3,v1)

  #primeiro ajuste
  cos_ajuste = np.inner(dir_R3,v1) / norma_v1
  ajuste = np.arccos(cos_ajuste)

  #Segunda e Terceira iterações para determinar quando o angulo deve ser somado ou subtraido
  for iter in range(2):
    if iter==0:
      X=np.array([a1,b1,np.cos(angulo_inicial+ajuste),np.sin(angulo_inicial+ajuste)])

    else:
      X=np.array([a1,b1,np.cos(angulo_inicial-ajuste),np.sin(angulo_inicial-ajuste)])

    cos_parada=2
    h=0.001
    pf_ate_geod=1

    while (cos_parada==2 or abs(cos_parada)>0.1):
      xi=np.array([t(X[0])*np.cos(X[1])])
      yi=np.array([t(X[0])*np.sin(X[1])])
      zi=np.array([r*np.sin(X[0])])                      

      ponto_geod=np.array([xi[0],yi[0],zi[0]])

      v2 = ponto_final - ponto_geod
      pf_ate_geod=np.linalg.norm(v2)

      cos_parada = np.inner(dir_R3,v2) / pf_ate_geod

      k1=h*np.array([X[2],X[3],df(X[0],X[1],X[2],X[3]),dg(X[0],X[1],X[2],X[3])])
      Y=X+0.5*k1
      k2=h*np.array([Y[2],Y[3],df(Y[0],Y[1],Y[2],Y[3]),dg(Y[0],Y[1],Y[2],Y[3])])
      Y=X+0.5*k2
      k3=h*np.array([Y[2],Y[3],df(Y[0],Y[1],Y[2],Y[3]),dg(Y[0],Y[1],Y[2],Y[3])])
      Y=X+k3
      k4=h*np.array([Y[2],Y[3],df(Y[0],Y[1],Y[2],Y[3]),dg(Y[0],Y[1],Y[2],Y[3])])
      X=X+(1/6)*(k1+2*k2+2*k3+k4)

    if iter==0:
      pf_ate_geod_1=pf_ate_geod
    else:
      pf_ate_geod_2=pf_ate_geod

  if pf_ate_geod_1<pf_ate_geod_2:
    orientaçao=1
    angulo_inicial+=ajuste
  else:
    orientaçao=-1
    angulo_inicial-=ajuste


  #Aplicando Runge kutta com ajuste da direção inicial
  erro_aceito=0.05
  pf_ate_geod=1
  while pf_ate_geod>erro_aceito:
    #print(pf_ate_geod)
    #Salvando pontos da geodésica para plot e calculo de erro
    X=np.array([a1,b1,np.cos(angulo_inicial),np.sin(angulo_inicial)])
    cos_parada=2
    passos=0
    h=0.001
    listax=np.array([])
    listay=np.array([])
    listaz=np.array([])

    pf_ate_geod=1
    matriz_geod = np.array([[0,0,0]])

    while (cos_parada==2 or abs(cos_parada)>0.15) and pf_ate_geod>erro_aceito:
      xi=np.array([t(X[0])*np.cos(X[1])])
      yi=np.array([t(X[0])*np.sin(X[1])])
      zi=np.array([r*np.sin(X[0])])                      

      listax=np.append(listax,xi)                  
      listay=np.append(listay,yi)
      listaz=np.append(listaz,zi)

      ponto_geod=np.array([xi[0],yi[0],zi[0]])
      matriz_geod = np.append(matriz_geod,[ponto_geod],axis=0)

      v2 = ponto_final - ponto_geod
      pf_ate_geod=np.linalg.norm(v2)

      cos_parada = np.inner(dir_R3,v2) / pf_ate_geod

      k1=h*np.array([X[2],X[3],df(X[0],X[1],X[2],X[3]),dg(X[0],X[1],X[2],X[3])])
      Y=X+0.5*k1

      k2=h*np.array([Y[2],Y[3],df(Y[0],Y[1],Y[2],Y[3]),dg(Y[0],Y[1],Y[2],Y[3])])
      Y=X+0.5*k2

      k3=h*np.array([Y[2],Y[3],df(Y[0],Y[1],Y[2],Y[3]),dg(Y[0],Y[1],Y[2],Y[3])])
      Y=X+k3

      k4=h*np.array([Y[2],Y[3],df(Y[0],Y[1],Y[2],Y[3]),dg(Y[0],Y[1],Y[2],Y[3])])
      X=X+(1/6)*(k1+2*k2+2*k3+k4)

      passos+=1
    
    if pf_ate_geod>erro_aceito:
      v1 = ponto_geod - ponto_inicial
      norma_v1 = np.linalg.norm(v1)

      vetor_normal=np.cross(dir_R3,v1)
      ref = np.inner(vetor_referencia,vetor_normal)

      #próximos ajustes por bissecção
      cos_ajuste = np.inner(dir_R3,v1) / norma_v1
      ajuste = 0.5*np.arccos(cos_ajuste)

      angulo_inicial+=np.sign(ref)*orientaçao*ajuste

  matriz_geod = np.delete(matriz_geod,0,0)

  return matriz_geod

  
 #Fazendo um terceira geodésica com a média das duas primeiras

#fazendo uma lista com os comprimentos das geodésicas de runge-kutta
comprimentos_geod=np.array([])
for i in range(len(pontos_u_iniciais)):
  matriz_geod_ida=runge_kutta_toro(i,i,pontos_u_iniciais,pontos_u_finais,pontos_v_iniciais,pontos_v_finais)
  matriz_geod_volta=runge_kutta_toro(i,i,pontos_u_finais,pontos_u_iniciais,pontos_v_finais,pontos_v_iniciais)

  pontos_ida = np.size(matriz_geod_ida,axis=0)
  pontos_volta = np.size(matriz_geod_volta,axis=0)
  dif_de_pontos = pontos_ida - pontos_volta

  #Deixando as duas matrizes com a mesma quantidade de pontos
  if dif_de_pontos>0:
    for x in range(len(matriz_geod_ida[:,0])):
      if x%(pontos_ida//dif_de_pontos)==0 and pontos_ida>pontos_volta:
        matriz_geod_ida = np.delete(matriz_geod_ida,x,0)
      pontos_ida=np.size(matriz_geod_ida,axis=0)
      
  if dif_de_pontos<0:
    dif_de_pontos = abs(dif_de_pontos)
    for x in range(len(matriz_geod_volta[:,0])):
      if x%(pontos_volta//dif_de_pontos)==0 and pontos_volta>pontos_ida:
        matriz_geod_volta = np.delete(matriz_geod_volta,x,0)  
      pontos_volta=np.size(matriz_geod_volta,axis=0)

  #Invertendo a ordem dos pontos da matriz da geodesica de volta
  def inverter_matriz(m):
      m_invert = np.array([[0,0,0]])
      l = np.size(m,0)
      for x in range(l-1,-1,-1):
        m_invert=np.append(m_invert,[m[x,:]],axis=0)
      m_invert=np.delete(m_invert,0,0)
      return m_invert
    
  matriz_geod_volta_inv = inverter_matriz(matriz_geod_volta)

  #Fazendo a média
  matriz_geod_media = np.array([[0,0,0]])
  par = np.linspace(0,1,pontos_ida)
  for x in range(pontos_ida-1):
    ponto_medio = matriz_geod_ida[x,:]*(1-par[x]) + matriz_geod_volta_inv[x,:]*par[x]
    matriz_geod_media = np.append(matriz_geod_media,[ponto_medio],axis=0)
  matriz_geod_media=np.delete(matriz_geod_media,0,0)

  #Plotando a geodésica média
  x=matriz_geod_media[:,0]
  y=matriz_geod_media[:,1]
  z=matriz_geod_media[:,2]

  ax2.plot(x,y,z, color='purple')


 
  comp_geod_media=0
  for linha in range(len(matriz_geod_ida[:,1])-2):
    comp_geod_media+=np.linalg.norm(matriz_geod_media[linha+1,:]-matriz_geod_media[linha,:])
  
  '''ax.plot(matriz_geod_media[:,0],matriz_geod_media[:,1],matriz_geod_media[:,2],color='purple')'''
  comprimentos_geod=np.append(comprimentos_geod,comp_geod_media)
  print('geodesicas calculadas:',i+1)





amostras=[50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300]
erros_medios=[]




for num in range(len(amostras)):
#Calculando caminhos minimos entre pontos inicial e final de cada geodesica para cada numero de pontos na amostra
  #pontos do domínio por distribuiço uniforme
  rng = np.random.default_rng(12345)
  n_pontos=amostras[num]                                      #distribuição igual à que ta sendo usada no runge kutta

  pontos_u = rng.uniform(-np.pi/6,np.pi/6,n_pontos)
  pontos_v = rng.uniform(-np.pi/6,np.pi/6,n_pontos)

  #passando pontos pela parametrização e colocando-os numa matriz
  pontos_R3=np.zeros((n_pontos,3))
  for i in range(len(pontos_u)):
      x = t(pontos_u[i])*np.cos(pontos_v[i])
      y = t(pontos_u[i])*np.sin(pontos_v[i])
      z = np.sin(pontos_u[i])

      pontos_R3[i,:]=np.array([x,y,z])
      '''ax.scatter(x,y,z,color='pink')'''

  #adicionando os pontos salvos para serem os iniciais e finais
  pontos_R3 = np.concatenate((pontos_R3_iniciais,pontos_R3_finais,pontos_R3),axis=0)

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

  #fazendo uma lista com os comprimentos dos caminhos minimos
  comprimentos_cm=np.array([])
  for i in range(len(pontos_u_iniciais)):
    #achando o caminho mínimo entre dois nós por dijkstra
    min_path = nx.dijkstra_path(G,i,i+len(pontos_u_iniciais))

    #plotando o caminho mínimo
    for i in range(len(min_path)-1):
      p1 = pontos_R3[min_path[i],:]
      p2 = pontos_R3[min_path[i+1],:]
    
      reta_x=np.array([p1[0],p2[0]])
      reta_y=np.array([p1[1],p2[1]])
      reta_z=np.array([p1[2],p2[2]])
    
      ax2.plot(reta_x,reta_y,reta_z, color='orange') 



    #colocando vertices do caminho minimo numa matriz
    vertices_min_path=np.array([[0,0,0]])
    for p in min_path:
      vertices_min_path = np.append(vertices_min_path,[pontos_R3[p,:]],axis=0)
    vertices_min_path = np.delete(vertices_min_path,0,0)
    '''ax.plot(vertices_min_path[:,0],vertices_min_path[:,1],vertices_min_path[:,2],color='orange')'''

   
      #lista com os comprimentos dos caminhos minimos
    comp_cm=0
    for ponto in range(len(vertices_min_path)-1):
      comp_cm+=np.linalg.norm(vertices_min_path[ponto+1,:]-vertices_min_path[ponto,:])
    comprimentos_cm=np.append(comprimentos_cm,comp_cm)
    
    


  
  #fazendo uma lista com os erros relativos entre os caminhos iminimos e as geodesicas e tirando a media desses erros
  erros_absolutos=comprimentos_cm-comprimentos_geod
  
  erros_relativos=[]
  for x in range(len(erros_absolutos)):
    erros_relativos.append(erros_absolutos[x]/comprimentos_geod[x])

  erro_med=sum(erros_relativos)/len(erros_relativos)



  erros_medios.append(erro_med)
  print('Amostras analisadas:',amostras[num])

print(np.array(erros_medios))
print(np.array(amostras))

ax.plot(amostras, erros_medios)     
ax.set_xlabel('Número de pontos da amostra')
ax.set_ylabel('Erro médio entre Djikstra e Runge-kutta')

plt.show()
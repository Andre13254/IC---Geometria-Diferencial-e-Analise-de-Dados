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


#Índices dos pontos inicial e final no grafo
indice_inicial=6
indice_final=10


#passando pontos pela parametrização e colocando-os numa matriz
pontos_R3=np.zeros((n_pontos,3))

for i in range(len(pontos_u)):
  x = np.sin(pontos_u[i])*np.cos(pontos_v[i])
  y = np.sin(pontos_u[i])*np.sin(pontos_v[i])
  z = np.cos(pontos_u[i])

  pontos_R3[i,:]=np.array([x,y,z])

  if i==indice_inicial or i==indice_final:
    ax.scatter(x,y,z,color='yellow')




#Geodésica Aproximada por Runge-Kutta
def runge_kutta_esfera(indice_inicial,indice_final):

  #amostra do domínio por distribuiço uniforme (essa amostra deve ser igual á que será usada no knn)
  rng = np.random.default_rng(12345)
  n_pontos=500
  
  pontos_u = rng.uniform(0.1,np.pi-0.1,n_pontos)
  pontos_v = rng.uniform(-np.pi/2,np.pi/2,n_pontos)


    #Ponto inicial e direções no domínio
  a1 = pontos_u[indice_inicial]     
  b1 = pontos_v[indice_inicial]     

  a2 = pontos_u[indice_final]-pontos_u[indice_inicial] 
  b2 = pontos_v[indice_final]-pontos_v[indice_inicial] 

  direcao = np.array([a2,b2]) / np.linalg.norm(np.array([a2,b2]))
  angulo_inicial = np.arctan2(direcao[1],direcao[0])
  
  ponto_inicial= pontos_R3[indice_inicial,:]
  ponto_final = pontos_R3[indice_final,:]
  dir_R3 = (ponto_final - ponto_inicial) / np.linalg.norm(ponto_final - ponto_inicial) #vetor unitario que vai do ponto inicial ao final em R3
  
   #derivadas de segunda ordem das variáveis do domínio em relação ao parâmetro
  def df(u,v,f,g):
   return g*np.sin(u)*np.cos(u)

  def dg(u,v,f,g):
    return -2*f*g*(np.cos(u)/np.sin(u))
  

  #Primeira iteração do runge kutta para determinar o primeiro ajuste para o angulo
  X=np.array([a1,b1,np.cos(angulo_inicial),np.sin(angulo_inicial)])
  cos_parada=2 
  h=0.001 #tamanho do passo
  pf_ate_geod=1
  erro_aceito=0.01

  while abs(cos_parada)>0.1 and pf_ate_geod>erro_aceito:  #a geodésica para se o angulo estiver próximo a 90graus ou se passar perto do ponto final que queremos
    
    xi=np.array([np.sin(X[0])*np.cos(X[1])])
    yi=np.array([np.sin(X[0])*np.sin(X[1])])
    zi=np.array([np.cos(X[0])])

    ponto_geod=np.array([xi[0],yi[0],zi[0]])

    v2 = ponto_geod-ponto_final    
    pf_ate_geod=np.linalg.norm(v2) #cálculo do erro

    cos_parada = np.inner(dir_R3,v2) / pf_ate_geod
  

    k1=h*np.array([X[2],X[3],df(X[0],X[1],X[2],X[3]),dg(X[0],X[1],X[2],X[3])])
    Y=X+0.5*k1
  
    k2=h*np.array([Y[2],Y[3],df(Y[0],Y[1],Y[2],Y[3]),dg(Y[0],Y[1],Y[2],Y[3])])
    Y=X+0.5*k2
                                                                                                   #Atualização do ponto e da direção no domínio (os pontos são atualizados com as derivadas de primeira ordem e as direções com as de segunda ordem)
    k3=h*np.array([Y[2],Y[3],df(Y[0],Y[1],Y[2],Y[3]),dg(Y[0],Y[1],Y[2],Y[3])])
    Y=X+k3
  
    k4=h*np.array([Y[2],Y[3],df(Y[0],Y[1],Y[2],Y[3]),dg(Y[0],Y[1],Y[2],Y[3])])
    X=X+(1/6)*(k1+2*k2+2*k3+k4)

 
  v1 = ponto_geod - ponto_inicial        
  norma_v1 = np.linalg.norm(v1)          
  vetor_referencia = np.cross(dir_R3,v1)  #produto vetorial entre vetores que saem do ponto inicial e vao até o ponto alvo e o último ponto da geodésica
                                          #será usado como referência para saber se devemos adicionar ou subtrair o ângulo do ângulo inicial no domínio 
  
  #primeiro ajuste
  cos_ajuste = np.inner(dir_R3,v1) / norma_v1
  ajuste = np.arccos(cos_ajuste)          #ângulo a ser somado ou subtraído no domínio


  #Segunda e Terceira iterações para determinar quando o angulo deve ser somado ou subtraido
  for iter in range(2):
    
    if iter==0:
      X=np.array([a1,b1,np.cos(angulo_inicial+ajuste),np.sin(angulo_inicial+ajuste)])
    else:                                                                                #roda uma vez somando e outra subtraindo e vê qual é melhor
      X=np.array([a1,b1,np.cos(angulo_inicial-ajuste),np.sin(angulo_inicial-ajuste)])
    cos_parada=2
    h=0.001
    pf_ate_geod=1

    while abs(cos_parada)>0.1 and pf_ate_geod>erro_aceito:
      
      xi=np.array([np.sin(X[0])*np.cos(X[1])])
      yi=np.array([np.sin(X[0])*np.sin(X[1])])
      zi=np.array([np.cos(X[0])])
  
      ponto_geod=np.array([xi[0],yi[0],zi[0]])
      v2 = ponto_geod-ponto_final
      pf_ate_geod=np.linalg.norm(v2)
      cos_parada = np.inner(dir_R3,v2) / pf_ate_geod
    
      k1=h*np.array([X[2],X[3],df(X[0],X[1],X[2],X[3]),dg(X[0],X[1],X[2],X[3])])
      Y1=X+0.5*k1
    
      k2=h*np.array([Y1[2],Y1[3],df(Y1[0],Y1[1],Y1[2],Y1[3]),dg(Y1[0],Y1[1],Y1[2],Y1[3])])
      Y2=X+0.5*k2
    
      k3=h*np.array([Y2[2],Y2[3],df(Y2[0],Y2[1],Y2[2],Y2[3]),dg(Y2[0],Y2[1],Y2[2],Y2[3])])
      Y3=X+k3
    
      k4=h*np.array([Y3[2],Y3[3],df(Y3[0],Y3[1],Y3[2],Y3[3]),dg(Y3[0],Y3[1],Y3[2],Y3[3])])
      X=X+(1/6)*(k1+2*k2+2*k3+k4)

    if iter==0:
      pf_ate_geod_1=pf_ate_geod    #grava os erros para cada caso
    else:
      pf_ate_geod_2=pf_ate_geod

  if pf_ate_geod_1<pf_ate_geod_2:  #decide a orientação dependendo de qual erro foi menor e faz a primeira atualização no ângulo
    orientaçao=1
    angulo_inicial+=ajuste
  else:
    orientaçao=-1
    angulo_inicial-=ajuste



  #Aplicando Runge kutta com ajuste da direção inicial
    
  erro_aceito=0.01 #erro é a distância entre o ponto final que queremos e o ponto final da geodésica
  pf_ate_geod=1

  while pf_ate_geod>erro_aceito:  #para completamente se a geodésica passar perto o suficiente (esse loop se repete para cada ajuste de angulo)
    X=np.array([a1,b1,np.cos(angulo_inicial),np.sin(angulo_inicial)])
    cos_parada=2
    passos=0
    h=0.001
    listax=np.array([])
    listay=np.array([])   #listas para salvar a geodésica por coordenadas
    listaz=np.array([])

    pf_ate_geod=1
    matriz_geod = np.array([[0,0,0]])  #matriz para salvar a geodésica completa (uma coordenada por coluna)

    while abs(cos_parada)>0.15 and pf_ate_geod>erro_aceito:  #(esse loop se repete para construir a geodésica)
      xi=np.array([np.sin(X[0])*np.cos(X[1])])
      yi=np.array([np.sin(X[0])*np.sin(X[1])])
      zi=np.array([np.cos(X[0])])
    
      listax=np.append(listax,xi)
      listay=np.append(listay,yi)
      listaz=np.append(listaz,zi)
  
      ponto_geod=np.array([xi[0],yi[0],zi[0]])
      matriz_geod = np.append(matriz_geod,[ponto_geod],axis=0)

      v2 = ponto_geod-ponto_final
      pf_ate_geod=np.linalg.norm(v2)
      cos_parada = np.inner(dir_R3,v2) / pf_ate_geod
    
      k1=h*np.array([X[2],X[3],df(X[0],X[1],X[2],X[3]),dg(X[0],X[1],X[2],X[3])])
      Y1=X+0.5*k1
    
      k2=h*np.array([Y1[2],Y1[3],df(Y1[0],Y1[1],Y1[2],Y1[3]),dg(Y1[0],Y1[1],Y1[2],Y1[3])])
      Y2=X+0.5*k2
    
      k3=h*np.array([Y2[2],Y2[3],df(Y2[0],Y2[1],Y2[2],Y2[3]),dg(Y2[0],Y2[1],Y2[2],Y2[3])])
      Y3=X+k3
    
      k4=h*np.array([Y3[2],Y3[3],df(Y3[0],Y3[1],Y3[2],Y3[3]),dg(Y3[0],Y3[1],Y3[2],Y3[3])])
      X=X+(1/6)*(k1+2*k2+2*k3+k4)

      passos+=1

    
    if pf_ate_geod>erro_aceito:
      v1 = ponto_geod - ponto_inicial 
      norma_v1 = np.linalg.norm(v1)

      vetor_normal=np.cross(dir_R3,v1)
      ref = np.inner(vetor_referencia,vetor_normal)  #se esse valor é positivo, a operação é a mesma feita da primeira vez

      #próximos ajustes de angulo por bissecção
      cos_ajuste = np.inner(dir_R3,v1) / norma_v1
      ajuste = 0.5*np.arccos(cos_ajuste)           

      angulo_inicial+=np.sign(ref)*orientaçao*ajuste

  matriz_geod = np.delete(matriz_geod,0,0)

  return listax,listay,listaz,matriz_geod 

x,y,z,matriz_geod_ida = runge_kutta_esfera(indice_inicial,indice_final)
ax.plot(x,y,z,color='green')

x,y,z,matriz_geod_volta = runge_kutta_esfera(indice_final,indice_inicial)
ax.plot(x,y,z,color='red')



#Fazendo um terceira geodésica com a média das duas primeiras
pontos_ida = np.size(matriz_geod_ida,axis=0)
pontos_volta = np.size(matriz_geod_volta,axis=0)  #número de pontos em cada geodésica
dif_de_pontos = pontos_ida - pontos_volta         #número de pontos que será retirado da maior 

#Deixando as duas matrizes com a mesma quantidade de pontos
if dif_de_pontos>0:
  for x in range(len(matriz_geod_ida[:,0])):
    if x%(pontos_ida//dif_de_pontos)==0 and pontos_ida>pontos_volta:   #tira os pontos espaçados igualmente
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


#Fazendo a média e plotando
matriz_geod_media = np.array([[0,0,0]])
par = np.linspace(0,1,pontos_ida)

for x in range(pontos_ida-1):
  ponto_medio = matriz_geod_ida[x,:]*(1-par[x]) + matriz_geod_volta_inv[x,:]*par[x]  #média ponderada com peso maior para quem ta mais perto do início (peso varia linearmente)
  matriz_geod_media = np.append(matriz_geod_media,[ponto_medio],axis=0)
matriz_geod_media=np.delete(matriz_geod_media,0,0)

x=matriz_geod_media[:,0]
y=matriz_geod_media[:,1]
z=matriz_geod_media[:,2]

ax.plot(x,y,z, color='purple')

#Comprimentos das Geodesicas
comp_geod_ida=0
for ponto in range(len(matriz_geod_ida[:,1])-1):
  comp_geod_ida+=np.linalg.norm(matriz_geod_ida[ponto+1]-matriz_geod_ida[ponto])

comp_geod_volta=0
for ponto in range(len(matriz_geod_volta[:,1])-1):
  comp_geod_volta+=np.linalg.norm(matriz_geod_volta[ponto+1]-matriz_geod_volta[ponto])

comp_geod_media=0
for ponto in range(len(matriz_geod_media[:,1])-1):
  comp_geod_media+=np.linalg.norm(matriz_geod_media[ponto+1]-matriz_geod_media[ponto])

print('comprimento de geodesica de ida:',comp_geod_ida)
print('comprimento de geodesica  de volta:',comp_geod_volta)
print('comprimento de geodesica media:',comp_geod_media)

#Vizualização(não funciona no linux)
#ax.set_aspect('equal')

plt.show()
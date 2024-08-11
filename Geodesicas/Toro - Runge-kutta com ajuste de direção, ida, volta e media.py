import numpy as np
import matplotlib.pyplot as plt


#Plot do Toro
fig = plt.figure()
ax = fig.add_subplot(projection='3d')


u = np.linspace(0,2*np.pi,100)
v = np.linspace(0,2*np.pi,100)
a=3
r=1

def t(u):
  return (a+r*np.cos(u))

x = np.outer(t(u),np.cos(v))
y = np.outer(t(u),np.sin(v))
z = np.outer(np.sin(u),r*np.ones(np.size(v)))

ax.plot_surface(x,y,z, alpha=0.3)



#Índices dos pontos inicial e final da geodesica
indice_inicial=6
indice_final=17

#pontos do domínio por distribuiço uniforme
rng = np.random.default_rng(12345)
n_pontos=500

pontos_u = 0.3*rng.standard_normal(n_pontos)
pontos_v = 0.3*rng.standard_normal(n_pontos)


#passando pontos pela parametrização e colocando-os numa matriz
pontos_R3=np.zeros((n_pontos,3))

for i in range(len(pontos_u)):
  x = t(pontos_u[i])*np.cos(pontos_v[i])
  y = t(pontos_u[i])*np.sin(pontos_v[i])
  z = np.sin(pontos_u[i])

  pontos_R3[i,:]=np.array([x,y,z])
  
  if i==indice_inicial or i==indice_final:
    ax.scatter(x,y,z,color='yellow')


#Geodésica Aproximada por Runge-Kutta
def runge_kutta_toro(indice_inicial,indice_final):
  #pontos do domínio por distribuiço normal
  rng = np.random.default_rng(12345)
  n_pontos=500
  
  pontos_u = 0.3*rng.standard_normal(n_pontos)
  pontos_v = 0.3*rng.standard_normal(n_pontos)

  #Expressões das equações geodésicas
  def df(u,v,f,g):
   return -(g**2)*(t(u)*np.sin(u))*(1/r)

  def dg(u,v,f,g):
    return 2*f*g*(r*np.sin(u)/t(u))

  #Ponto inicial e direções no domínio e em R3
  a1 = pontos_u[indice_inicial]    
  b1 = pontos_v[indice_inicial]     

  a2 = pontos_u[indice_final]-pontos_u[indice_inicial] 
  b2 = pontos_v[indice_final]-pontos_v[indice_inicial] 

  direcao = np.array([a2,b2]) / np.linalg.norm(np.array([a2,b2]))
  angulo_inicial = np.arctan2(direcao[1],direcao[0])
  

  ponto_inicial= pontos_R3[indice_inicial,:]
  ponto_final = pontos_R3[indice_final,:]
  dir_R3 = (ponto_final - ponto_inicial) / np.linalg.norm(ponto_final - ponto_inicial)

  #Primeira iteração do runge kutta para determinar o primeiro ajuste para o angulo
  X=np.array([a1,b1,np.cos(angulo_inicial),np.sin(angulo_inicial)])
  cos_parada=2
  h=0.001
  pf_ate_geod=1
  erro_aceito=0.01

  listax=np.array([])
  listay=np.array([])
  listaz=np.array([])

  while abs(cos_parada)>0.2 and pf_ate_geod>erro_aceito:
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

    while abs(cos_parada)>0.2 and pf_ate_geod>erro_aceito:
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
  erro_aceito=0.01
  pf_ate_geod=1
  while pf_ate_geod>erro_aceito:

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

    while abs(cos_parada)>0.2 and pf_ate_geod>erro_aceito:
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

  return listax,listay,listaz,matriz_geod

x,y,z,matriz_geod_ida = runge_kutta_toro(indice_inicial,indice_final)
ax.plot(x,y,z,color='green')

x,y,z,matriz_geod_volta = runge_kutta_toro(indice_final,indice_inicial)
ax.plot(x,y,z,color='red')



   #Fazendo um terceira geodésica com a média das duas primeiras

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

#Fazendo a média e plotando
matriz_geod_media = np.array([[0,0,0]])
par = np.linspace(0,1,pontos_ida)

for x in range(pontos_ida-1):
  ponto_medio = matriz_geod_ida[x,:]*(1-par[x]) + matriz_geod_volta_inv[x,:]*par[x]
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




#Vizualização
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.set_zlim(-2,2)

plt.show()
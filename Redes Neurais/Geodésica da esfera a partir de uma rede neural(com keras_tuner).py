import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

#Plot da Esfera
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax2 = plt.figure(2).add_subplot(projection='3d')

 #Malha do domínio
n=500
u = np.linspace(0,np.pi,n)
v = np.linspace(0,2*np.pi,n)

 #passando pela parametrização
x = np.outer(np.sin(u),np.cos(v))
y = np.outer(np.sin(u),np.sin(v))
z = np.outer(np.cos(u),np.ones(np.size(v)))

ax.plot_surface(x,y,z, alpha=0.3)
ax2.plot_surface(x,y,z, alpha=0.3)



#Rede neural
modelo = keras.models.load_model('Rede com keras_tuner para a esfera.keras')
modelo.load_weights('pesos rede com keras_tuner/.weights.h5')


#Colocando inputs no formato da rede
u = u.reshape(n,1)
v = v.reshape(n,1)
Input = np.concatenate((u,v),axis=1).reshape(n,1,2)

#Colocando outputs no formato da rede
xo=np.cos(u)*np.cos(v)
yo=np.cos(u)*np.sin(v)
zo=np.sin(u)
Output = np.concatenate((xo,yo,zo),axis=1).reshape(n,1,3)




#amostra do domínio por distribuiço uniforme
rng = np.random.default_rng(12345)
n_pontos=500

pontos_u = rng.uniform(0.1,np.pi-0.1,n_pontos)
pontos_v = rng.uniform(-np.pi/2,np.pi/2,n_pontos)

#passando pontos pela parametrização e colocando-os numa matriz
pontos_R3=np.zeros((n_pontos,3))

for i in range(len(pontos_u)):
  x1 = np.sin(pontos_u[i])*np.cos(pontos_v[i])
  y1 = np.sin(pontos_u[i])*np.sin(pontos_v[i])
  z1 = np.cos(pontos_u[i])

  pontos_R3[i,:]=np.array([x1,y1,z1])
  
  #if i%5==0:
  ax.scatter(x1,y1,z1, color='pink')




#Função que retorna os símbolos de Christoffel aproximados num ponto
def Simbolos_de_Christoffel(p,modelo):

  tensor_x = tf.constant(tf.raw_ops.Reshape(tensor= p, shape=(1,1,2)))  #colocar ponto p na forma de entradda da rede
  with tf.GradientTape(persistent=True) as t1:
    t1.watch(tensor_x)
    with tf.GradientTape() as t2:
       t2.watch(tensor_x)
       y=modelo(tensor_x)
  
    j = t2.jacobian(y,tensor_x) #Jacobiana

    del_y_del_x1_i=tf.raw_ops.Reshape(tensor= j[:,:,:,:,:,0],shape= (1,3)) #Derivadas parciais da 'parametrização'
    del_y_del_x2_i=tf.raw_ops.Reshape(tensor= j[:,:,:,:,:,1],shape= (1,3))

    E = tf.matmul(del_y_del_x1_i,tf.transpose(del_y_del_x1_i))[0,0] #Coeficientes da primeira forma
    F = tf.matmul(del_y_del_x1_i,tf.transpose(del_y_del_x2_i))[0,0]
    G = tf.matmul(del_y_del_x2_i,tf.transpose(del_y_del_x2_i))[0,0]
    
  dE_dx = tf.raw_ops.Reshape(tensor= t1.gradient(E,tensor_x), shape=(2))  #Derivadas da primeira forma
  dF_dx = tf.raw_ops.Reshape(tensor= t1.gradient(F,tensor_x), shape=(2))
  dG_dx = tf.raw_ops.Reshape(tensor= t1.gradient(G,tensor_x), shape=(2))


  t = 2*(E*G-F**2) #Expressão que se repete nos simbolos de christoffel

  SC_111 = (G*dE_dx[0] - 2*F*dF_dx[0] + F*dE_dx[1])/t  #Símbolos de Christoffel
  SC_211 = (2*E*dF_dx[0] - E*dE_dx[1] - F*dE_dx[0])/t
  SC_112 = (G*dE_dx[1] - F*dG_dx[0])/t
  SC_212 = (E*dG_dx[0] - F*dE_dx[1])/t
  SC_122 = (2*G*dF_dx[1] - G*dG_dx[0] - F*dG_dx[1])/t
  SC_222 = (E*dG_dx[1] - 2*F*dF_dx[1] + F*dG_dx[0])/t

  return SC_111,SC_211,SC_112,SC_212,SC_122,SC_222



#Geodésica Aproximada por Runge-Kutta
def runge_kutta_esfera_rede(indice_inicial,indice_final,modelo):

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
  

  #Primeira iteração do runge kutta para determinar o primeiro ajuste para o angulo
  X=np.array([a1,b1,np.cos(angulo_inicial),np.sin(angulo_inicial)])
  h=0.01 #tamanho do passo
  pf_ate_geod=2
  passos=0

  listax=np.array([])
  listay=np.array([])   #listas para salvar a geodésica por coordenadas
  listaz=np.array([])

  while passos<=200:  #a geodésica para se o angulo estiver próximo a 90graus ou se passar perto do ponto final que queremos
    print(pf_ate_geod)

    ui = X[0]
    vi = X[1]
    Input = np.array([ui,vi]).reshape(1,1,2)
    Output = modelo(Input)
    
    xi = Output[0,0,0]
    yi = Output[0,0,1]
    zi = Output[0,0,2]

    listax=np.append(listax,xi)
    listay=np.append(listay,yi)
    listaz=np.append(listaz,zi)

    ponto_geod=np.array([xi,yi,zi])

    v2 = ponto_geod-ponto_final    
    pf_ate_geod=np.linalg.norm(v2) #cálculo do erro

    
  
    #Símbolos de christofel no ponto
    SC_111,SC_211,SC_112,SC_212,SC_122,SC_222 = Simbolos_de_Christoffel(np.array([ui,vi]),modelo)
  
         #derivadas de segunda ordem das variáveis do domínio em relação ao parâmetro
    def df(u,v,f,g):
     return -((f**2)*SC_111 + 2*f*g*SC_112 + (g**2)*SC_122)
  
    def dg(u,v,f,g):
      return -((f**2)*SC_211 + 2*f*g*SC_212 + (g**2)*SC_222)


    k1=h*np.array([X[2],X[3],df(X[0],X[1],X[2],X[3]),dg(X[0],X[1],X[2],X[3])])
    Y=X+0.5*k1
  
    k2=h*np.array([Y[2],Y[3],df(Y[0],Y[1],Y[2],Y[3]),dg(Y[0],Y[1],Y[2],Y[3])])
    Y=X+0.5*k2
                                                                                                   #Atualização do ponto e da direção no domínio (os pontos são atualizados com as derivadas de primeira ordem e as direções com as de segunda ordem)
    k3=h*np.array([Y[2],Y[3],df(Y[0],Y[1],Y[2],Y[3]),dg(Y[0],Y[1],Y[2],Y[3])])
    Y=X+k3
  
    k4=h*np.array([Y[2],Y[3],df(Y[0],Y[1],Y[2],Y[3]),dg(Y[0],Y[1],Y[2],Y[3])])
    X=X+(1/6)*(k1+2*k2+2*k3+k4)

    passos+=1

  return listax,listay,listaz




#Índices dos pontos inicial e final no grafo
indice_inicial=6
indice_final=10

#geodésica
x,y,z = runge_kutta_esfera_rede(indice_inicial,indice_final,modelo)
ax.plot(x,y,z,color='red',label='\'Geodésica\' a partir da rede')





#plot do fit da rede
rng2 = np.random.default_rng(1)
n=500
ut=rng.uniform(0,2*np.pi,n)
vt=rng.uniform(0,2*np.pi,n)

In_teste=np.zeros((n,1,2))
for i in range(n):
    In_teste[i,0,0]=ut[i]
    In_teste[i,0,1]=vt[i]

Out_teste = modelo.predict(In_teste)

x_pred=np.zeros(n)
y_pred=np.zeros(n)
z_pred=np.zeros(n)

for i in range(n):
    x_pred[i]=Out_teste[i,0,0]
    y_pred[i]=Out_teste[i,0,1]
    z_pred[i]=Out_teste[i,0,2]

ax2.scatter(x_pred,y_pred,z_pred, color='red', label='Fit da rede') 






#####Plot da geodésica pela parametrização



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

  while abs(cos_parada)>0.1 and pf_ate_geod>0.01:  #a geodésica para se o angulo estiver próximo a 90graus ou se passar perto do ponto final que queremos
    
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

    while abs(cos_parada)>0.1 and pf_ate_geod>0.1 and pf_ate_geod<2:
      
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
#ax.plot(x,y,z,color='green')

x,y,z,matriz_geod_volta = runge_kutta_esfera(indice_final,indice_inicial)
#ax.plot(x,y,z,color='red')



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

ax.plot(x,y,z, color='purple',label='\'Geodésica\' a partir do Runge-kutta')





ax2.legend()
ax.legend()
plt.show()
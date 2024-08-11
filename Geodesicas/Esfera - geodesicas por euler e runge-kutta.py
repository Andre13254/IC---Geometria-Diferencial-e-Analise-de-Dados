import numpy as np
import matplotlib.pyplot as plt

#Esfera
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

#Superfície
u = np.linspace(0,np.pi,100)
v = np.linspace(0,2*np.pi,100)

x = np.outer(np.sin(u),np.cos(v))
y = np.outer(np.sin(u),np.sin(v))
z = np.outer(np.cos(u),np.ones(np.size(v)))

ax.plot_surface(x,y,z, alpha=0.6)

#Ponto inicial e direção no domínio
a1 =0.5+ np.pi*np.random.rand()
b1 =0.5 + np.pi*np.random.rand()

a2 = np.pi*np.random.rand()
b2 = np.pi*np.random.rand()

#Método de Euler
X=np.array([a1,b1,a2,b2])
h=0.001
i=0
listax=np.array([])
listay=np.array([])
listaz=np.array([])

while i<=500:
  xi=np.array([np.sin(X[0])*np.cos(X[1])])
  yi=np.array([np.sin(X[0])*np.sin(X[1])])
  zi=np.array([np.cos(X[0])])

  listax=np.append(listax,xi)
  listay=np.append(listay,yi)
  listaz=np.append(listaz,zi)

  dX=np.array([X[2],X[3],X[3]*np.sin(X[0])*np.cos(X[0]),-2*X[2]*X[3]*((np.cos(X[0]))/np.sin(X[0]))])
  X= X + h*dX

  #vetores do Plano(pega os dois primmeiros pontos do Euler)
  if i==0:
    w=np.array([xi[0],yi[0],zi[0]])
    ax.scatter(xi,yi,zi,color='yellow')
  if i==1:
    s=np.array([xi[0],yi[0],zi[0]])
  if i==2000:
    ax.scatter(xi,yi,zi,color='yellow')

  i+=1

#Plot da geodésica pelo metodo de euler
ax.plot(listax,listay,listaz,linestyle='dashed',color='green',label='euler')


#Método de Runge-Kutta
X=np.array([a1,b1,a2,b2])
h=0.001
i=0
listax2=np.array([])
listay2=np.array([])
listaz2=np.array([])

while i<=500:
  xi=np.array([np.sin(X[0])*np.cos(X[1])])
  yi=np.array([np.sin(X[0])*np.sin(X[1])])
  zi=np.array([np.cos(X[0])])

  listax2=np.append(listax2,xi)
  listay2=np.append(listay2,yi)
  listaz2=np.append(listaz2,zi)


  k1=h*np.array([X[2],X[3],X[3]*np.sin(X[0])*np.cos(X[0]),-2*X[2]*X[3]*(np.cos(X[0])/np.sin(X[0]))])
  Y1=X+0.5*k1

  k2=h*np.array([Y1[2],Y1[3],Y1[3]*np.sin(Y1[0])*np.cos(Y1[0]),-2*Y1[2]*Y1[3]*(np.cos(Y1[0])/np.sin(Y1[0]))])
  Y2=X+0.5*k2

  k3=h*np.array([Y2[2],0.1*Y2[3],Y2[3]*np.sin(Y2[0])*np.cos(Y2[0]),-2*Y2[2]*Y2[3]*(np.cos(Y2[0])/np.sin(Y2[0]))])
  Y3=X+k3

  k4=h*np.array([Y3[2],Y3[3],Y3[3]*np.sin(Y3[0])*np.cos(Y3[0]),-2*Y3[2]*Y3[3]*(np.cos(Y3[0])/np.sin(Y3[0]))])
  X=X+(1/6)*(k1+2*k2+2*k3+k4)


  if i==2000:
    ax.scatter(xi,yi,zi,color='yellow')

  #vetores do Plano(pega os dois primeiros pontos do Runge-kutta)
  if i==0:
    w=np.array([xi[0],yi[0],zi[0]])
    ax.scatter(xi,yi,zi,color='yellow')
  if i==1:
    s=np.array([xi[0],yi[0],zi[0]])
  if i==2000:
    ax.scatter(xi,yi,zi,color='yellow')

  i+=1


#Plot da geodésica pelo metodo de runge-kutta
ax.plot(listax2,listay2,listaz2,color='red',label='runge-kutta')


#Plano
wx,wy,wz=w[0],w[1],w[2]
sx,sy,sz=s[0],s[1],s[2]

u1=np.linspace(-1.5,1.5,100)
v1=np.linspace(-1.5,1.5,100)
X,Y=np.meshgrid(u1,v1)

def f(x,y):
  return wz*(x-sx*((y-x*wy/wx)/(sy-sx*wy/wx)))/wx + sz*((y-x*wy/wx)/(sy-sx*wy/wx))

Z=f(X,Y)

ax.plot_surface(X,Y,Z,color='green',alpha=0.4)

#Vizualização
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)


#Cálculo do erro

 #Pontos da geodésica aproximada numa matriz
geodesica_euler=np.array([listax,listay,listaz])
geodesica_rk=np.array([listax2,listay2,listaz2])
 #Pontos da Geodésica exata numa matriz

plt.legend()
plt.show()


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

ax.plot_surface(x,y,z, alpha=0.6)


#Geodésica Aproximada por Runge-Kutta

  #Ponto inicial e direções no domínio
a1 = 10*np.random.rand()
b1 = 10*np.random.rand()

a2 = 2*np.pi*np.random.rand()
b2 = 2*np.pi*np.random.rand()

  #Expressões das equações geodésicas
def df(u,v,f,g):
 return -(g**2)*(t(u)*np.sin(u))*(1/r)

def dg(u,v,f,g):
  return 2*f*g*(r*np.sin(u)/t(u))



#Método de Runge-Kutta
X=np.array([a1,b1,np.cos(a2),np.sin(b2)])
i=0
h=0.001
n=1000
listax=np.array([])
listay=np.array([])
listaz=np.array([])
while i<=n:
  xi=np.array([t(X[0])*np.cos(X[1])])
  yi=np.array([t(X[0])*np.sin(X[1])])
  zi=np.array([r*np.sin(X[0])])    

  listax=np.append(listax,xi)                  
  listay=np.append(listay,yi)
  listaz=np.append(listaz,zi)

  k1=h*np.array([X[2],X[3],df(X[0],X[1],X[2],X[3]),dg(X[0],X[1],X[2],X[3])])
  Y=X+0.5*k1
  k2=h*np.array([Y[2],Y[3],df(Y[0],Y[1],Y[2],Y[3]),dg(Y[0],Y[1],Y[2],Y[3])])
  Y=X+0.5*k2
  k3=h*np.array([Y[2],Y[3],df(Y[0],Y[1],Y[2],Y[3]),dg(Y[0],Y[1],Y[2],Y[3])])
  Y=X+k3
  k4=h*np.array([Y[2],Y[3],df(Y[0],Y[1],Y[2],Y[3]),dg(Y[0],Y[1],Y[2],Y[3])])
  X=X+(1/6)*(k1+2*k2+2*k3+k4)

  if i==1 or i==n:
    ax.scatter(xi,yi,zi,color='yellow')

  i+=1

ax.plot(listax,listay,listaz,color='red')


ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.set_zlim(-2,2)

plt.show()
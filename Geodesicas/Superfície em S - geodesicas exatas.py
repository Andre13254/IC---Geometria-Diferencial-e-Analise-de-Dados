import numpy as np
import matplotlib.pyplot as plt

#Superfície em S
fig = plt.figure()
ax = fig.add_subplot(projection='3d')


u = np.linspace(-np.pi,2*np.pi,100)
v = np.linspace(0,5,100)


pi = np.pi

x = np.outer(np.cos(u),np.ones(np.size(u)))
y = np.outer(np.ones(np.size(v)),v)
z = np.outer((np.sin(u)-1)*(pi/2 - u)/(abs(pi/2 - u)),np.ones(np.size(v)))

ax.plot_surface(x,y,z, alpha=0.6)



#Geodésica Aproximada(exata, pois essa superfície é isométrica ao plano) por Euler

  #Ponto inicial e direções no domínio
a1 = 3.2*np.random.rand()
b1 = 5*np.random.rand()
a2 = 2*np.pi*np.random.rand()


#Método de Euler
X=np.array([a1,b1,np.cos(a2),np.sin(a2)])
h=0.01
i=0
listax=np.array([])
listay=np.array([])
listaz=np.array([])

while i<=200:
  xi=np.array([np.cos(X[0])])
  yi=np.array([X[1]])
  zi=np.array([(np.sin(X[0])-1)*((pi/2 - X[0])/(abs(pi/2 - X[0])))])

  listax=np.append(listax,xi)
  listay=np.append(listay,yi)
  listaz=np.append(listaz,zi)

  dX=np.array([X[2],X[3],0,0])
  X= X + h*dX

  if i==1 or i==200:
    ax.scatter(xi,yi,zi, color='yellow')
  
  i+=1
ax.plot(listax,listay,listaz,color='red')

plt.show()
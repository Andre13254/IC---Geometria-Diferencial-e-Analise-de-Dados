import numpy as np
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt





#Amostra de pontos da esfera
rng = np.random.default_rng(0)
n=2000
u = rng.uniform(0,2*np.pi,n)
v = rng.uniform(0,2*np.pi,n)

x=np.cos(u)*np.cos(v)
y=np.cos(u)*np.sin(v)
z=np.sin(u)

f1 = plt.figure(1).add_subplot(projection='3d')
f1.scatter(x,y,z)


#colocando os pontos do domínio no formato de entrada da rede
Input=np.zeros((n,1,2))
for i in range(n):
    Input[i,0,0]=u[i]
    Input[i,0,1]=v[i]


#Colocando os pontos da imagem no formato de saída da rede
Output=np.zeros((n,1,3))
for i in range(n):
    Output[i,0,0]=x[i]
    Output[i,0,1]=y[i]
    Output[i,0,2]=z[i]


#Rede neural
def rede():
    inputs=keras.Input(shape=(1,2))
    dense1=layers.Dense(10,activation='elu')
    dense2=layers.Dense(10,activation='elu')
    dense3=layers.Dense(10,activation='elu')
    dense4=layers.Dense(3,activation='linear')
    outputs=dense4(dense3(dense2(dense1(inputs))))

    model=keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss="mse")
    return  model


modelo=rede()



#Salvando e carregando a rede e os pesos
modelo.save('Rede para a esfera.keras')

modelo = keras.models.load_model('Rede para a esfera.keras')
history = modelo.fit(Input,Output, epochs=500, verbose=1)
#modelo.save_weights('.weights.h5', overwrite=True)

print(modelo.summary())

#gráfico da função de erro
f2=plt.figure(2)
loss = history.history['loss']
plt.plot(loss)



#amostra de teste
rng2 = np.random.default_rng(1)
n=500
ut=rng.uniform(0,2*np.pi,n)
vt=rng.uniform(0,2*np.pi,n)

In_teste=np.zeros((n,1,2))
for i in range(n):
    In_teste[i,0,0]=ut[i]
    In_teste[i,0,1]=vt[i]

#Previsão da rede
Out_teste = modelo.predict(In_teste)

x_pred=np.zeros(n)
y_pred=np.zeros(n)
z_pred=np.zeros(n)

for i in range(n):
    x_pred[i]=Out_teste[i,0,0]
    y_pred[i]=Out_teste[i,0,1]
    z_pred[i]=Out_teste[i,0,2]

f1.scatter(x_pred,y_pred,z_pred, color='red') 

plt.show()

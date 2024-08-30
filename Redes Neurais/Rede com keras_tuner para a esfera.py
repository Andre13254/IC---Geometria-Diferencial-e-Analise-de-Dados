#faremos uma rede para aproximar a esfera com o keras tuner para decidir parte da arquitetura


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras import layers
import keras_tuner as kt

#Amostra de pontos da esfera
rng = np.random.default_rng(0)
n=2000
u = rng.uniform(0,np.pi,n)
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

#Dividindo a amostra entre treino e teste
in_treino, in_teste, out_treino, out_teste = train_test_split(Input, Output, test_size=0.2, shuffle=False)



#Criando o modelo
def build_model(hp):
    model = keras.Sequential() 

    model.add(layers.Dense(2, activation=hp.Choice('activation',['elu','selu','gelu','tanh'])))      
    for i in range(hp.Int('num_layers',1,6,step=1)):
      model.add(layers.Dense(hp.Int(f'units_{i}',32,512,step=8), 
                             activation=hp.Choice('activation',['elu','selu','gelu','tanh'])))
    model.add(layers.Dense(3,activation=hp.Choice('activation',['elu','selu','gelu','tanh'])))
    
    lr=hp.Float('lr',1e-4,1e-1, sampling='log')

    model.compile(optimizer=hp.Choice('optimizer',['adam','sgd']),
                  loss='mse',
                  metrics=['accuracy'])

    return model

#Ajustando o modelo
tuner = kt.BayesianOptimization(hypermodel=build_model, 
                                objective='val_accuracy',
                                max_trials=100,
                                overwrite=False,
                                project_name='Keras_tuner para a esfera')

tuner.search(in_treino,out_treino, epochs=30, validation_data=(in_teste,out_teste))


#Salvando o melhor modelo
models = tuner.get_best_models(num_models=1)
best_model = models[0]
history = best_model.fit(in_treino,out_treino, epochs=50,verbose=1)
print(best_model.summary())
print(history.history.keys())

#Plotando predição do modelo na amostra de teste
out_pred = best_model.predict(in_teste)

n_teste=len(in_teste)

x_pred=np.zeros(n_teste)
y_pred=np.zeros(n_teste)
z_pred=np.zeros(n_teste)

for i in range(n_teste):
    x_pred[i]=out_pred[i,0,0]
    y_pred[i]=out_pred[i,0,1]
    z_pred[i]=out_pred[i,0,2]
f1.scatter(x_pred,y_pred,z_pred, color='red')


#Plot das funções de erro e acurácia
loss = history.history['loss']
acc = history.history['accuracy']

f2 = plt.figure(2).add_subplot()
f2.plot(loss, label='Erro no teste')
f2.plot(acc, label='Precisão no teste')

plt.legend()
plt.show()

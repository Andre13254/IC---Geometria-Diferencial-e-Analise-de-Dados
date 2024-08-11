import numpy as np

grafo1 = np.array([[0,50,30,100,10],[50,0,5,20,-1],[30,5,0,50,-1],[100,20,50,0,10],[10,-1,-1,10,0]])
grafo2 = np.array([[0,35,3,-1,-1,-1],[16,0,6,-1,-1,21],[-1,-1,0,2,-1,7],[4,-1,2,0,-1,-1],[15,-1,-1,9,0,-1],[-1,5,-1,-1,-1,0]])

def dijkstra(grafo,s,t): #s é a linha e t a coluna
    antecessores=np.zeros(len(grafo[:,0]))
    distancias = -np.ones(len(grafo[:,0]))
    chave = np.ones(len(grafo[:,0])) #1 é temporário

    #Colocando a distancia de s ate s para zero e permanente
    for x in range(len(distancias)):
        if x==s:
            distancias[x]=0
            chave[x]=0

    #criando lista numerada de nós
    nos=[]
    for x in range(len(chave)):
        nos.append(x)

    p=s #p é o antecessor de i
    while p!=t:
        for i in nos:
            if chave[i]==1 and grafo[p,i]>=0:
                dist_atual=distancias[i]
                if dist_atual==-1:
                    distancias[i]=distancias[p]+grafo[p,i]
                else:
                    distancias[i]=min(dist_atual,distancias[p]+grafo[p,i])

                if dist_atual!=distancias[i]:
                    antecessores[i]=p

        #lista das distancias e indices temporarios
        dist_temporarias=[]
        indices_temporarios=[]
        for x in nos:
            if chave[x]==1 and distancias[x]>=0:
                dist_temporarias.append(distancias[x])
                indices_temporarios.append(x)


        for j in nos:
            if chave[j]==1 and distancias[j]==min(dist_temporarias):
                chave[j]=0
                p=j

    #montando o caminho a partir dos antecessores
    caminnho_inverso=[]
    x=t
    while x!=s:
      caminnho_inverso.append(x)
      x=int(antecessores[x])
    caminnho_inverso.append(s)

    caminho=[]
    for x in range(len(caminnho_inverso)-1,-1,-1):
        caminho.append(caminnho_inverso[x])

    return distancias[t],caminho



print(dijkstra(grafo2,4,5))


      


  
    







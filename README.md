Este repositório contém os programas feitos na iniciação científica sobre geometria diferencial e análise de dados realizada na UFS entre 2023 e 2024.

Aluno: André Luís de Almeida Sousa <br />
Orientador: Gastão Florêncio Miranda Júnior

O objetivo deste trabalho foi analisar formas de calcular distâncias entre pontos de dados de uma amostra discreta. Os métodos usados foram:

   1. Montar um grafo de vizinhaças com KNN e calcular caminhos mínimos no grafo com Dijkstra
   2. Aproximar geodésicas de superfícies com parametrização conhecida por Euler e Runge-Kutta de quarta ordem
   3. Usar uma rede neural(MLP) para achar uma função que poderia ser usada como parametrização. Isso permitiria usar o método de Runge-Kutta para casos em que não há parametrização(conjuntos de dados em geral).

A aplicação de redes neurais não funcionou. Aparentemente, o cálculo de derivadas automáticas aninhadas gera muito erro, o que não permitiu boas aproximações para geodésicas mesmo no caso simples da esfera. Por conta disso, o método não foi testado em conjuntos gerais de dados.

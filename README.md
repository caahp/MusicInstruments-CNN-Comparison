# **🎶 Reconhecimento de Instrumentos Musicais usando CNN: Transfer Learning vs Sem Transfer Learning**

Este projeto explora a aplicação de Redes Neurais Convolucionais (CNNs) no reconhecimento de instrumentos musicais. O objetivo é classificar imagens de sete instrumentos, incluindo violão, guitarra, trompete, flauta, bateria, piano e saxofone.

## **📁 Estrutura do Projeto**

O notebook está dividido em duas partes principais:

1. **CNN sem Transfer Learning:**  
   Uma arquitetura personalizada de CNN é criada e treinada a partir do zero com um conjunto de dados de instrumentos musicais. Não utilizamos modelos pré-treinados, o que torna o processo mais desafiador, mas também permite explorar a capacidade da rede de aprender diretamente com os dados.

2. **CNN com Transfer Learning:**  
   Modelos pré-treinados, como ResNet50, MobileNetV2, VGG16 e InceptionV3 podem empregados para aplicar Transfer Learning, utilizamos InceptionV3 neste. Isso reduz o tempo de treinamento e melhora o desempenho em comparação com as CNNs personalizadas, aproveitando conhecimentos adquiridos em grandes datasets como o ImageNet.

## **💡 Objetivo**
Este projeto visa comparar a eficácia entre redes CNN personalizadas e aquelas que utilizam Transfer Learning na tarefa de classificação de instrumentos musicais, além de destacar os ganhos em precisão e eficiência.

## **🚀 Como Utilizar**
1. Clone o repositório para sua máquina local.
2. Instale as dependências necessárias (TensorFlow, OpenCV, etc.).
3. Execute o notebook no Kaggle ou em um ambiente com suporte a GPUs para otimizar o tempo de execução.

O link para acesso ao conjunto de dados e execução no Kaggle será fornecido aqui:  
- [Dataset](#https://www.kaggle.com/datasets/caahps/instrumentsdataset/settings)
- [Modelo](#https://www.kaggle.com/code/caahps/musicinstrumentsmodels)

## **🖥️ Requisitos**
- Recomendamos o uso de uma GPU para treinar o modelo, pois pode ser computacionalmente intensivo.

## **🔍 Contribuições**
Estamos abertos a colaborações! Devido à complexidade do dataset e da tarefa, novas ideias para melhorar o desempenho e a eficiência são bem-vindas.

## **🏫 Sobre**
Este trabalho faz parte da disciplina **Tópicos Especiais em Deep Learning** da **Universidade do Estado do Amazonas (UEA)**, e foi desenvolvido com o intuito de explorar diferentes abordagens de Deep Learning aplicadas à visão computacional.

## **👥 Autores**
- [Caroline P. Souza](https://github.com/caahp)
- [Eric D. Perin](https://github.com/ericperinn)

# -*- coding: utf-8 -*-
"""
Métodos de ensamble

No existe un modelo superior metodologia NTFL?

busqueda de hyperparametros
multiples entrenamientos
analisis de neuronas

el metodo de ensamble se basa en entrenar multiples modelos con variedad de 
redes con el objetivo de concatenar la respuesta

Metodo de ensamble
* Linear Reegression -> 12.5
* SVM -> 11.2
* mlp -> 13.9
* Desicion tree -> 10.8

Los metodos de ensamble se dividen en metodos 
    Homogéneos 
    Heterogéneos


Los metodos de ensamble pueden ser por
    
    Votación (Media o promedio de los modelos)
        Aquellos los que le da la misma importancia a todos los modelos
        Eligiendo aquellos con la mayor cantidad de coincidencia o "Votos"
    
    Ponderación 
        El promedio ponderado 
    
    Predictores sobre predictores
    
    Mezclas de expertos


Bootstrapping es un metodo de muestreo
El resultado de combinar esto con los metodos de ensamble se conoce como Bagging

El randomForest es basicamente Baggin homogeneo donde todos los metodos son arboles de decisiones,
    Donde se procura que cada arbol sea diferente

Articulos a revisar sobre el tema
    
    A theory of the learnable (Valiant, 1984)
    
Strong vs weak learners

    The strenght of weak learnabl¿ility (Schapire 1990)
    
    
    
Tecnica: Boosting
    Se entrena usando nuevos modelos los cuales son creados con las fallas del modelo
    anterior permitiendo que se sobreentrene donde se equivoco para acertar,
    Tras ellos se usan los meotodos de ensamble para juntar los modelos resultantes
    
AdaBoost (Adaptiove boosting)
    Remuestreo de los datasets utilizando un sesgo enfocado en los errores como en boosting
    Mas pasa datos regulares con el objetivo de evitar un sobrenetrenamiento
    
GradientBoost
XGBoost

"""


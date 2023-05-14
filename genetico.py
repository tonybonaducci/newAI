import utilities
import math
import numpy as np
import random as rd
import time

tam_poblacion = 50
p_cruce_generacional = 0.7
p_cruce_estacionario = 1
p_mutacion = 0.1
alpha = 0.3

my_generator = np.random.default_rng()

#seeds = [my_generator.uniform(low=0.0,high=1.0,size=8) for i in range(tam_poblacion)]
#for i in range(tam_poblacion):
#   seeds[i] = my_generator.uniform(low=0.0,high=1.0,size=len(atributes))
#padres = seeds
#print(padres)


def cruce_aritmetico(cr_1,cr_2,n_atr):   
    h_1 = np.zeros(n_atr)
    h_2 = np.zeros(n_atr)
    alpha = my_generator.uniform(low=0.0,high=1.0)

    for i in range(n_atr):
        h_1[i] = alpha*cr_1[i] + (1-alpha)*cr_2[i]
        h_2[i] = alpha*cr_2[i] + (1-alpha)*cr_1[i]
    return h_1, h_2

#LOS HIJOS DADOS se pueden salir de [0,1]
def BLX_alpha(cr_1,cr_2,n_atr,alpha):
    h_1 = np.zeros(n_atr)
    h_2 = np.zeros(n_atr)

    for i in range(n_atr):
        c_max=max(cr_1[i],cr_2[i])
        c_min=min(cr_1[i],cr_2[i])
        I = c_max - c_min
        print("generamos en el intervalo: ",c_min - I*alpha," - ",c_max + I*alpha)
        h_1[i] = my_generator.uniform(low=c_min - I*alpha,high=c_max + I*alpha) 
        h_2[i] = my_generator.uniform(low=c_min - I*alpha,high=c_max + I*alpha)

        if h_1[i] < 0:
            h_1[i] = 0
        elif h_1[i] >1:
            h_1[i] = 1
        if h_2[i] < 0:
            h_2[i] = 0
        elif h_2[i] >1:
            h_2[i] = 1

###################### DUDA: tenemos que acotar en intervalo [0-1]? ################################
    print("hemos generado: ", h_1," y ",h_2) 

    return h_1, h_2


#sol_1,sol_2 = BLX_alpha(seeds[0],seeds[1],alpha)

#sol_3, sol_4 = cruce_artimetico(seeds[0],seeds[1])


#duda: (2,3) y (3,2) son elementos distintos?
#emparejamiento fijo? 

def torneo_binario(cr_1,cr_2,train_set, test_set, train_set_size, test_set_size,atributes):
        clasificacion_1 = utilities.weighted_1NN(train_set, test_set, train_set_size, test_set_size,cr_1,atributes)      
        f_f_1, c_r_1, r_r_1 = utilities.weighted_fitness_function(train_set,test_set,test_set_size,clasificacion_1,cr_1)
  
        clasificacion_2 = utilities.weighted_1NN(train_set, test_set, train_set_size, test_set_size,cr_2,atributes)      
        f_f_2, c_r_2, r_r_2 = utilities.weighted_fitness_function(train_set,test_set,test_set_size,clasificacion_2,cr_2)

        if(f_f_1 < f_f_2):
            return cr_2

        else:
            return cr_1

def seleccion_AGE(poblacion,train_set, test_set, train_set_size, test_set_size,atributes):

    orden_1 = rd.sample(range(tam_poblacion),2)
    orden_2 = rd.sample(range(tam_poblacion),2)
    orden = [orden_1,orden_2]

    padre_1 = torneo_binario(poblacion[orden_1[0]],poblacion[orden_1[1]],train_set, test_set, train_set_size, test_set_size,atributes)
    padre_2 = torneo_binario(poblacion[orden_2[0]],poblacion[orden_2[1]],train_set, test_set, train_set_size, test_set_size,atributes)

    nueva_poblacion = [torneo_binario(poblacion[orden[i][0]],poblacion[orden[i][1]],train_set, test_set, train_set_size, test_set_size,atributes) for i in range(2)]
    
    return padre_1, padre_2


def seleccion(poblacion,train_set, test_set, train_set_size, test_set_size,atributes):
    nueva_poblacion = poblacion
    orden = rd.sample(range(tam_poblacion),tam_poblacion)

    for i in range(tam_poblacion-1):
        nueva_poblacion[i] = torneo_binario(poblacion[orden[i]],poblacion[orden[i+1]],train_set, test_set, train_set_size, test_set_size,atributes)
    
    nueva_poblacion[tam_poblacion - 1] = torneo_binario(poblacion[orden[0]],poblacion[orden[tam_poblacion - 1]],train_set, test_set, train_set_size, test_set_size,atributes)
    return nueva_poblacion

###########NO ESTOY SEGURA DE QUE LA MUTACIÓN SE HAGA ASÍ###################

def mutacion_normal(cromosoma,gen,n_atr):
    z=rd.gauss(mu=0.0, sigma=0.3)
    mutacion = utilities.normalizar(cromosoma[gen]+z)

    return mutacion

def valor_cromosoma(cromosoma,train_set, test_set, train_set_size, test_set_size,atributes):

    clasificacion = utilities.weighted_1NN(train_set, test_set, train_set_size, test_set_size,cromosoma,atributes)      
    f_f, c_r, r_r = utilities.weighted_fitness_function(train_set,test_set,test_set_size,clasificacion,cromosoma)

    return f_f

def valor_poblacion(seeds,train_set, test_set, train_set_size, test_set_size,atributes): 

    print("Valoramos la población: ",seeds)
    mejor = 0.0
    peor = 0.0
    indice_mejor = 0
    indice_peor = 0

    for i in range(tam_poblacion):
        print(i)
        clasificacion = utilities.weighted_1NN(train_set, test_set, train_set_size, test_set_size,seeds[i],atributes)      
        f_f, c_r, r_r = utilities.weighted_fitness_function(train_set,test_set,test_set_size,clasificacion,seeds[i])
        
        if(mejor < f_f):
            mejor = f_f
            indice_mejor = i 
        
        elif(peor > f_f):
            peor = f_f
            indice_peor = i

    return mejor, indice_mejor, indice_peor

def valor_poblacion_age(seeds,train_set, test_set, train_set_size, test_set_size,atributes):
    print("Valoramos la población: ",seeds)
    peor = 0.0
    segundo_peor = 0.0
    indice_peor = 0
    indice_segundo_peor = 0


    for i in range(tam_poblacion):
        print(i)
        clasificacion = utilities.weighted_1NN(train_set, test_set, train_set_size, test_set_size,seeds[i],atributes)      
        f_f, c_r, r_r = utilities.weighted_fitness_function(train_set,test_set,test_set_size,clasificacion,seeds[i])
        
        if( mejor < f_f):
            mejor = f_f
            indice_mejor = i

        if(peor > f_f):
            peor = f_f
            indice_peor = i 
        
        elif(segundo_peor > f_f):
            segundo_peor = f_f
            indice_segundo_peor = i

    return mejor, peor, segundo_peor, indice_mejor, indice_peor, indice_segundo_peor

def valor_cromosoma(seed,train_set, test_set, train_set_size, test_set_size,atributes):
    
    clasificacion = utilities.weighted_1NN(train_set, test_set, train_set_size, test_set_size,seed,atributes)      
    f_f, c_r, r_r = utilities.weighted_fitness_function(train_set,test_set,test_set_size,clasificacion,seed)
    
    return f_f  

def AGG_BLX(train_set, test_set, train_set_size, test_set_size,weights,atributes):

    poblacion = [my_generator.uniform(low=0.0,high=1.0,size=8) for i in range(tam_poblacion)]

    n_esperado_cruces =  int(p_cruce_generacional*tam_poblacion)
    n_esperado_mutaciones = int(p_mutacion*tam_poblacion*8)
    p_intermedia = seleccion(poblacion,train_set, test_set, train_set_size, test_set_size,atributes) 
    #print(ordenacion)
    mejor_fit, indice_mejor, indice_peor = valor_poblacion(poblacion,train_set, test_set, train_set_size, test_set_size,atributes)

    n_evaluaciones = 0
    while(n_evaluaciones<5):
        p_intermedia = seleccion(poblacion,train_set, test_set, train_set_size, test_set_size,atributes) 

        print("Partimos de la población:",p_intermedia)
        #----------cruce---------- 
        for i in range(n_esperado_cruces-1):

            #h_1 = np.zeros(len(atributes))
            #h_2 = np.zeros(len(atributes))

            n_atr = len(atributes)
            h_1, h_2 =  BLX_alpha(p_intermedia[i],p_intermedia[i+1],n_atr,alpha)
            print("cruce de ",p_intermedia[i]," ",p_intermedia[i+1],""," genera: ",h_1,h_2)
            p_intermedia[i],p_intermedia[i+1] = h_1, h_2

        #----------mutación-------
        for i in range(n_esperado_mutaciones):
                cromosoma = rd.randint(0,tam_poblacion-1)
                #print(cromosoma)
                gen = rd.randint(0,n_atr-1)
                #print(gen)
                print("mutacion gen: ",gen," cromosoma: ",cromosoma)
                p_intermedia[cromosoma][gen] = mutacion_normal(p_intermedia[cromosoma],gen,n_atr)
                print("resultado mutacion: ",p_intermedia[cromosoma])
        
        #------reemplazamiento----
        #en caso de que nuestra mejor solucion empeore, sustituir una solucion de la nueva generacion por la mejor de la generacion anterior.
        nuevo_fit, nuevo_i_mejor, nuevo_i_peor = valor_poblacion(p_intermedia,train_set, test_set, train_set_size, test_set_size,atributes)
        n_evaluaciones = n_evaluaciones + 1
        if(nuevo_fit < mejor_fit):
            print("nuevo fit < mejor fit: ", nuevo_fit," < ",mejor_fit<" dado por el cromosoma: ",p_intermedia[nuevo_i_peor])
            p_intermedia[nuevo_i_peor] = poblacion[indice_mejor]
            indice_mejor = nuevo_i_peor

        else: 
            mejor_fit = nuevo_fit
            indice_mejor = nuevo_i_mejor

        
        poblacion = p_intermedia

    return poblacion, mejor_fit, indice_mejor

def AGG_aritmetico(train_set, test_set, train_set_size, test_set_size,weights,atributes):

    poblacion = [my_generator.uniform(low=0.0,high=1.0,size=8) for i in range(tam_poblacion)]

    n_esperado_cruces =  int(p_cruce_generacional*tam_poblacion)
    n_esperado_mutaciones = int(p_mutacion*tam_poblacion*8)
    p_intermedia = seleccion(poblacion,train_set, test_set, train_set_size, test_set_size,atributes) 
    #print(ordenacion)
    mejor_fit, indice_mejor, indice_peor = valor_poblacion(poblacion,train_set, test_set, train_set_size, test_set_size,atributes)

    n_evaluaciones = 0
    while(n_evaluaciones<5):
        p_intermedia = seleccion(poblacion,train_set, test_set, train_set_size, test_set_size,atributes) 

        print("Partimos de la población:",p_intermedia)
        #----------cruce---------- 
        for i in range(n_esperado_cruces-1):

            #h_1 = np.zeros(len(atributes))
            #h_2 = np.zeros(len(atributes))

            n_atr = len(atributes)
            h_1, h_2 =  cruce_aritmetico(p_intermedia[i],p_intermedia[i+1],n_atr)
            print("cruce de ",p_intermedia[i]," ",p_intermedia[i+1],""," genera: ",h_1,h_2)
            p_intermedia[i],p_intermedia[i+1] = h_1, h_2

        #----------mutación-------
        for i in range(n_esperado_mutaciones):
                cromosoma = rd.randint(0,tam_poblacion-1)
                #print(cromosoma)
                gen = rd.randint(0,n_atr-1)
                #print(gen)
                print("mutacion gen: ",gen," cromosoma: ",cromosoma)
                p_intermedia[cromosoma][gen] = mutacion_normal(p_intermedia[cromosoma],gen,n_atr)
                print("resultado mutacion: ",p_intermedia[cromosoma])
        
        #------reemplazamiento----
        #en caso de que nuestra mejor solucion empeore, sustituir una solucion de la nueva generacion por la mejor de la generacion anterior.
        nuevo_fit, nuevo_i_mejor, nuevo_i_peor = valor_poblacion(p_intermedia,train_set, test_set, train_set_size, test_set_size,atributes)
        n_evaluaciones = n_evaluaciones + 1
        if(nuevo_fit < mejor_fit):
            print("nuevo fit < mejor fit: ", nuevo_fit," < ",mejor_fit<" dado por el cromosoma: ",p_intermedia[nuevo_i_peor])
            p_intermedia[nuevo_i_peor] = poblacion[indice_mejor]
            indice_mejor = nuevo_i_peor

        else: 
            mejor_fit = nuevo_fit
            indice_mejor = nuevo_i_mejor

        
        poblacion = p_intermedia

    return poblacion, mejor_fit, indice_mejor

def AGE_BLX(train_set, test_set, train_set_size, test_set_size,weights,atributes):

    n_atr = len(atributes)
    poblacion = [my_generator.uniform(low=0.0,high=1.0,size=n_atr) for i in range(tam_poblacion)]

    n_esperado_cruces =  int(p_cruce_generacional*tam_poblacion)
    n_esperado_mutaciones = int(p_mutacion*2*n_atr)
    ordenacion = seleccion()  
    p_intermedia = [np.zeros(n_atr) for i in range(2)]
 
    #print(ordenacion)

    mejor_fit, indice_mejor, indice_peor = valor_poblacion(poblacion,train_set, test_set, train_set_size, test_set_size,atributes)

    n_evaluaciones = 0
    while(n_evaluaciones<5):
        print("Partimos de la población:",poblacion)
        #--------seleccion--------
        padre_1, padre_2 = seleccion_AGE(poblacion,train_set, test_set, train_set_size, test_set_size,atributes)
        print("Seleccionamos padres: ",padre_1,padre_2)
        #----------cruce---------- 
        #se cruzan solo los dos padres que tenemos 
        c_1, c_2 = BLX_alpha(padre_1,padre_2,n_atr,alpha) 
        print("cruce de ",padre_1," ",padre_2,""," genera: ",c_1,c_2)
        p_intermedia[0] = c_1
        p_intermedia[1] = c_2

        print("Poblacion intermedia: ",p_intermedia)

        #---------mutación--------
        #no sé cómo funciona 
        for i in range(n_esperado_mutaciones):
                cromosoma = rd.randint(0,1)
                #print(cromosoma)
                gen = rd.randint(0,n_atr-1)
                #print(gen)
                print("mutacion gen: ",gen," cromosoma: ",cromosoma)
                p_intermedia[cromosoma][gen] = mutacion_normal(p_intermedia,cromosoma,n_atr)
                print("resultado mutacion: ",p_intermedia[cromosoma])

        #-----reemplazamiento-----
        #los dos hijos compiten para sustituir a el/los peores de la población
        
        mejor_fit, peor, segundo_peor, i_mejor, i_peor, i_segundo_peor = valor_poblacion(poblacion,train_set, test_set, train_set_size, test_set_size,atributes)
        n_evaluaciones = n_evaluaciones + 1

        v_hijo_1 = valor_cromosoma(p_intermedia[0],train_set, test_set, train_set_size, test_set_size,atributes)
        v_hijo_2 = valor_cromosoma(p_intermedia[1],train_set, test_set, train_set_size, test_set_size,atributes)
        
        print("Mejor: ",i_mejor,"-",mejor," Peor: ",i_peor,"-",peor, " Segundo peor: ",i_segundo_peor,"-",segundo_peor)
        print("Hijo_1: ",v_hijo_1," Hijo_2: ",v_hijo_2)
        n_evaluaciones = n_evaluaciones + 2

        if(v_hijo_1 < v_hijo_2):
            if(v_hijo_1 < peor):
            
                poblacion[i_peor] = p_intermedia[0]

                if(v_hijo_2 < segundo_peor):
                    poblacion[i_segundo_peor]= p_intermedia[1]

            elif(v_hijo_1 < segundo_peor):

                poblacion[i_segundo_peor] = p_intermedia[0]

    return poblacion, mejor_fit, i_mejor


def AGE_BLX(train_set, test_set, train_set_size, test_set_size,weights,atributes):

    n_atr = len(atributes)
    poblacion = [my_generator.uniform(low=0.0,high=1.0,size=n_atr) for i in range(tam_poblacion)]

    n_esperado_cruces =  int(p_cruce_generacional*tam_poblacion)
    n_esperado_mutaciones = int(p_mutacion*2*n_atr)
    ordenacion = seleccion()  
    p_intermedia = [np.zeros(n_atr) for i in range(2)]
 
    #print(ordenacion)

    mejor_fit, indice_mejor, indice_peor = valor_poblacion_age(poblacion,train_set, test_set, train_set_size, test_set_size,atributes)

    n_evaluaciones = 0
    while(n_evaluaciones<5):
        print("Partimos de la población:",poblacion)
        #--------seleccion--------
        padre_1, padre_2 = seleccion_AGE(poblacion,train_set, test_set, train_set_size, test_set_size,atributes)
        print("Seleccionamos padres: ",padre_1,padre_2)
        #----------cruce---------- 
        #se cruzan solo los dos padres que tenemos 
        c_1, c_2 = BLX_alpha(padre_1,padre_2,n_atr,alpha) 
        print("cruce de ",padre_1," ",padre_2,""," genera: ",c_1,c_2)
        p_intermedia[0] = c_1
        p_intermedia[1] = c_2

        print("Poblacion intermedia: ",p_intermedia)

        #---------mutación--------
        #no sé cómo funciona 
        for i in range(n_esperado_mutaciones):
                cromosoma = rd.randint(0,1)
                #print(cromosoma)
                gen = rd.randint(0,n_atr-1)
                #print(gen)
                print("mutacion gen: ",gen," cromosoma: ",cromosoma)
                p_intermedia[cromosoma][gen] = mutacion_normal(p_intermedia,cromosoma,n_atr)
                print("resultado mutacion: ",p_intermedia[cromosoma])

        #-----reemplazamiento-----
        #los dos hijos compiten para sustituir a el/los peores de la población
        
        mejor_fit, peor, segundo_peor, i_mejor, i_peor, i_segundo_peor = valor_poblacion_age(poblacion,train_set, test_set, train_set_size, test_set_size,atributes)
        n_evaluaciones = n_evaluaciones + 1

        v_hijo_1 = valor_cromosoma(p_intermedia[0],train_set, test_set, train_set_size, test_set_size,atributes)
        v_hijo_2 = valor_cromosoma(p_intermedia[1],train_set, test_set, train_set_size, test_set_size,atributes)
        
        print("Mejor: ",i_mejor,"-",mejor," Peor: ",i_peor,"-",peor, " Segundo peor: ",i_segundo_peor,"-",segundo_peor)
        print("Hijo_1: ",v_hijo_1," Hijo_2: ",v_hijo_2)
        n_evaluaciones = n_evaluaciones + 2

        if(v_hijo_1 < v_hijo_2):
            if(v_hijo_1 < peor):
            
                poblacion[i_peor] = p_intermedia[0]

                if(v_hijo_2 < segundo_peor):
                    poblacion[i_segundo_peor]= p_intermedia[1]

            elif(v_hijo_1 < segundo_peor):

                poblacion[i_segundo_peor] = p_intermedia[0]

    return poblacion, mejor_fit, i_mejor
#selección: torneo binario?
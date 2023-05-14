import utilities
import genetico
import math
import numpy as np
import random as rd
import time

my_generator = np.random.default_rng()

def fcv_5(full_set, n_examples, part_lengths,atributes,i,set_type,seed):
      print("Launching partition ",i)
      train_set, test_set, train_set_size, test_set_size = utilities.select_partition(full_set, n_examples, part_lengths, i,set_type,atributes)
      weights = np.ones(len(atributes))
      clases = utilities.get_clases(test_set, test_set_size)
      #print("Clasificacion correcta: ",clases)
      start = time.time()
      poblacion, mejor_fit, indice_mejor = genetico.AGG_BLX(train_set, test_set, train_set_size, test_set_size,weights,atributes)
      end = time.time()   

      print("Poblacion: ",poblacion)
      print("Mejor valor fitness: ",mejor_fit,"Indice: ",indice_mejor)
      print("Tiempo de ejecucion AGG BLX_alpha: ",end-start)

      start = time.time()
      poblacion, mejor_fit, indice_mejor = genetico.AGG_aritmetico(train_set, test_set, train_set_size, test_set_size,weights,atributes)
      end = time.time()   

      print("Poblacion: ",poblacion)
      print("Mejor valor fitness: ",mejor_fit,"Indice: ",indice_mejor)
      print("Tiempo de ejecucion AGG aritmético: ",end-start)

      start = time.time()
      poblacion, mejor_fit, indice_mejor = genetico.AGE_BLX(train_set, test_set, train_set_size, test_set_size,weights,atributes)
      end = time.time()   

      print("Poblacion: ",poblacion)
      print("Mejor valor fitness: ",mejor_fit,"Indice: ",indice_mejor)
      print("Tiempo de ejecucion AGE BLX_alpha: ",end-start)

      start = time.time()
      poblacion, mejor_fit, indice_mejor = genetico.AGG_aritmetico(train_set, test_set, train_set_size, test_set_size,weights,atributes)
      end = time.time()   

      print("Poblacion: ",poblacion)
      print("Mejor valor fitness: ",mejor_fit,"Indice: ",indice_mejor)
      print("Tiempo de ejecucion AGE aritmético: ",end-start)
      """
      start = time.time()
      clasificacion = utilities.weighted_1NN(train_set, test_set, train_set_size, test_set_size,weights,atributes)      
      end = time.time()
      #print("Clasificacion: ",clasificacion)
      f_f, c_r, r_r = utilities.weighted_fitness_function(train_set,test_set,test_set_size,clasificacion,weights)
      print("f_f: ",f_f," c_r: ",c_r," r_r: ",r_r," T clasificador 1-NN: ",end-start)


      start = time.time()
      relief_weights = Relief.relief(train_set, test_set, train_set_size, test_set_size,atributes,i)
      end = time.time()
      #print(clasificacion)

      print("Particion ",i," : weights ",relief_weights," T relief: ",end-start)

      start = time.time()
      clasificacion = utilities.weighted_1NN(train_set, test_set, train_set_size, test_set_size,relief_weights,atributes)
      end = time.time()
      
      f_f, c_r, r_r = utilities.weighted_fitness_function(train_set,test_set,test_set_size,clasificacion,relief_weights)
      print("f_f: ",f_f," c_r: ",c_r," r_r: ",r_r," T clasificación relief: ",end-start)

      print("Partimos de la semilla: ",seed)
      start = time.time()
      BL_weights, f_f, c_r, r_r = BL.BL(train_set, test_set, train_set_size, test_set_size,atributes,i,seed) 
      end = time.time()

      print("Particion ",i," : weights ",BL_weights," T BL: ",end-start)

      print("f_f: ",f_f," c_r: ",c_r," r_r: ",r_r)

    
      print("Particion ",i," : c_r ",c_r," r_r ",r_r," f_f ",f_f) 
     """ 

def main():
    while(1):
        data_set = input("Introduzca conjunto de datos o FIN para terminar la ejecución:")
        if(data_set == "ozone"):
            atributes,full_set,n_examples,part_lengths = utilities.processData(utilities.ozone,utilities.o_type)
        elif(data_set == "diabetes"):
            atributes,full_set,n_examples,part_lengths = utilities.processData(utilities.diabetes,utilities.d_type)
        elif(data_set == "spectf_heart"):
            atributes,full_set,n_examples,part_lengths = utilities.processData(utilities.spectf_heart,utilities.s_type)
        elif(data_set == "FIN"):
            break
        else:
            print("Conjunto no válido.")

        seed = my_generator.uniform(low=0.0,high=1.0,size=len(atributes))

        for i in range(5):
            fcv_5(full_set,n_examples,part_lengths,atributes,i,data_set,seed)

main()
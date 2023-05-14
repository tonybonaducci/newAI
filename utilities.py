from scipy.io import arff
import pandas as pd
import random as rd
import numpy as np
import math
from sklearn import preprocessing

t=1
n_files=5
n_caracteristicas=0
disc_threshold=0.1
alpha=0.8
MAX = 100
n_sets_ficheros = 3

diabetes = ["Instancias_APC/diabetes_1.arff","Instancias_APC/diabetes_2.arff","Instancias_APC/diabetes_3.arff","Instancias_APC/diabetes_4.arff","Instancias_APC/diabetes_5.arff"]
ozone = ["Instancias_APC/ozone-320_1.arff","Instancias_APC/ozone-320_2.arff","Instancias_APC/ozone-320_3.arff","Instancias_APC/ozone-320_4.arff","Instancias_APC/ozone-320_5.arff"]
spectf_heart = ["Instancias_APC/spectf-heart_1.arff","Instancias_APC/spectf-heart_2.arff","Instancias_APC/spectf-heart_3.arff","Instancias_APC/spectf-heart_4.arff","Instancias_APC/spectf-heart_5.arff"]
fichero = "Instancias_APC/diabetes_1.arff"
d_type = [('preg', '<f8'), ('plas', '<f8'), ('pres', '<f8'), ('skin', '<f8'), ('insu', '<f8'), ('mass', '<f8'), ('pedi', '<f8'), ('age', '<f8'), ('class', 'S15')]
o_type = [('V1', '<f8'), ('V2', '<f8'), ('V3', '<f8'), ('V4', '<f8'), ('V5', '<f8'), ('V6', '<f8'), ('V7', '<f8'), ('V8', '<f8'), ('V9', '<f8'), ('V10', '<f8'), ('V11', '<f8'), ('V12', '<f8'), ('V13', '<f8'), ('V14', '<f8'), ('V15', '<f8'), ('V16', '<f8'), ('V17', '<f8'), ('V18', '<f8'), ('V19', '<f8'), ('V20', '<f8'), ('V21', '<f8'), ('V22', '<f8'), ('V23', '<f8'), ('V24', '<f8'), ('V25', '<f8'), ('V26', '<f8'), ('V27', '<f8'), ('V28', '<f8'), ('V29', '<f8'), ('V30', '<f8'), ('V31', '<f8'), ('V32', '<f8'), ('V33', '<f8'), ('V34', '<f8'), ('V35', '<f8'), ('V36', '<f8'), ('V37', '<f8'), ('V38', '<f8'), ('V39', '<f8'), ('V40', '<f8'), ('V41', '<f8'), ('V42', '<f8'), ('V43', '<f8'), ('V44', '<f8'), ('V45', '<f8'), ('V46', '<f8'), ('V47', '<f8'), ('V48', '<f8'), ('V49', '<f8'), ('V50', '<f8'), ('V51', '<f8'), ('V52', '<f8'), ('V53', '<f8'), ('V54', '<f8'), ('V55', '<f8'), ('V56', '<f8'), ('V57', '<f8'), ('V58', '<f8'), ('V59', '<f8'), ('V60', '<f8'), ('V61', '<f8'), ('V62', '<f8'), ('V63', '<f8'), ('V64', '<f8'), ('V65', '<f8'), ('V66', '<f8'), ('V67', '<f8'), ('V68', '<f8'), ('V69', '<f8'), ('V70', '<f8'), ('V71', '<f8'), ('V72', '<f8'), ('class', 'S3')]
s_type = [('V1', '<f8'), ('V2', '<f8'), ('V3', '<f8'), ('V4', '<f8'), ('V5', '<f8'), ('V6', '<f8'), ('V7', '<f8'), ('V8', '<f8'), ('V9', '<f8'), ('V10', '<f8'), ('V11', '<f8'), ('V12', '<f8'), ('V13', '<f8'), ('V14', '<f8'), ('V15', '<f8'), ('V16', '<f8'), ('V17', '<f8'), ('V18', '<f8'), ('V19', '<f8'), ('V20', '<f8'), ('V21', '<f8'), ('V22', '<f8'), ('V23', '<f8'), ('V24', '<f8'), ('V25', '<f8'), ('V26', '<f8'), ('V27', '<f8'), ('V28', '<f8'), ('V29', '<f8'), ('V30', '<f8'), ('V31', '<f8'), ('V32', '<f8'), ('V33', '<f8'), ('V34', '<f8'), ('V35', '<f8'), ('V36', '<f8'), ('V37', '<f8'), ('V38', '<f8'), ('V39', '<f8'), ('V40', '<f8'), ('V41', '<f8'), ('V42', '<f8'), ('V43', '<f8'), ('V44', '<f8'), ('class', 'S3')]

#print(files[0])
#attributes = list()
#print(files[8])

def processData(ficheros,s):
    #print("hola")
    dataset = []
    for i in range(len(ficheros)): 
        data , meta = arff.loadarff(ficheros[i])
        dat = arff.loadarff(ficheros[i])
        dataset.append(data)


    dlist = data.tolist()
    attributes = list()
    norm = list()

    for i in range(len(meta.names())-1):
        attributes.append(meta.names()[i])

    #print(dataset)
    #print(attributes)

    n_atributes = len(attributes)-1

    part_lengths = []

    n_examples = 0

    for set in dataset:
        part_lengths.append(len(set))
        n_examples+=len(set)
        #print("Actualizamos a ",n_examples," ejemplos")

    #print("part_lengths: ",part_lengths)
    final_set = np.ndarray(shape = (n_examples),dtype= s)

    for i in range(n_examples):
        if (i<len(dataset[0])):
            final_set[i] = dataset[0][i]

        elif (i< len(dataset[0]) + len(dataset[1])):
            final_set[i] = dataset[1][i - len(dataset[0])]

        elif (i< len(dataset[0]) + len(dataset[1]) + len(dataset[2])):
            final_set[i] = dataset[2][i - (len(dataset[0]) + len(dataset[1])) ]

        elif (i< len(dataset[0]) + len(dataset[1]) + len(dataset[2]) + len(dataset[3])):
            final_set[i] = dataset[3][i - (len(dataset[0]) + len(dataset[1]) + len(dataset[2]))]

        else:
            final_set[i] = dataset[4][i - (len(dataset[0]) + len(dataset[1]) + len(dataset[2]) + len(dataset[3]) )]

    #print(type(final_set))


    for atribute in attributes:
        final_set[atribute]= (preprocessing.minmax_scale(final_set[atribute]))
    #print("ATRIBUTOS QUE VAMOS A DEVOLVER: ",attributes)
    return [atribute for atribute in attributes],final_set, n_examples, part_lengths
#print(final_set)

##############attributes,final_set,n_examples,part_lengths = processData(ozone)
#print("ATRIBUTOS DEVUELTOS: ",attributes)
"""
def catch(set, length,atributes):

  for atribute in atributes:
    for i in range(length):
      if (set[atribute][i] > 1):
        #print("El atributo ",atribute," del ejemplo ",i," del conjunto no está correctamente normalizado.")
        break
"""

def select_partition(full_set, n_examples, part_lengths, i,set_type,atributes):

    if(set_type == "diabetes"):
        s = d_type
    elif(set_type == "ozone"):
        s = o_type
    else:
        s = s_type
    
    suma = 0
    #print("Tamaño de cada particion: ",part_lengths)
    last_pos_train = 0
    test_start_pointer = 0
    flag=0

    ### control ###
    #catch(full_set, n_examples, attributes)

    train_set = np.ndarray(shape = (n_examples - part_lengths[i]),dtype=s)
    test_set = np.ndarray(shape = (part_lengths[i]),dtype=s)


    test_set_size = part_lengths[i]
    train_set_size = n_examples - test_set_size
    for j in range(len(part_lengths)):

        if (j < i):
            for k in range(part_lengths[j]):
                train_set[k + last_pos_train] = full_set[k + last_pos_train]

            test_start_pointer += part_lengths[j]
            last_pos_train += part_lengths[j]

        if (i == j):
            for k in range(part_lengths[i]):
                test_set[k] = full_set[test_start_pointer + k]
        
        if (j > i):
            #if(flag == 1):
                #print("OJO")
            
            for k in range(part_lengths[j]):
                #print("Ejemplo train num ", k + suma)
                for atribute in atributes: 
                    train_set[atribute][last_pos_train + k] = full_set[atribute][ test_start_pointer + test_set_size -1 + k]
                    #print("Añadimos el valor ",train_set[atribute][last_pos_train + k]," en la posicion ",last_pos_train + k," del train set")

                train_set['class'][last_pos_train + k] = full_set['class'][ test_start_pointer + test_set_size -1 + k]
                #print(train_set['class'][last_pos_train + k])

            last_pos_train += part_lengths[j]
            #flag=1
            suma+=part_lengths[j]

    #print("Particion ",i)

    ### control ###
    
    #catch(train_set,train_set_size, attributes)
    #catch(test_set,test_set_size,attributes)
    #print("Train set. Size =  ",train_set_size,train_set)
    #print("Test set. Size =  ",test_set_size,test_set)


    return train_set, test_set, train_set_size, test_set_size

def distancia_euclidea(set_1,set_2,i,j,w,atributes):
    suma_w = 0
    w_index = 0
    n_atributes = len(atributes)
    p=0
    dist = 0
    #print("Entramos distancia euclidea. Vamos a recorrer el num de características")
    for k in range(n_atributes):
      #print(" caracteristica ",k)
      #suma_w += pow(set_1[atribute][i]-set_2[atribute][j],2)
      if((i<len(set_1)) and (j<len(set_2))):
        #print("ENTRAMOS 1")
        #print("Trabajamos con los ejemplos train ",i, set_1[i], " y test",j,set_2[j]," de la particion ",particion)
        #print("En concreto con los atributos de valor ",set_1[i][k]," y ",set_2[j][k])
        #print("Diferencia entre componentes: train ",set_1[i][k]-set_2[j][k])
        p = pow((set_1[i][k]-set_2[j][k]),2)
        #print(p)
        if(w_index < len(w)):
         #print("ENTRAMOS 2")
         suma_w += w[w_index]*p
         w_index+=1
    try:
        dist = math.sqrt(suma_w)
    except ValueError:
        print("Entramos en valueError ante los valores:")
        print("w:",w)
        print("w_index: ", w_index," p: ",p)
        print("Último w index ejecutado:",w_index-1)
        print("Último sumando en suma_w: ",w_index-1)
        print("%d\txxx" % suma_w)
        return 0
    #print("La distancia euclidea entre el train ", i," y el test ", j ," es: ",dist)
    return dist

#devuelve la fila del conjunto de entrenamiento en la que se encuentra el vecino más cercano

def nearest_neighbour(train_set, test_set, train_set_size,i,w,atributes):
  #print("Entramos nearest_neighbour con entrenamiento de ",train_set_size)
  d_min = MAX
  n_neighbour=i
 # print("Vamos a recorrer el conjunto de entrenamiento.")
  for j in range(train_set_size):
    #print("Train example ",j)
    d = distancia_euclidea(train_set,test_set,j,i,w,atributes)
    if(d<d_min):
        d_min=d
        n_neighbour=j

  return n_neighbour, d_min

def class_match(test_set,clasificacion,i):

  test_class = test_set['class'][i]

  if(test_class == clasificacion[i]):
     return 1
  
  return 0

def weighted_1NN(train_set, test_set, train_set_size, test_set_size,w,atributes):
    clasificacion = [('class', 'S15')]*test_set_size
    #print("Entramos 1NN. Vamos a recorrer el cojunto de test")

    for i in range (test_set_size):
      #print("Test example ",i)
      n_neighbour, d_min = nearest_neighbour(train_set, test_set, train_set_size,i,w,atributes)
      clasificacion[i] = train_set['class'][n_neighbour]
      #print("El vecino más cercano de ",i," es ",n_neighbour," con una distancia euclídea de ",d_min)
    #print("Acabamos 1NN")
    return clasificacion

def class_rate(test_set,test_set_size,clasificacion):
    instancias_bien_clasificadas = 0

    for i in range(test_set_size):
      if (class_match(test_set,clasificacion,i)):
        instancias_bien_clasificadas+=1
    
    return 100*(instancias_bien_clasificadas / len(test_set))

def reduction_rate(w):
    b = 0
    b = below(disc_threshold,w)
    r_r = 100*(b/len(w))
    #print("Nuestro vector es ",w," y el umbral es ",disc_threshold," -> ",below," quedan por debajo. r_r = ",r_r)

    return r_r
"""
def fitness_function(train_set,test_set,test_set_size,clasificacion):
    c_r = class_rate(test_set,test_set_size,clasificacion)
    r_r=0
    #if(r_r > 0):
      #print("TASA REDUCCION")
    f_f = alpha*c_r + (1-alpha)*r_r
    #print("fitness function va a devolver: f_f ",f_f," c_r ",c_r," r_r ",r_r)

    return f_f, c_r, r_r
"""
def weighted_fitness_function(train_set,test_set,test_set_size,clasificacion,w):
    c_r = class_rate(test_set,test_set_size,clasificacion)
    r_r = reduction_rate(w)

      #print("TASA REDUCCION")
    f_f = alpha*c_r + (1-alpha)*r_r
    #print("fitness function va a devolver: f_f ",f_f," c_r ",c_r," r_r ",r_r)

    return f_f, c_r, r_r
#train_set, test_set, train_set_size, test_set_size = select_partition(final_set, n_examples, part_lengths,0)

def below(threshold,vector):
    count=0
    for i in range(len(vector)):
        if(vector[i] < threshold):
            count+=1
    return count

def normalizar(num):

    if num<0:
        return 0
    elif num>1:
        return 1
    else:
        return num

def get_clases(test_set, test_set_size):
    clases = [('class', 'S15')]*test_set_size
    for i in range(test_set_size):
        clases[i] = test_set['class']
    return clases

def get_seed(n_atributes):
    seed=[]
    for i in range(0,n_atributes):
        print("Introduzca coordenada ",i," de la semilla:")
        l=int(input())
        seed.append(l)  

    return seed   

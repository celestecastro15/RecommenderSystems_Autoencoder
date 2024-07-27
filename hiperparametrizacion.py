######## HIPERPARAMETRIZACION DEL MODELO DE RECOMMENDER SYSTEM ########
#Última modificación: 26/06/2023
#Autor: Celeste Castro Granados celsgazu@ciencias.unam.mx

'''Este código permite realizar un proceso de hiperparametrización considerando distintos rangos de valores para distintos parámetros. 
Al final se genera un archivo csv con los resultados de todas las pruebas ejecutadas para su posterior consulta'''

from IPython.display import display, clear_output
import pandas as pd
import time
import json
from collections import OrderedDict
from tqdm.auto import tqdm
from collections import OrderedDict, namedtuple
from itertools import product

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from livelossplot import PlotLosses
from tqdm import tqdm
import matplotlib.pyplot as pyplot

######## Clases para realizar la hiperparametrización ########

class RunBuilder():
    @staticmethod 
    def get_runs(params):
        
        Run=namedtuple('Run', params.keys())
        
        runs=[]
        for v in product(*params.values()):
            runs.append(Run(*v))
        
        return runs

class TrainingStatsManager():
    def __init__(self, batches_per_epoch=1): 
        self.run_start_time = None        
        self.epoch_start_time = None
        self.epoch_count = 0
        self.run_count = 0
        self.epoch_data = None
        self.epoch_results = None
        self.run_results = []
        self.progress = None
        self.batches_per_epoch = batches_per_epoch
        
    def begin_run(self, run):        
        self.run_start_time = time.time()
        self.run_count += 1
        self.run=run

    def begin_epoch(self):
        self.epoch_start_time = time.time()        
        self.epoch_count += 1
        self.epoch_data = OrderedDict()
        self.epoch_results = OrderedDict()
        
        self.clear_displayed_results()
        self.display_progress()
        self.display_run_results()
           
    def track(self, key, value):
        if key not in self.epoch_data:            
            self.epoch_data[key] = [value]
        else:
            self.epoch_data[key].append(value)
            
    def add_result(self, key, value):
        self.epoch_results[key] = value
            
    def end_epoch(self):        
        results = OrderedDict()
        results['run']=self.run_count
        results['epoch'] = self.epoch_count
        results['epoch duration'] = time.time() - self.epoch_start_time
        results['run duration'] = time.time() - self.run_start_time
        results['batch_train']=self.run.batch_train
        results['batch_test']=self.run.batch_test
        results['hly1']=self.run.hly1 
        results['hly2']=self.run.hly2
        results['hly3']=self.run.hly3
        results['activation']=self.run.activation
        results['constrained']=self.run.constrained
        results['dropout']=self.run.dropout
        results['last_act']=self.run.last_act
        results['lr']=self.run.lr
        results['epochs']=self.run.epochs
        
        
        for key in self.epoch_data.keys():
            self.epoch_results[f'Promedios_{key}']=np.mean(self.epoch_data[key])
        
        for k, v in self.epoch_results.items(): results[k] = v
        
        self.run_results.append(results)
        self.progress.close()
        
    def end_run(self):
        self.clear_displayed_results()
        self.display_run_results()
        self.epoch_count=0
        
        
    def display_progress(self):
        self.progress = tqdm(
            total=self.batches_per_epoch
            , desc=f'Epoch ({self.epoch_count}) Progress'
        )
        
    def display_run_results(self):
        if len(self.run_results) > 0:
            display(pd.DataFrame.from_dict(self.run_results, orient='columns'))
            
    def clear_displayed_results(self):
        clear_output(wait=True)
    
    def save(self, fileName):        
        pd.DataFrame.from_dict(
            self.run_results
            ,orient='columns'
        ).to_csv(f'{fileName}.csv')
        
def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

######## Clases para construir el Recommender ########
        
# Clase TestDataset
class TestDataset(Dataset):
    def __init__(self, test_file, transform=None):
        # Cargar los datos del archivo CSV y elimina la primera columna (Compuesto)
        #Al final los datos a completar se encuentran en las columnas de las propiedades y no en la que contiene la información de los compuestos
        self.data = pd.read_csv(test_file)
        self.data = self.data.iloc[:,1:]
        self.transform = transform
        
        # Aplicar la función de transformación si se proporciona
        if transform is not None:
            self.data = self.transform(np.array(self.data))
        
    def __len__(self):
        # Devolver la longitud del conjunto de datos de prueba
        return len(self.data[0])
    
    def __getitem__(self, ind):
        # Devolver el vector de compuesto correspondiente al índice especificado
        user_vector = self.data.data[0][ind]
        return user_vector
    
# Clase TrainDataset
class TrainDataset(Dataset):
    def __init__(self, train_file, transform=None):
        # Cargar los datos del archivo CSV y elimina la primera columna (Compuesto)
        #Al final los datos a completar se encuentran en las columnas de las propiedades y no en la que contiene la información de los compuestos
        self.data = pd.read_csv(train_file)
        self.data = self.data.iloc[:,1:]
        self.transform = transform
        
        # Aplicar la función de transformación si se proporciona
        if transform is not None:
            self.data = self.transform(np.array(self.data))
        
    def __len__(self):
        # Devolver la longitud del conjunto de datos de entrenamiento
        return len(self.data[0])
    
    def __getitem__(self, ind):
        # Devolver el vector de compuesto correspondiente al índice especificado
        user_vector = self.data.data[0][ind]
        return user_vector
    
#Función de pérdida
class MSEloss_with_Mask(nn.Module):
  #Esta clase hereda de la clase nn.Module de PyTorch
  def __init__(self):
    super(MSEloss_with_Mask,self).__init__()

  def forward(self,inputs, targets):
    #inputs, targets - entradas y salidas del modelo respectivamente 
    # Enmascaramiento en un vector de 1's y 0's.
    mask = (targets!=0)
    mask = mask.float()

    # Número real de datos en la matriz.
    # Se toma el máximo para evitar la división por cero en el cálculo de la pérdida.
    other = torch.Tensor([1.0])
    other = other.cuda(num_gpu)
    number_ratings = torch.max(torch.sum(mask),other)

    # Cálculo del error cuadrático medio (MSE) y la pérdida
    error = torch.sum(torch.mul(mask,torch.mul((targets-inputs),(targets-inputs))))
    loss = error.div(number_ratings)

    return loss[0]

#Función de activación
def activation(input, type):
  
    if type.lower()=='selu':
        return F.selu(input)
    elif type.lower()=='elu':
        return F.elu(input)
    elif type.lower()=='relu':
        return F.relu(input)
    elif type.lower()=='relu6':
        return F.relu6(input)
    elif type.lower()=='lrelu':
        return F.leaky_relu(input)
    elif type.lower()=='tanh':
        return F.tanh(input)
    elif type.lower()=='sigmoid':
        return F.sigmoid(input)
    elif type.lower()=='swish':
        return F.sigmoid(input)*input
    elif type.lower()=='identity':
        return input
    else:
        raise ValueError("Unknown non-Linearity Type")
    
#Clase Autoencoder
class AutoEncoder(nn.Module):
    # Constructor de la clase, recibe como parámetros layer_sizes, nl_type, is_constrained, dp_drop_prob y last_layer_activations
    def __init__(self, layer_sizes, nl_type='selu', is_constrained=True, dp_drop_prob=0.0, last_layer_activations=True):
        """
        layer_sizes = tamaño de cada capa en el modelo de encoder. Should start with feature size (e.g. dimensionality of x)
        Por ejemplo: [10000, 1024, 512] resultará en:
            - 2 capas de encoder: 10000x1024 y 1024x512. La capa de representación (z) será de 512.
            - 2 capas de decoder: 512x1024 y 1024x10000.
    
        nl_type = tipo de no linealidad (por defecto: 'selu').
        is_constrained = Si es verdadero, entonces los pesos del encoder y decoder están atados.
        dp_drop_prob = probabilidad de Dropout.
        last_layer_activations = Si es verdadero, se aplica una activación en la última capa del decoder.
        """

        # Llama al constructor de la clase padre (nn.Module)
        super(AutoEncoder, self).__init__()

        # Asignación de parámetros a variables de la instancia
        self.layer_sizes = layer_sizes
        self.nl_type = nl_type
        self.is_constrained = is_constrained
        self.dp_drop_prob = dp_drop_prob
        self.last_layer_activations = last_layer_activations

        # Si dp_drop_prob > 0, se inicializa una instancia de Dropout
        if dp_drop_prob>0:
            self.drop = nn.Dropout(dp_drop_prob)

        self._last = len(layer_sizes) - 2

        # Inicialización de pesos del encoder
        self.encoder_weights = nn.ParameterList( [nn.Parameter(torch.rand(layer_sizes[i+1], layer_sizes[i])) for i in range(len(layer_sizes) - 1)  ] )

        # "Inicialización Xavier" (Entendiendo la dificultad en entrenar redes neuronales profundas de alimentación directa - por Glorot, X. & Bengio, Y.)
        # (Los valores se muestrean a partir de una distribución uniforme)
        for weights in self.encoder_weights:
            init.xavier_uniform_(weights)

        # Bias del encoder
        self.encoder_bias = nn.ParameterList( [nn.Parameter(torch.zeros(layer_sizes[i+1])) for i in range(len(layer_sizes) - 1) ] )

        # Lista de layer_sizes invertida
        reverse_layer_sizes = list(reversed(layer_sizes)) 
        # reversed retorna un iterador

        # Inicialización de pesos del decoder si is_constrained es falso
        if is_constrained == False:
            self.decoder_weights = nn.ParameterList( [nn.Parameter(torch.rand(reverse_layer_sizes[i+1], reverse_layer_sizes[i])) for i in range(len(reverse_layer_sizes) - 1) ] )

            # Inicialización Xavier de los pesos del decoder
            for weights in self.decoder_weights:
                init.xavier_uniform_(weights)

        # Bias del decoder
        self.decoder_bias = nn.ParameterList( [nn.Parameter(torch.zeros(reverse_layer_sizes[i+1])) for i in range(len(reverse_layer_sizes) - 1) ] ) 
    
    
    def encode(self, x):
     #Realiza la codificación de la entrada x'
     # Recorremos la lista de pesos del codificador
        for i, w in enumerate(self.encoder_weights):
            # Aplicamos la operación de multiplicación matricial entre la entrada x y el peso w del codificador, y sumamos el sesgo correspondiente
            x = F.linear(input=x, weight=w, bias=self.encoder_bias[i])
            # Aplicamos la función de activación correspondiente al tipo de no linealidad definida
            x = activation(input=x, type=self.nl_type)

        # Aplicamos Dropout en la última capa, si se define una probabilidad de eliminación de nodos
        if self.dp_drop_prob > 0:
            x = self.drop(x)

    # Retornamos el tensor resultante
        return x
    
    def build_latent_rep(self,x):
        #Obtiene la representación latente de nuevos datos una vez que el encoder ya está entrenado
        self.eval()
        x=self.encode(x)
        #será necesario agregar el paso de x = x.detach().numpy()?
        return x 


    def decode(self, x):
        # Si se trata de un modelo con pesos atados, se aplica la operación de decodificación con los pesos del codificador invertidos
        if self.is_constrained == True:
            # Los pesos están atados, por lo que recorremos los pesos del codificador en orden inverso
            for i, w in zip(range(len(self.encoder_weights)), list(reversed(self.encoder_weights))):
                # Aplicamos la operación de multiplicación matricial entre la entrada x y el peso w del decodificador, y sumamos el sesgo correspondiente
                x = F.linear(input=x, weight=w.t(), bias=self.decoder_bias[i])
                # Aplicamos la función de activación correspondiente al tipo de no linealidad definida, excepto para la última capa si se define que no tenga activación
                x = activation(input=x, type=self.nl_type if i != self._last or self.last_layer_activations else 'identity')

        else:
            # Los pesos no están atados, por lo que recorremos los pesos del decodificador
            for i, w in enumerate(self.decoder_weights):
                # Aplicamos la operación de multiplicación matricial entre la entrada x y el peso w del decodificador, y sumamos el sesgo correspondiente
                x = F.linear(input=x, weight=w, bias=self.decoder_bias[i])
                # Aplicamos la función de activación correspondiente al tipo de no linealidad definida, excepto para la última capa si se define que no tenga activación
                x = activation(input=x, type=self.nl_type if i != self._last or self.last_layer_activations else 'identity')

    # Retornamos el tensor resultante
        return x


    def forward(self, x):
        return self.decode(self.encode(x))
    
#Función del proceso de entrenamiento
def train(model, criterion, optimizer, train_dl, test_dl, control, num_epochs=40):
  # Definimos la función "train" que toma como entrada un modelo, un criterio de pérdida, un optimizador, 
  # un conjunto de entrenamiento y un conjunto de pruebas, y un número de épocas.

  # Inicializamos la biblioteca livelossplot para graficar en tiempo real la pérdida de entrenamiento y validación
  liveloss = PlotLosses()
  
  #Listas de pérdidas de entrenamiento y validación
  lr_tr_loss = []
  lr_val_loss= []
  
  # Iteramos sobre cada época
  for epoch in range(num_epochs):
    control.begin_epoch()
    #Matrices completas
    matrix_out_train = torch.Tensor([])
    matrix_out_test = torch.Tensor([])
    
    # Creamos dos listas vacías para guardar la pérdida de entrenamiento y la de validación en cada época
    train_loss, valid_loss = [], []
    
    # Creamos un diccionario vacío para guardar los registros de la pérdida en la biblioteca livelossplot
    logs = {}
    
    # Definimos un prefijo para los registros (en este caso, vacío)
    prefix = ''

    # Entrenamiento
    # Ponemos el modelo en modo de entrenamiento
    model.train()
    
    best_loss=10000000000000
    
    # Iteramos sobre los datos en el conjunto de entrenamiento
    for i, data in enumerate(train_dl, 0):
      # Obtenemos los datos de entrada y las etiquetas
      inputs = labels = data
      
      # Enviamos los datos a la GPU si está disponible
      inputs = inputs.cuda(num_gpu)
      labels = labels.cuda(num_gpu)

      # Convertimos los datos en tensores flotantes
      inputs = inputs.float() 
      labels = labels.float()

      # Ponemos los gradientes en cero
      optimizer.zero_grad()

      # Se aplica la función forward a inputs 
      outputs = model(inputs)
      
      # Enviamos la salida del modelo a la GPU si está disponible
      outputs = outputs.cuda(num_gpu)
      
      # Calculamos la pérdida
      loss = criterion(outputs,labels)
      
      # Realizamos el retroceso
      loss.backward()
      
      # Actualizamos los pesos
      optimizer.step()
      control.track('Train loss', loss.item()) #train loss before refeeding

      # -> Re-alimentación densa iterativa de salida <- #
      
      # Ponemos los gradientes en cero
      optimizer.zero_grad()
      
      # Es importante "detach()" la salida para evitar la construcción innecesaria del grafo computacional
      outputs = model(outputs.detach()) 
      #revisar si la salida del modelo está en cpu
      
      # Enviamos la salida a la GPU si está disponible
      outputs = outputs.cuda(num_gpu)
      
      # Calculamos la pérdida
      loss = criterion(outputs, labels)
      
      # loss en este punto contiene un grafo computacional, el cual contiene los pasos llevados a cabo para llegar a este calculo
      # Realizamos el retroceso
      loss.backward()
      
      
      #ya no existe el grafo
      
      # Actualizamos los pesos
      optimizer.step()
      control.track('train loss refeeding', loss.item())

      # Guardamos la pérdida de entrenamiento en la lista "train_loss"
      train_loss.append(loss.item())
      
      # Guardamos la pérdida en el diccionario de registros
      logs[prefix + 'MMSE loss'] = loss.item()
      
      outputs = outputs.to('cpu')
      #Guardamos el batch reconstruido de entrenamiento
      matrix_out_train = torch.cat([matrix_out_train, outputs], 0)
      
      # Iterar a través de los datos de test_dl utilizando el índice i comenzando desde 0.
    for i, data in enumerate(test_dl, 0):
      # Establecer el modelo en modo de evaluación.
      model.eval()
      # Asignar los datos a las variables inputs y labels.
      inputs = labels = data
      # Mover las variables inputs y labels a la GPU si está disponible.
      inputs = inputs.cuda(num_gpu)
      labels = labels.cuda(num_gpu)

      # Convertir las variables inputs y labels a float.
      inputs = inputs.float()
      labels = labels.float()

      # Calcular las predicciones utilizando el modelo con las variables inputs.
      outputs = model(inputs)
      # Mover las predicciones a la GPU si está disponible.
      outputs = outputs.cuda(num_gpu)
      # Calcular la pérdida utilizando las predicciones y las etiquetas.
      loss = criterion(outputs, labels)
      control.track('Valid loss', loss.item())

      # Agregar la pérdida a la lista de pérdidas de validación.
      valid_loss.append(loss.item())
      # Establecer el prefijo como 'val_'
      prefix = 'val_'
      # Agregar la pérdida a los registros con el prefijo.
      logs[prefix + 'MMSE loss'] = loss.item()
      
      #Guardamos el batch reconstruido de entrenamiento
      outputs = outputs.to('cpu')
      matrix_out_test = torch.cat([matrix_out_test, outputs], 0)

    
    
    # Calcular la media de las pérdidas de entrenamiento y agregarla a la lista de pérdidas de entrenamiento.
    lr_tr_loss.append(np.mean(train_loss))
    # Calcular la media de las pérdidas de validación y agregarla a la lista de pérdidas de validación.
    lr_val_loss.append(np.mean(valid_loss))
    #checkpoint
    if valid_loss[-1] < best_loss:
          checkpoint(model, f'model_checkpoint_220823_50/best_model.pt')
          best_loss=valid_loss[-1]
    # Actualizar los registros utilizando la librería liveloss.
    liveloss.update(logs)
    # Dibujar el gráfico utilizando la librería liveloss.
    liveloss.draw()

    # Imprimir el número de la época actual y la pérdida de entrenamiento y validación promedio.
    print("Epoch:", epoch+1, " Training Loss: ", np.mean(train_loss), " Valid Loss: ", np.mean(valid_loss))
    
    control.add_result('Train Loss', np.mean(train_loss)) #refeeding 
    control.add_result('Valid Loss', np.mean(valid_loss))
    control.end_epoch()
    
    # Si la época actual es la última, devolver las predicciones.
    if epoch == num_epochs -1:
      return [matrix_out_train, matrix_out_test]
  
        
        
######## Main Process ########
if __name__=="__main__":
    
    #Número de GPU
    num_gpu = 2
    
    #Rutas de los archivos
    ruta_train = '/home/celeste/tesis/recommender/Recommender-Adsorbentes/train_50porciento_random.csv'
    ruta_test = '/home/celeste/tesis/recommender/Recommender-Adsorbentes/test_50porciento_random.csv'
    
    #Matrices de datos originales
    train_original = pd.read_csv(ruta_train)
    test_original = pd.read_csv(ruta_test)
    
    print('### Importación de datos completada ###')
    
    #Valores de los parámetros a probar
    params=OrderedDict(
        batch_train=[128],
        batch_test=[8],
        hly1 = [75],
        hly2 = [75,64],
        hly3 = [85,64],
        activation = ['relu'],
        constrained = [True,False],
        dropout = [0.0,0.2],
        last_act = [True,False],
        lr = [0.0005, 0.001],
        epochs=[800]
        )
    
    
    control = TrainingStatsManager()
    for run in RunBuilder.get_runs(params):

        control.begin_run(run)
        #Preparación de los datos
        transformations = transforms.Compose([transforms.ToTensor()])
        train_dat = TrainDataset(ruta_train, transformations)
        test_dat = TestDataset(ruta_test, transformations)

        train_dl = DataLoader(dataset=train_dat, batch_size = run.batch_train, shuffle=False, num_workers = 0)
        test_dl = DataLoader(dataset=test_dat, batch_size = run.batch_test, shuffle=False, num_workers=0)
    
        print('### Preparación de datos completada ###')
    
        #Modelo
        layer_sizes = [90, run.hly1, run.hly2, run.hly3]
        model= AutoEncoder(layer_sizes=layer_sizes, nl_type = run.activation, is_constrained=run.constrained, dp_drop_prob=run.dropout, last_layer_activations=run.last_act)
        model = model.cuda(num_gpu)
    
        #Loss Function
        criterion = MSEloss_with_Mask()
        criterion = criterion.cuda(num_gpu)
    
        #Optimizer
        optimizer = optim.SGD(model.parameters(), lr=run.lr)
    
        print(" ### Inicia el entrenamiento ### ")
        
        #Fit the model
        out = train(model, criterion, optimizer, train_dl, test_dl, control, run.epochs)

        control.end_run()
        control.save('parametros_50porciento_220823')
        print('### Proceso completado ###')
        
        
        #Cargar el modelo pt
        #checkpoint = torch.load(ruta_best_model.pt)
        #model.load_state_dict(checkpoint['model_state_dict'])
        
        #para seguir entrenando
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        

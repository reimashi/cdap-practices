'''
Programa para Apache Spark para contar numero de pasajeros en destino por agencia

Uso:
    spark-submit ./program.py {Nombre_Agencia}
    Los archivos de entrada file_dest*.txt y file_num*.txt deben estar en el directorio actual.

Susana Rey Fernández
Aitor González Fernández
'''

# Importaciones de la librería de spark y sistema
from pyspark import SparkContext
import sys

# Configuramos el contexto de ejecucion en spark
sc = SparkContext(appName="PR13")
sc.setLogLevel("WARN")

# Crea una tupla a partir del nombre del destino y el numero de pasajeros
def create_pair(line):
    words = line.split(",")
    return (words[0].strip(' \t\n\r'), int(float(words[1].strip(' \t\n\r'))))

# Comprobamos el parametro de entrada para la agencia seleccionada
if len(sys.argv) < 2:
    print ("Debes introducir el nombre de la agencia por parámetro")
    sys.exit();

agencia = sys.argv[1]
print ("Obteniendo visitantes de destinos ofrecidos por " + agencia)

# Se cargan los destinos de las agencias y los visitantes por cada destino
destinos = sc.textFile("file_dest*.txt")
pasajeros = sc.textFile("file_num*.txt")

# Filtramos por la agencia pasada por parametro
destinos = destinos.filter(lambda line: agencia in line)

# Nos quedamos solo con el nombre del destino
destinos = destinos.map(lambda line: line.split(",")[0].strip(' \t\n\r'))

# Si la agencia no tiene destinos, salimos
if destinos.count() == 0:
    print ("La agencia " + agencia + " no tiene destinos conocidos.")
    sys.exit()

# Mapeamos las localizaciones y numero de pasajeros
pasajerosPair = pasajeros.map(create_pair)

# Reducimos en funcion de las localizaciones
pasajerosPair = pasajerosPair.reduceByKey(lambda a, b: a + b)

# Filtramos los numeros de pasajeros de las localizaciones que no nos interesan
destinos = destinos.collect();
pasajerosPair = pasajerosPair.filter(lambda tupla: tupla[0] in destinos)

# Imprimimos el resultado por pantalla y archivo
total = 0;
fileout = open("output.txt", "w")
for destino in pasajerosPair.toLocalIterator():
    print("%s: %s" % destino);
    fileout.write("%s: %s\n" % destino)
    total += destino[1]

print("TOTAL: %s" % (total));
fileout.write("TOTAL: %s" % (total))

from PIL import Image
from math import sqrt
import numpy as np
import time

# def get_spaced_colors(n):
# 	max_value = 16581375
# 	interval = int(max_value / n)
# 	colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

# 	return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

def elegirCentroides(datos, n):
	centroides = [datos[0]]
	for i in range(1,n):
		distancias = np.array([min([np.linalg.norm(c - punto) for c in centroides])  for punto in datos[1:]])
		probas = distancias / sum(distancias)
		probasAcumuladas = np.cumsum(probas)
		aleatorio = np.random.rand()
		nuevoCentro = 0
		for pos, dato in enumerate(probasAcumuladas):
			if dato > aleatorio:
				centroides.append(datos[pos])
				break

	return np.array(centroides)

def cargarArchivo(nombre, separador):
	data = []
	resultados = {}
	lines = open(nombre, 'r').readlines()
	np.random.shuffle(lines)
	i = 0
	for line in lines:
		data.append([])
		for word in line.split(separador)[:-1]:
			data[i].append(float(word))
		data[i] = np.array(data[i])
		resultados[str(data[i])] = line[(len(line) - 2)]

		i += 1

	return np.array(data), resultados

def kMeans(datos, n):
	centroides = elegirCentroides(datos, n)
	print('Ya tengo centroides iniciales')

	parada = False
	clusters = [[] for y in range(n)]
	lugarPuntos = [0 for y in range(len(datos))]

	while not parada:
		centricos = [[] for y in range(n)]
		clusters = [[] for y in range(n)]
		contador = 0
		for punto in datos:
			dist = float('inf')
			cluster = 0
			for j in range(n):
				aux = np.linalg.norm(punto - centroides[j])

				if aux < dist:
					cluster = j
					dist = aux

			clusters[cluster].append(punto)
			lugarPuntos[contador] = cluster
			contador += 1

		for i in range(n):
			if len(clusters[i]) != 0:
				centricos[i] = np.mean(clusters[i], axis=0)
			else:
				centricos[i] = 0

		if np.array_equal(centricos, centroides):
			parada = True
		else:
			centroides = centricos

	return clusters, centroides, lugarPuntos

def iris(k):
	datos, clasificacion = cargarArchivo('iris-normal.txt', ',')
	clusters, centroides, lugarPuntos= kMeans(datos, k)

	for elem in clusters:
		c1, c2, c3 = 0, 0, 0
		for ele in elem:
			if clasificacion[str(ele)] == '0':
				c1 += 1
			elif clasificacion[str(ele)] == '1':
				c2 += 1
			else:
				c3 += 1
		
		print("Tipo 0 en cluster: ", c1, "Tipo 1 en cluster: ", c2, "Tipo 2 en cluster: ", c3)
	print('\n')

def imagen(k):
	nombre = "pointillism.jpg"
	im = Image.open(nombre)
	imageSalida = Image.new("RGB", im.size)
	pix = np.array(im.getdata())
	colores = k
	clusters, centroides, lugarPuntos = kMeans(pix, colores)
	for i in range(colores):
		centroides[i] = (int(centroides[i][0]), int(centroides[i][1]), int(centroides[i][2]))

	#colores = get_spaced_colors(colores)
	pixeles = []
	contador = 0
	for elem in pix:
		pixeles.append(centroides[lugarPuntos[contador]])
		contador += 1

	imageSalida.putdata(pixeles)
	imageSalida.save(nombre[:len(nombre) - 4] + str('[') + str(k) + str(']') + '.jpg')

def main():
	for k in range(2,6):
		print("Para K=",k)
		iris(k)
	
	K = [2,4,8,16,32,64,128]

	for k in K:
		print("Ejecutando con k =", k)
		start = time.time()
		imagen(k)
		end = time.time()
		print("Ćon k={0} tardé {1} segundos".format(k, end - start))

if __name__ == '__main__':
	main()
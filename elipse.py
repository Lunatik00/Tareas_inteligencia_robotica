import numpy as np
from matplotlib import pyplot as plt

#conversores de coordenadas cartesianas a polares y polares a cartesianas
#esto es para poder cambiar facilmente entre sistema de coordenadas segun sea mas facil
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


#obtener parametros de la elipse de manera aleatoria en coordenadas polares
Mayor=np.random.rand()
Menor=(Mayor/4)+(np.random.rand()*(Mayor/4))
Angulo=2*np.pi*np.random.rand()
if np.random.randint(2)==1:
    a=Mayor
    b=Menor
else:
    a=Menor
    b=Mayor



#obtener parametros del centro de manera aleatoria en coordenadas polares
distancia_centro=(0.1*Menor)+np.random.rand()*(0.8*Menor)
Angulo_centro=2*np.pi*np.random.rand()



#obtener puntos para graficar la elipse original
Theta = np.linspace(0, 2*np.pi, 50000)
elipse=(1/(( (np.square(np.cos(Theta))/a**2) + (np.square(np.sin(Theta))/b**2) )**0.5))

x , y= pol2cart(elipse,Theta+Angulo)
x_centro , y_centro= pol2cart(distancia_centro,Angulo_centro)
x += x_centro
y += y_centro


#generar puntos aleatorios en la elipse original
angulos_aleatorios =2*np.pi*np.random.rand(1,5000)
puntos_aleatorios=(1/(( (np.square(np.cos(angulos_aleatorios))/a**2) + (np.square(np.sin(angulos_aleatorios))/b**2) )**0.5))
x_aleatorio , y_aleatorio= pol2cart(puntos_aleatorios,angulos_aleatorios+Angulo)
x_aleatorio += x_centro
y_aleatorio += y_centro

#agregar ruido a los puntos
for i in range(len(x_aleatorio)):
    ruido=(0.05* np.random.rand())
    angulo_ruido=2*np.pi*np.random.rand()
    x_ruido , y_ruido= pol2cart(ruido,angulo_ruido)
    x_aleatorio[i]= x_aleatorio[i]+x_ruido
    y_aleatorio[i]= y_aleatorio[i]+y_ruido

print(x_aleatorio[0])
#para hacer la regresion se utiliza polyfit, esta funcion utiliza minimos cuadrados para ajustar los datos a un polinomio
polinomio=np.poly1d(np.polyfit(np.array(x_aleatorio), np.array(y_aleatorio), 3))


fig, figura = plt.subplots(2,2)
figura[0,0].plot(x,y)
figura[0,0].set_title('Elipse Original: Eje mayor = '+ str(round(Mayor,5))+'; Eje menor = '+ str(round(Menor,5))+';centro = ('+str(round(x_centro,5))+','+ str(round(y_centro,5))+')')
figura[0,1].scatter(x_aleatorio,y_aleatorio)
figura[0,1].set_title('5000 puntos aleatorios de la elipse con perturbacion')
figura[1,0].plot(x_aleatorio,polinomio(x_aleatorio))
figura[1,0].set_title('Elipse minimos cuadrados: Eje mayor = '+ str(round(Mayor,5))+'; Eje menor = '+ str(round(Menor,5))+';centro = ('+str(round(x_centro,5))+','+ str(round(y_centro,5))+')')
#elipse_original = fig.add_subplot(111, projection='polar')
#elipse_original.scatter(Theta+Angulo_centro,elipse)
#plt.polar(Theta_2,elipse)
plt.savefig("polar_coordinates_01.png", bbox_inches='tight')

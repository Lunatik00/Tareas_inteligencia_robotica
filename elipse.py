import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from numpy.linalg import eig, inv, svd
from math import atan2

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
#generar puntos de la elipse
def elipse(center, width, height, phi,datapoints):
    t = np.linspace(0, 2*np.pi, datapoints)
    ellipse_x = center[0] + width*np.cos(t)*np.cos(phi)-height*np.sin(t)*np.sin(phi)
    ellipse_y = center[1] + width*np.cos(t)*np.sin(phi)+height*np.sin(t)*np.cos(phi)
    return [ellipse_x, ellipse_y]

#generar puntos aleatorios con ruido para la elipse
def elipse_random(center, width, height, phi,datapoints):
    t = 2*np.pi*np.random.rand(1,datapoints)
    x_noise, y_noise = 0.025*max(width,height)*(np.random.rand(2, datapoints)-0.5)
    ellipse_x = center[0] + width*np.cos(t)*np.cos(phi)-height*np.sin(t)*np.sin(phi)+x_noise
    ellipse_y = center[1] + width*np.cos(t)*np.sin(phi)+height*np.sin(t)*np.cos(phi)+y_noise
    return [ellipse_x, ellipse_y]

#para generar los parametros a partir de los puntos, se utilizo una combinacion de codigos,
#aparentemente los algoritmos se basan en este paper http://cseweb.ucsd.edu/~mdailey/Face-Coord/ellipse-specific-fitting.pdf
def fit_ellipse(x, y):
    x, y = x[:, np.newaxis], y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S, C = np.dot(D.T, D), np.zeros([6, 6])
    C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1
    U, s, V = svd(np.dot(inv(S), C))
    a_1 = U[:, 0]

    b, c, d, f, g, a = a_1[1] / 2, a_1[2], a_1[3] / 2, a_1[4] / 2, a_1[5], a_1[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    center=np.array([x0, y0])

    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * (
        (c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
    )
    down2 = (b * b - a * c) * (
        (a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
    )
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    axis=np.array([res1, res2])

    rotation_angle=atan2(2 * b, (a - c)) / 2

    return center,axis,rotation_angle

#obtener parametros de la elipse de manera aleatoria en coordenadas polares
Mayor=10*np.random.rand()
Menor=(Mayor/4)+(np.random.rand()*(Mayor/4))
Angulo=2*np.pi*np.random.rand()
a=Mayor
b=Menor
distancia_centro=(0.1*Menor)+np.random.rand()*(0.8*Menor)
Angulo_centro=2*np.pi*np.random.rand()
centro= pol2cart(distancia_centro,Angulo_centro)

data_ellipse=elipse(centro,a,b,Angulo,20000)
data_ellipse_random=elipse_random(centro,a,b,Angulo,5000)
fit_center, fit_axis, fit_angle = fit_ellipse(data_ellipse_random[0][0],data_ellipse_random[1][0])

fig, figura = plt.subplots(2,2)
figura[0,0].plot(data_ellipse[0],data_ellipse[1])
figura[0,0].set_title('Elipse Original: Eje mayor = '+ str(round(Mayor,5))+'; Eje menor = '+ str(round(Menor,5))+';centro = ('+str(round(centro[0],5))+','+ str(round(centro[1],5))+')')
figura[0,1].scatter(data_ellipse_random[0],data_ellipse_random[1])
figura[0,1].set_title('5000 puntos aleatorios de la elipse con perturbacion')
figura[1,0].scatter(data_ellipse[0],data_ellipse[1])
figura[1,0].set_title('Elipse minimos cuadrados: Eje mayor = '+ str(round(Mayor,5))+'; Eje menor = '+ str(round(Menor,5))+';centro = ('+str(round(centro[0],5))+','+ str(round(centro[1],5))+')')

ellipse = Ellipse(
        xy=fit_center, width=2*fit_axis[0], height=2*fit_axis[1], angle=np.rad2deg(fit_angle),
        edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
    )
figura[1,0].add_patch(ellipse)

limite=a+1
fig = plt.figure()

ax1 = plt.subplot(511)
ellipse_original = Ellipse(
        xy=centro, width=2*a, height=2*b, angle=np.rad2deg(Angulo),
        edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
    )
ax1.add_artist(ellipse_original)
ax1.set_title('Elipse Original: Eje mayor = '+ str(round(Mayor,5))+'; Eje menor = '+ str(round(Menor,5))+';centro = ('+str(round(centro[0],5))+','+ str(round(centro[1],5))+')')
ax1.set_xlim([-limite,limite])
ax1.set_ylim([-limite,limite])

ax2 = plt.subplot(513)
ax2.scatter(data_ellipse_random[0],data_ellipse_random[1])
ax2.set_title('5000 puntos aleatorios de la elipse con perturbacion')
ax2.set_xlim([-limite,limite])
ax2.set_ylim([-limite,limite])

ax3 = plt.subplot(515)
ellipse_fit = Ellipse(
        xy=fit_center, width=2*max(fit_axis), height=2*min(fit_axis), angle=np.rad2deg(fit_angle),
        edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
    )
ax3.add_artist(ellipse_fit)
ax3.set_title('Elipse minimos cuadrados: Eje mayor = '+ str(round(Mayor,5))+'; Eje menor = '+ str(round(Menor,5))+';centro = ('+str(round(centro[0],5))+','+ str(round(centro[1],5))+')')
ax3.set_xlim([-limite,limite])
ax3.set_ylim([-limite,limite])

plt.savefig("Elipses", bbox_inches='tight')

import scipy.integrate
#areas
function_ellipse_original = lambda x : (b/(a**2))*(((a**2)-(x**2))**0.5)
function_ellipse_fit = lambda x : (fit_axis[1]/(fit_axis[0]**2))*(((fit_axis[0]**2)-(x**2))**0.5)
area_original, err_area_original = scipy.integrate.quad(function_ellipse_original, -a/2, a/2)
area_fit, err_area_fit = scipy.integrate.quad(function_ellipse_fit, -fit_axis[0]/2, fit_axis[0]/2)
error_abs_area = abs(area_original-area_fit)
error_rel_area = error_abs_area/area_original
error_per_area = error_rel_area*100

#perimetros
function_ellipse_original_perimeter = lambda x : (1+(((-x*b)/((a**2)*((a-(x**2))**0.5)))**2))**0.5
function_ellipse_fit__perimeter = lambda x : (1+(((-x*fit_axis[1])/((fit_axis[0]**2)*((fit_axis[0]-(x**2))**0.5)))**2))**0.5
perimetro_original, err_perimetro_original = scipy.integrate.quad(function_ellipse_original_perimeter, -a/2, a/2)
perimetro_fit, err_perimetro_fit = scipy.integrate.quad(function_ellipse_fit__perimeter, -fit_axis[0]/2, fit_axis[0]/2)
error_abs_perimetro = abs(perimetro_original-perimetro_fit)
error_rel_perimetro = error_abs_perimetro/perimetro_original
error_per_perimetro = error_rel_perimetro*100

print('Area original : '+str(area_original))
print('Area ajustada : '+str(area_fit))
print('Error Absoluto Area : '+str(error_abs_area))
print('Error Relativo Area : '+str(error_rel_area))
print('Error Porcentual Area : '+str(error_per_area)+'%')

print('Perimetro original : '+str(perimetro_original))
print('Perimetro ajustada : '+str(perimetro_fit))
print('Error Absoluto Perimetro : '+str(error_abs_perimetro))
print('Error Relativo Perimetro : '+str(error_rel_perimetro))
print('Error Porcentual Perimetro : '+str(error_per_perimetro)+'%')


datos_original=np.array([centro,a,b,Angulo])
datos_fit = np.array([fit_center,fit_axis[0],fit_axis[1],np.rad2deg(fit_angle)])
datos_area = np.array([area_original,area_fit,error_abs_area,error_rel_area,error_per_area])
datos_perimetro = np.array([perimetro_original,perimetro_fit,error_abs_perimetro,error_rel_perimetro,error_per_perimetro])
#guardar datos en orden:
#datos elipse original, datos elipse ajustada a los puntos, datos areas de elipses, datos perimetros elipses y datos de los puntos aleatorios
archivo=np.array([datos_original,datos_fit,datos_area,datos_perimetro,data_ellipse_random])
np.save('datos',archivo)

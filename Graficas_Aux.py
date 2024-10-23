import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D #axes3d
import plotly.graph_objects as go

from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')

import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
plt.rcParams['font.family'] = 'serif'  # Use a generic serif font
plt.rcParams['font.serif'] = 'DejaVu Serif'

# Leer el archivo CSV y cargarlo en un DataFrame
datosXYZ = pd.read_csv('PosicionXYZ.csv', header=None)


ReferenciaXY = pd.read_csv('ReferenciaXY.csv', header=None)
ReferenciaZ = pd.read_csv('ReferenciaZ.csv', header=None)


InputZ = pd.read_csv('InputZ.csv', header=None)
InputX = pd.read_csv('InputX.csv', header=None)
InputY = pd.read_csv('InputY.csv', header=None)



num_xticks=5
num_yticks=5

Grosor_linea = 3

# Function to convert string representation to float
def str_to_float(string):
    # Remove square brackets and split by spaces
    values = string.replace('[', '').replace(']', '').split()
    # Convert each part to float and return as numpy array
    return np.array([float(value) for value in values])

# Crear una variable de índice basada en la longitud de los datos

Array_without_str = datosXYZ.applymap(str_to_float).values.flatten()
Array_without_str_ref = ReferenciaXY.applymap(str_to_float).values.flatten()

ZValores = [arr[0] for arr in Array_without_str]###Extraer valores
XValores = [arr[1] for arr in Array_without_str] ###Extraer valores
YValores = [arr[2] for arr in Array_without_str] ###Extraer valores

RefX = np.array([arr[0] for arr in Array_without_str_ref])
RefY = np.array([arr[1] for arr in Array_without_str_ref])
RefZ = ReferenciaZ.applymap(str_to_float).values.flatten()

u1 = InputX.values.flatten()
u2 = InputY.values.flatten()
u3 = InputZ.values.flatten()


ErrorZ = RefZ - ZValores
ErrorX = RefX - XValores
ErrorY = RefY - YValores
Error_intZ= 0
Error_intX= 0
Error_intY= 0

Error_absZ= 0
Error_absX= 0
Error_absY= 0
ISE_z = []
ISE_x = []
ISE_y = []

IAE_z = []
IAE_x = []
IAE_y = []

ITSE_z = []
ITSE_x = []
ITSE_y = []

ITAE_z = []
ITAE_x = []
ITAE_y = []

dt = 0.0025


#breakpoint()

x = np.arange(len(datosXYZ.columns))
# # Definir el número de elementos
num_elementos = len(x)

# Definir la frecuencia de muestreo inicial (30 Hz)
fs_inicial = 400  # Hz
# Duración total en segundos
duration_seconds = num_elementos / fs_inicial

time_vector = np.linspace(0, duration_seconds, num_elementos)

# Combinar los vectores de tiempo
tiempo= time_vector#np.concatenate((tiempo_30s, tiempo_restante))

for i in range(len(ErrorZ)):
    Error_intZ += (ErrorZ[i]**2)*dt
    Error_intX += (ErrorX[i]**2)*dt
    Error_intY += (ErrorY[i]**2)*dt
    ISE_z.append(Error_intZ) #= np.sum(ErrorZ**2) * dt
    ISE_x.append(Error_intX)
    ISE_y.append(Error_intY)
    ITSE_z.append(Error_intZ*tiempo[i]) #= np.sum(ErrorZ**2) * dt
    ITSE_x.append(Error_intX*tiempo[i])
    ITSE_y.append(Error_intY*tiempo[i])

# ISE_z = np.sum(ErrorZ**2) * dt
# ISE_x = np.sum(ErrorX**2) * dt
# ISE_y = np.sum(ErrorY**2) * dt

for i in range(len(ErrorZ)):
    Error_absZ += np.abs(ErrorZ[i])*dt
    Error_absX += np.abs(ErrorX[i])*dt
    Error_absY += np.abs(ErrorY[i])*dt
    IAE_z.append(Error_absZ) #= np.sum(ErrorZ**2) * dt
    IAE_x.append(Error_absX)
    IAE_y.append(Error_absY)
    ITAE_z.append(Error_absZ*tiempo[i]) #= np.sum(ErrorZ**2) * dt
    ITAE_x.append(Error_absX*tiempo[i])
    ITAE_y.append(Error_absY*tiempo[i])

print("Los indices ISE de z, x, y son:", ISE_z[len(ErrorZ) - 1], ISE_x[len(ErrorZ) - 1], ISE_y[len(ErrorZ) - 1])
print("Los indices ITSE de z, x, y son:", ITSE_z[len(ErrorZ) - 1], ITSE_x[len(ErrorZ) - 1], ITSE_y[len(ErrorZ) - 1])
#print("Los indice ISE de zm x, y son", ISE_z, ISE_x, ISE_y )
print("Los indices IAE de z, x, y son:", IAE_z[len(ErrorZ) - 1], IAE_x[len(ErrorZ) - 1], IAE_y[len(ErrorZ) - 1])
print("Los indices ITAE de z, x, y son:", ITAE_z[len(ErrorZ) - 1], ITAE_x[len(ErrorZ) - 1], ITAE_y[len(ErrorZ) - 1])

##=================Calculo de la energia==========================
energia1 = np.sum(np.square(np.abs(u1)))
energia2 = np.sum(np.square(np.abs(u2)))
energia3 = np.sum(np.square(np.abs(u3)))
print("El calculo de la energia (x,y,z) es:", energia1, energia2, energia3)

######################################################################################################################################
######################################### GRAFICAS COMBINADAS ########################################################################
fig, (ax5, ax6, ax7) = plt.subplots(3, 1, figsize=(12, 8))



# Grafica X
ax5.plot(tiempo, XValores, linewidth=Grosor_linea, label=r'$x$')
ax5.plot(tiempo, RefX, linewidth=Grosor_linea, label=r'$x_{ref}$', color='red')
ax5.set_ylim(-1.25, 1.25)
#ax5.set_xlabel('Time [s]', fontsize=28)

ax5.grid(True, linestyle='-.')
ax5.legend(loc='lower left',ncol=2, fontsize='28')
ax5.tick_params(axis='both', labelsize=28)

# Grafica Y
ax6.plot(tiempo, YValores, linewidth=Grosor_linea, label=r'$y$')
ax6.plot(tiempo, RefY, linewidth=Grosor_linea, label=r'$y_{ref}$', color='red')
ax6.set_ylabel('Position [m]', fontsize=28)
ax6.set_ylim(-1.95, 1)
#ax6.set_ylabel('Distance [m]', fontsize=28)
ax6.grid(True, linestyle='-.')
ax6.legend(loc='lower left',ncol=2, fontsize='28')
ax6.tick_params(axis='both', labelsize=28)


# Grafica Z
ax7.plot(tiempo, ZValores, linewidth=Grosor_linea, label=r'$z$')
ax7.plot(tiempo, RefZ, linewidth=Grosor_linea, label=r'$z_{ref}$', color='red')
ax7.set_xlabel('Time [s]', fontsize=28)
#ax4.set_ylabel('Distance [m]', fontsize=28)
ax7.grid(True, linestyle='-.')
ax7.set_ylim(-0.50, 1.15)
ax7.legend(loc='lower left',ncol=2, fontsize='28')
ax7.tick_params(axis='both', labelsize=28)

plt.tight_layout()
plt.savefig("Graficas_xyz_sinEst-PID_classic.pdf")


# Crear una figura con tres subgráficas
fig, (ax2, ax3, ax4) = plt.subplots(3, 1, figsize=(12, 8))



# Graficar el error de seguimiento para el eje x en la segunda subgráfica
ax2.plot(tiempo, ErrorX, linewidth=Grosor_linea)
#ax2.set_xlabel('Time[s]', fontsize=20)
ax2.set_ylabel(r'$\epsilon_{x}$ [m]', fontsize=28)
ax2.grid(True, linestyle='-.')
ax2.set_ylim(-0.40, 1.10)
#ax2.legend(ncol=2,fontsize='28')
ax2.tick_params(axis='x', labelsize=28)  # Tamaño de la fuente para los números del eje x
ax2.tick_params(axis='y', labelsize=28)  # Tamaño de la fuente para los números del eje y

# Graficar el error de seguimiento para el eje y en la tercera subgráfica
ax3.plot(tiempo, ErrorY, linewidth=Grosor_linea)

ax3.set_ylabel(r'$\epsilon_{y}$ [m]', fontsize=28)
ax3.grid(True, linestyle='-.')
#ax3.legend(ncol=2,fontsize='28')
ax3.set_ylim(-0.40, 1.10)
ax3.tick_params(axis='x', labelsize=28)  # Tamaño de la fuente para los números del eje x
ax3.tick_params(axis='y', labelsize=28)  # Tamaño de la fuente para los números del eje y


# Graficar el error de seguimiento para el eje z en la primera subgráfica
ax4.plot(tiempo, ErrorZ, linewidth=Grosor_linea)
#ax1.set_xlabel('Time[s]', fontsize=20)
ax4.set_xlabel('Time[s]', fontsize=28)
ax4.set_ylabel(r'$\epsilon_{z}$ [m]', fontsize=28)
ax4.grid(True, linestyle='-.')
ax4.set_ylim(-0.40, 1.10)
#ax4.legend(ncol=2,fontsize='28')
ax4.tick_params(axis='x', labelsize=28)  # Tamaño de la fuente para los números del eje x
ax4.tick_params(axis='y', labelsize=28)  # Tamaño de la fuente para los números del eje y

# Ajustar el diseño de las subgráficas para evitar superposiciones
plt.tight_layout()

# Guardar la figura en un archivo PDF
plt.savefig("ErrorSeguimiento-PID_classic.pdf")

###############################################################################################################
# # Crear una figura con tres subgráficas
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

# # Graficar el error de seguimiento para el eje z en la primera subgráfica
# ax1.plot(tiempo, ISE_z, linewidth=Grosor_linea, label= "ISE z")
# #ax1.set_xlabel('Time[s]', fontsize=20)
# ax1.set_ylabel(r'$\epsilon_{z}$ [m]', fontsize=28)
# ax1.grid(True, linestyle='-.')
# ax1.legend(fontsize='28')
# ax1.tick_params(axis='x', labelsize=28)  # Tamaño de la fuente para los números del eje x
# ax1.tick_params(axis='y', labelsize=28)  # Tamaño de la fuente para los números del eje y

# # Graficar el error de seguimiento para el eje x en la segunda subgráfica
# ax2.plot(tiempo, ISE_x, linewidth=Grosor_linea, label= "ISE x")
# #ax2.set_xlabel('Time[s]', fontsize=20)
# ax2.set_ylabel(r'$\epsilon_{x}$ [m]', fontsize=28)
# ax2.grid(True, linestyle='-.')
# ax2.legend(fontsize='28')
# ax2.tick_params(axis='x', labelsize=28)  # Tamaño de la fuente para los números del eje x
# ax2.tick_params(axis='y', labelsize=28)  # Tamaño de la fuente para los números del eje y

# # Graficar el error de seguimiento para el eje y en la tercera subgráfica
# ax3.plot(tiempo, ISE_y, linewidth=Grosor_linea, label= "ISE y")
# ax3.set_xlabel('Time[s]', fontsize=28)
# ax3.set_ylabel(r'$\epsilon_{y}$ [m]', fontsize=28)
# ax3.grid(True, linestyle='-.')
# ax3.legend(fontsize='28')
# ax3.tick_params(axis='x', labelsize=28)  # Tamaño de la fuente para los números del eje x
# ax3.tick_params(axis='y', labelsize=28)  # Tamaño de la fuente para los números del eje y

# # Ajustar el diseño de las subgráficas para evitar superposiciones
# plt.tight_layout()

# # Guardar la figura en un archivo PDF
# plt.savefig("ErrorISE.pdf")

###############################################################################################################
# # Crear una figura con tres subgráficas
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

# # Graficar el error de seguimiento para el eje z en la primera subgráfica
# ax1.plot(tiempo, IAE_z, linewidth=Grosor_linea, label= "IAE z")
# #ax1.set_xlabel('Time[s]', fontsize=20)
# ax1.set_ylabel(r'$\epsilon_{z}$ [m]', fontsize=28)
# ax1.grid(True, linestyle='-.')
# ax1.legend(fontsize='28')
# ax1.tick_params(axis='x', labelsize=28)  # Tamaño de la fuente para los números del eje x
# ax1.tick_params(axis='y', labelsize=28)  # Tamaño de la fuente para los números del eje y

# # Graficar el error de seguimiento para el eje x en la segunda subgráfica
# ax2.plot(tiempo, IAE_x, linewidth=Grosor_linea, label= "IAE x")
# #ax2.set_xlabel('Time[s]', fontsize=20)
# ax2.set_ylabel(r'$\epsilon_{x}$ [m]', fontsize=28)
# ax2.grid(True, linestyle='-.')
# ax2.legend(fontsize='28')
# ax2.tick_params(axis='x', labelsize=28)  # Tamaño de la fuente para los números del eje x
# ax2.tick_params(axis='y', labelsize=28)  # Tamaño de la fuente para los números del eje y

# # Graficar el error de seguimiento para el eje y en la tercera subgráfica
# ax3.plot(tiempo, IAE_y, linewidth=Grosor_linea, label= "IAE y")
# ax3.set_xlabel('Time[s]', fontsize=28)
# ax3.set_ylabel(r'$\epsilon_{y}$ [m]', fontsize=28)
# ax3.grid(True, linestyle='-.')
# ax3.legend(fontsize='28')
# ax3.tick_params(axis='x', labelsize=28)  # Tamaño de la fuente para los números del eje x
# ax3.tick_params(axis='y', labelsize=28)  # Tamaño de la fuente para los números del eje y

# # Ajustar el diseño de las subgráficas para evitar superposiciones
# plt.tight_layout()

# # Guardar la figura en un archivo PDF
# plt.savefig("ErrorIAE.pdf")

###############################################################################################################
###############################################################################################################
T = np.arange(len(u1))

num_elementos = len(u1)

# # Crear el vector de tiempo
time = np.linspace(0, duration_seconds, num_elementos)

print(len(u1), len(u2), len(u3))

fig, (ax8, ax9, ax10) = plt.subplots(3, 1, figsize=(12, 8))



# Grafica X
ax8.plot(time, u1, linewidth=Grosor_linea, label=r'$u_1$')
#ax5.set_xlabel('Time [s]', fontsize=28)
ax8.set_ylim(-30, 100)
ax8.grid(True, linestyle='-.')
ax8.legend(loc='upper right', fontsize='28')
ax8.tick_params(axis='both', labelsize=28)

# Grafica Y
ax9.plot(time, u2, linewidth=Grosor_linea, label=r'$u_2$')
ax9.set_ylabel('Input', fontsize=28)
ax9.set_ylim(-30, 100)
#ax9.set_ylabel('u3', fontsize=28)
ax9.grid(True, linestyle='-.')
ax9.legend(loc='upper right', fontsize='28')
ax9.tick_params(axis='both', labelsize=28)


# Grafica Z
ax10.plot(time, u3, linewidth=Grosor_linea, label=r'$u_3$')
ax10.set_xlabel('Time [s]', fontsize=28)
ax10.set_ylim(-30, 110)
#ax7.set_ylabel('u1', fontsize=28)
ax10.grid(True, linestyle='-.')
ax10.legend(fontsize='28') 
ax10.tick_params(axis='both', labelsize=28)


plt.tight_layout()
plt.savefig("Inputs-PID_classic.pdf")


################################################################################################################

fig = plt.figure()
ax8 =  fig.add_subplot(projection='3d')

Positions2 = np.array(ZValores).reshape(-1, 1)
Positions1 = np.array(ReferenciaZ).reshape(-1, 1)

ax8.plot(XValores, YValores, ZValores,linewidth=Grosor_linea, label='UAS trajectory')
ax8.plot(RefX, RefY, RefZ,linewidth=Grosor_linea, label='Desired trajectory', color='red')
ax8.legend(loc='upper left',ncol=2, fontsize=14)

# Configurar los límites de los ejes
# Asegúrate de que los valores en los ejes x, y, z sean los mismos para que la misma cantidad de elementos aparezca en cada eje
valores_min = min(min(XValores), min(YValores), min(RefX), min(RefY))
valores_max = max(max(XValores), max(YValores), max(RefX), max(RefY))

# Establecer los límites de los ejes x, y, z
# Set the number of ticks on the x-axis and y-axis
ax8.locator_params(axis='x', nbins=5)
ax8.locator_params(axis='y', nbins=5)
ax8.locator_params(axis='z', nbins=5)

ax8.set_xlim(valores_min, valores_max)
ax8.set_ylim(valores_min, valores_max)
#ax8.set_zlim(valores_min, valores_max)


# Etiquetas de los ejes
ax8.set_xlabel(r'$x$[m]', fontsize=14)
ax8.set_ylabel(r'$y$[m]', fontsize=14, rotation=60)
ax8.set_zlabel(r'$z$[m]', fontsize=14,rotation=60)
# Para el eje x
ax8.tick_params(axis='x', pad=2)  # Ajusta el espacio entre los números y las etiquetas del eje x

# Para el eje y
ax8.tick_params(axis='y', pad=2)  # Ajusta el espacio entre los números y las etiquetas del eje y

# Para el eje z (en una gráfica 3D)
ax8.tick_params(axis='z', pad=2)  # Ajusta el espacio entre los números y las etiquetas del eje z

#ax8.yaxis._axinfo['label']['space_factor'] = 10.0

plt.xticks(fontsize='14')
plt.yticks(fontsize='14')
ax8.tick_params(axis='z', labelsize=14)  # Tamaño de la fuente para los ticks del eje z


plt.savefig("MovimientoUAV-PID_classic.pdf")



#plt.show()


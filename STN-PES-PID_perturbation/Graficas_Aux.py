import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D #axes3d
#import plotly.graph_objects as go
import math
import matplotlib.ticker as ticker
from scipy.integrate import cumtrapz
from scipy.integrate import trapz
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
plt.rcParams['font.family'] = 'serif'  # Use a generic serif font
plt.rcParams['font.serif'] = 'DejaVu Serif'

# Leer el archivo CSV y cargarlo en un DataFrame
datosXYZ = pd.read_csv('PosicionXYZ.csv', header=None)
########################################################
Kp = pd.read_csv('Kp.csv', header=None)
Ki = pd.read_csv('Ki.csv', header=None)
Kd = pd.read_csv('Kd.csv', header=None)
########################################################
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
    #print("Input string:", string)
    # Remove square brackets and split by spaces
    values = string.replace('[', '').replace(']', '').split()
    #print("Values after splitting:", values)
    # Convert each part to float and return as numpy array
    try:
        float_values = [float(value) for value in values]
        #print("Float values:", float_values)
        return np.array(float_values)
    except ValueError as e:
        #print("Error converting to float:", e)
        return #np.array([0, 0, 0])  # Return an empty array if conversion fails


# Crear una variable de índice basada en la longitud de los datos

Array_without_str = datosXYZ.applymap(str_to_float).values.flatten()
Array_without_str_ref = ReferenciaXY.applymap(str_to_float).values.flatten()

Array_Kp = Kp.applymap(str_to_float).values.flatten()
Array_Ki = Ki.applymap(str_to_float).values.flatten()
Array_kd = Kd.applymap(str_to_float).values.flatten()

ZValores = [arr[0] for arr in Array_without_str] ###Extraer valores
XValores = [arr[1] for arr in Array_without_str] ###Extraer valores
YValores = [arr[2] for arr in Array_without_str] ###Extraer valores

RefX = np.array([arr[0] for arr in Array_without_str_ref])
RefY = np.array([arr[1] for arr in Array_without_str_ref])
RefZ = ReferenciaZ.applymap(str_to_float).values.flatten()

u1 = InputX.values.flatten()
u2 = InputY.values.flatten()
u3 = InputZ.values.flatten()

DerivateU1 = np.zeros(len(u1) - 1)
DerivateU2 = np.zeros(len(u1) - 1)
DerivateU3 = np.zeros(len(u1) - 1)

DerivateZ = np.zeros(len(u1) - 1)
DerivateX = np.zeros(len(u1) - 1)
DerivateY = np.zeros(len(u1) - 1)



# for i in range(len(u1) - 1):
#     DerivateU1[i] = u1[i+1] - u1[i]
#     DerivateU2[i] = u2[i+1] - u2[i]
#     DerivateU3[i] = u3[i+1] - u3[i]
#     print("Valor de U", u3[i], u3[i+1])
#     print("Derivada de U",DerivateU1[i], DerivateU2[i], DerivateU3[i])

# for i in range(len(ZValores) - 1):
#     DerivateZ[i] = ZValores[i] - u1[i+1]
#     DerivateX[i] = XValores[i] - u2[i+1]
#     DerivateY[i] = YValores[i] - u3[i+1]


# # Check for division by zero
# if np.any(DerivateU3 != 0):
#     Gamma1 = DerivateZ / DerivateU3
# else:
#     # Handle the case when division by zero occurs
#     Gamma1 = np.zeros_like(DerivateZ)  # Assign zeros or any other suitable value
# # Check for division by zero
# if np.any(DerivateU1 != 0):
#     Gamma2 = DerivateX / DerivateU1
# else:
#     # Handle the case when division by zero occurs
#     Gamma2 = np.zeros_like(DerivateX)  # Assign zeros or any other suitable value
# # Check for division by zero
# if np.any(DerivateU2 != 0):
#     Gamma3 = DerivateY / DerivateU2
# else:
#     # Handle the case when division by zero occurs
#     Gamma3 = np.zeros_like(DerivateY)  # Assign zeros or any other suitable value

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

# for i in range(len(ErrorZ)):
#     Error_intZ += (ErrorZ[i]**2)*dt
#     Error_intX += (ErrorX[i]**2)*dt
#     Error_intY += (ErrorY[i]**2)*dt
#     ISE_z.append(Error_intZ) #= np.sum(ErrorZ**2) * dt
#     ISE_x.append(Error_intX)
#     ISE_y.append(Error_intY)
#     ITSE_z.append(Error_intZ*tiempo[i]) #= np.sum(ErrorZ**2) * dt
#     ITSE_x.append(Error_intX*tiempo[i])
#     ITSE_y.append(Error_intY*tiempo[i])

# # ISE_z = np.sum(ErrorZ**2) * dt
# # ISE_x = np.sum(ErrorX**2) * dt
# # ISE_y = np.sum(ErrorY**2) * dt

# for i in range(len(ErrorZ)):
#     Error_absZ += np.abs(ErrorZ[i])*dt
#     Error_absX += np.abs(ErrorX[i])*dt
#     Error_absY += np.abs(ErrorY[i])*dt
#     IAE_z.append(Error_absZ) #= np.sum(ErrorZ**2) * dt
#     IAE_x.append(Error_absX)
#     IAE_y.append(Error_absY)
#     ITAE_z.append(Error_absZ*tiempo[i]) #= np.sum(ErrorZ**2) * dt
#     ITAE_x.append(Error_absX*tiempo[i])
#     ITAE_y.append(Error_absY*tiempo[i])


# Calcular el error al cuadrado
errores_al_cuadrado_z = ErrorZ**2
# Calcular el error al cuadrado
errores_al_cuadrado_x = ErrorX**2
# Calcular el error al cuadrado
errores_al_cuadrado_y = ErrorY**2

# Calcular la Integral del Error al Cuadrado usando el método del trapecio
ISE_z = trapz(errores_al_cuadrado_z, tiempo)
# Calcular la Integral del Error al Cuadrado usando el método del trapecio
ISE_x = trapz(errores_al_cuadrado_x, tiempo)
# Calcular la Integral del Error al Cuadrado usando el método del trapecio
ISE_y = trapz(errores_al_cuadrado_y, tiempo)

# Calcular el error al cuadrado multiplicado por el tiempo
error_cuadrado_por_tiempo_z = errores_al_cuadrado_z * tiempo
# Calcular el error al cuadrado multiplicado por el tiempo
error_cuadrado_por_tiempo_x = errores_al_cuadrado_x * tiempo
# Calcular el error al cuadrado multiplicado por el tiempo
error_cuadrado_por_tiempo_y = errores_al_cuadrado_y * tiempo



# Calcular la Integral del Error Cuadrado Multiplicado por el Tiempo usando el método del trapecio
ITSE_z = trapz(error_cuadrado_por_tiempo_z, tiempo)
# Calcular la Integral del Error Cuadrado Multiplicado por el Tiempo usando el método del trapecio
ITSE_x = trapz(error_cuadrado_por_tiempo_x, tiempo)
# Calcular la Integral del Error Cuadrado Multiplicado por el Tiempo usando el método del trapecio
ITSE_y = trapz(error_cuadrado_por_tiempo_y, tiempo)



# Calcular el valor absoluto de cada error
errores_absolutos_z = np.abs(ErrorZ)
# Calcular el valor absoluto de cada error
errores_absolutos_x = np.abs(ErrorX)
# Calcular el valor absoluto de cada error
errores_absolutos_y = np.abs(ErrorY)



# Calcular la Integral del Valor Absoluto del Error usando el método del trapecio
IAE_z = trapz(errores_absolutos_z, tiempo)
# Calcular la Integral del Valor Absoluto del Error usando el método del trapecio
IAE_x = trapz(errores_absolutos_x, tiempo)
# Calcular la Integral del Valor Absoluto del Error usando el método del trapecio
IAE_y = trapz(errores_absolutos_y, tiempo)


# Calcular el valor absoluto del error multiplicado por el tiempo
error_absoluto_por_tiempo_z = errores_absolutos_z * tiempo
# Calcular el valor absoluto del error multiplicado por el tiempo
error_absoluto_por_tiempo_x = errores_absolutos_x * tiempo
# Calcular el valor absoluto del error multiplicado por el tiempo
error_absoluto_por_tiempo_y = errores_absolutos_y * tiempo



# Calcular la Integral del Valor Absoluto del Error Multiplicado por el Tiempo usando el método del trapecio
ITAE_z = trapz(error_absoluto_por_tiempo_z, tiempo)
# Calcular la Integral del Valor Absoluto del Error Multiplicado por el Tiempo usando el método del trapecio
ITAE_x = trapz(error_absoluto_por_tiempo_x, tiempo)
# Calcular la Integral del Valor Absoluto del Error Multiplicado por el Tiempo usando el método del trapecio
ITAE_y = trapz(error_absoluto_por_tiempo_y, tiempo)




print("Integral del Error al Cuadrado (ISE):", ISE_z, ISE_x, ISE_y)
print("Integral del Error Cuadrado Multiplicado por el Tiempo (ITSE):", ITSE_z, ITSE_x, ITSE_y)
print("Integral del Valor Absoluto del Error (IAE):", IAE_z, IAE_x, IAE_y)
print("Integral del Valor Absoluto del Error Multiplicado por el Tiempo (ITAE):", ITAE_z, ITAE_x, ITAE_y)

z = np.zeros_like(time_vector)
x_1 = np.zeros_like(time_vector)
y = np.zeros_like(time_vector)
# Set values based on conditions
for i, T in enumerate(time_vector):
    if (T > 15 and T < 20) or (T > 30 and T < 33) or (T > 40 and T < 43):
        z[i] = 20
    if (T > 40 and T < 45) or (T > 55 and T < 57):
        x_1[i] = 20
    if (T > 48 and T < 53) or (T > 60 and T < 63):
        y[i] = 20



ZKp = [arr[0] if arr is not None else 160  for arr in Array_Kp]
XKp = [arr[1] if arr is not None else 60 for arr in Array_Kp] ###Extraer valores
YKp = [arr[2] if arr is not None else 60 for arr in Array_Kp] ###Extraer valores

ZKi = [arr[0] if arr is not None else 195 for arr in Array_Ki]###Extraer valores
XKi = [arr[1] if arr is not None else 195 for arr in Array_Ki] ###Extraer valores
YKi = [arr[2] if arr is not None else 195 for arr in Array_Ki] ###Extraer valores

ZKd = [arr[0] if arr is not None else 25 for arr in Array_kd]###Extraer valores
XKd = [arr[1] if arr is not None else 15 for arr in Array_kd] ###Extraer valores
YKd = [arr[2] if arr is not None else 15 for arr in Array_kd] ###Extraer valores

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
######################################################################################################################################
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

    # Graficar los datos en los ejes correspondientes
ax1.plot(tiempo, XKp, linewidth=Grosor_linea, label='$K_p$ gain')
#ax1.set_xlabel('Time [s]')
#ax1.set_ylabel('P')
ax1.grid(True, linestyle='-.')
ax1.legend(loc='upper left', fontsize='28')
ax1.tick_params(axis='both', labelsize=28)

ax2.plot(tiempo, XKi,linewidth=Grosor_linea, label='$K_i$ gain')
#ax2.set_xlabel('Time [s]')
#ax2.set_ylabel('I')
ax2.grid(True, linestyle='-.')
ax2.legend(loc='upper left', fontsize='28')
ax2.tick_params(axis='both', labelsize=28)

ax3.plot(tiempo, XKd, linewidth=Grosor_linea, label='$K_d$ gain')
ax3.set_xlabel('Time[s]', fontsize=28)

#ax3.set_ylabel('D')
ax3.grid(True, linestyle='-.')
ax3.legend(loc='lower left', fontsize='28')
ax3.tick_params(axis='both', labelsize=28)
ax3.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    

    # Establecer etiquetas y límites de los ejes


    # Ajustar el diseño de la figura para que no haya superposición de subgráficos
plt.tight_layout()

    # Guardar la figura en un archivo PDF
plt.savefig("GananciasX.pdf")


#######################################################################################################33
    # Crear una nueva figura y ejes
fig6, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

    # Graficar los datos en los ejes correspondientes
ax1.plot(tiempo, YKp, linewidth=Grosor_linea, label=r'$K_p$ gain')
#ax1.set_xlabel('Time [s]')
#ax1.set_ylabel('P')
ax1.grid(True, linestyle='-.')
ax1.legend(loc='lower left', fontsize='28')
ax1.tick_params(axis='x', labelsize=28)  # Tamaño de la fuente para los números del eje x
ax1.tick_params(axis='y', labelsize=28)  # Tamaño de la fuente para los números del eje y

ax2.plot(tiempo, YKi,linewidth=Grosor_linea, label=r'$K_i$ gain')
#ax2.set_xlabel('Time [s]')
#ax2.set_ylabel('I')
ax2.grid(True, linestyle='-.')
ax2.legend(loc='lower left', fontsize='28')
ax2.tick_params(axis='x', labelsize=28)  # Tamaño de la fuente para los números del eje x
ax2.tick_params(axis='y', labelsize=28)  # Tamaño de la fuente para los números del eje y

ax3.plot(tiempo, YKd, linewidth=Grosor_linea, label=r'$K_d$ gain')
ax3.set_xlabel('Time [s]', fontsize=28)
#ax3.set_ylabel('D')
ax3.grid(True, linestyle='-.')
ax3.legend(loc='lower left', fontsize='28')
ax3.tick_params(axis='x', labelsize=28)  # Tamaño de la fuente para los números del eje x
ax3.tick_params(axis='y', labelsize=28)  # Tamaño de la fuente para los números del eje y
    

    # Establecer etiquetas y límites de los ejes



    # Ajustar el diseño de la figura para que no haya superposición de subgráficos


plt.tight_layout()

    # Guardar la figura en un archivo PDF
fig6.savefig("GananciasY.pdf")


########################################################################################################
    # Crear una nueva figura y ejes
fig7, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

    # Graficar los datos en los ejes correspondientes
ax1.plot(tiempo, ZKp, linewidth=Grosor_linea, label=r'$K_p$ gain')
ax2.plot(tiempo, ZKi, linewidth=Grosor_linea, label=r'$K_i$ gain')
ax3.plot(tiempo, ZKd, linewidth=Grosor_linea, label=r'$K_d$ gain')
#plt.ticklabel_format(useOffset=False, style='plain')  ########
  
    # Establecer etiquetas y límites de los ejes
#ax1.set_xlabel('Time [s]')
#ax1.set_ylabel('P')
ax1.grid(True, linestyle='-.')
ax1.legend(loc='upper left', fontsize='28')
ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=True))
ax2.grid(True, linestyle='-.')
ax2.legend(loc='lower right', fontsize='28')

ax3.set_xlabel('Time [s]', fontsize=28)
ax3.grid(True, linestyle='-.')
ax3.legend(loc='lower right', fontsize='28')
ax3.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

formatter = ScalarFormatter()
formatter.set_powerlimits((-3, 3))  # Limitar la notación científica para valores menores a 10^(-3) y mayores a 10^3
formatter.set_scientific(True)     # Habilitar la notación científica
formatter.set_useMathText(True)     # Usar notación científica con exponentes en formato LaTeX
ax1.yaxis.set_major_formatter(formatter)
    # Ajustar el diseño de la figura para que no haya superposición de subgráficos
ax1.tick_params(axis='x', labelsize=28)  # Tamaño de la fuente para los números del eje x
ax1.tick_params(axis='y', labelsize=28)  # Tamaño de la fuente para los números del eje y
ax2.tick_params(axis='x', labelsize=28)  # Tamaño de la fuente para los números del eje x
ax2.tick_params(axis='y', labelsize=28)  # Tamaño de la fuente para los números del eje y
ax3.tick_params(axis='x', labelsize=28)  # Tamaño de la fuente para los números del eje x
ax3.tick_params(axis='y', labelsize=28)


plt.gcf().get_axes()[0].yaxis.get_offset_text().set_size(28)

plt.tight_layout()

    # Guardar la figura en un archivo PDF
fig7.savefig("GananciasZ.pdf")

######################################################################################################################################
######################################### GRAFICAS COMBINADAS ########################################################################
fig, (ax4, ax5, ax6) = plt.subplots(3, 1, figsize=(12, 8))

# Grafica Z
ax4.plot(tiempo, ZValores, linewidth=Grosor_linea, label=r'$z$')
ax4.plot(tiempo, RefZ, "--", linewidth=Grosor_linea, label=r'$z_{ref}$', color='red')
#ax4.set_xlabel('Time [s]', fontsize=28)
#ax4.set_ylabel('Distance [m]', fontsize=28)
ax4.grid(True, linestyle='-.')
ax4.legend(fontsize='28') 
ax4.tick_params(axis='both', labelsize=28)

# Grafica X
ax5.plot(tiempo, XValores, linewidth=Grosor_linea, label=r'$x$')
ax5.plot(tiempo, RefX, "--", linewidth=Grosor_linea, label=r'$x_{ref}$', color='red')
#ax5.set_xlabel('Time [s]', fontsize=28)
ax5.set_ylabel('Position [m]', fontsize=28)
ax5.grid(True, linestyle='-.')
ax5.legend(loc='upper left', fontsize='28')
ax5.tick_params(axis='both', labelsize=28)

# Grafica Y
ax6.plot(tiempo, YValores, linewidth=Grosor_linea, label=r'$y$')
ax6.plot(tiempo, RefY, "--", linewidth=Grosor_linea, label=r'$y_{ref}$', color='red')
ax6.set_xlabel('Time [s]', fontsize=28)
#ax6.set_ylabel('Distance [m]', fontsize=28)
ax6.grid(True, linestyle='-.')
ax6.legend(loc='upper left', fontsize='28')
ax6.tick_params(axis='both', labelsize=28)

plt.tight_layout()
plt.savefig("Graficas_xyz_sinEst.pdf")
############===================================================================================================
fig, (ax5) = plt.subplots(figsize=(12, 8))

# Grafica X vs Y
ax5.plot(XValores, YValores, linewidth=Grosor_linea, label=r'$x,y$')
ax5.plot(RefX, RefY, "--", linewidth=Grosor_linea, label=r'$x_{ref}, y_{ref}$', color='red')
ax5.set_xlabel('X Distance [m]', fontsize=28)
ax5.set_ylabel('Y Distance [m]', fontsize=28)
ax5.grid(True, linestyle='-.')
ax5.legend(loc='upper left', fontsize='28')
ax5.tick_params(axis='both', labelsize=28)

plt.tight_layout()
plt.savefig("Grafica_X_vs_Y.pdf")
############===================================================================================================

# Crear una figura con tres subgráficas
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

# Graficar el error de seguimiento para el eje z en la primera subgráfica
ax1.plot(tiempo, ErrorZ, linewidth=Grosor_linea, label=r'$\epsilon_{z}$')
#ax1.set_xlabel('Time[s]', fontsize=20)
ax1.set_ylabel(r'$\epsilon_{z}$ [m]', fontsize=28)
ax1.grid(True, linestyle='-.')
ax1.legend(fontsize='28')
ax1.tick_params(axis='x', labelsize=28)  # Tamaño de la fuente para los números del eje x
ax1.tick_params(axis='y', labelsize=28)  # Tamaño de la fuente para los números del eje y

# Graficar el error de seguimiento para el eje x en la segunda subgráfica
ax2.plot(tiempo, ErrorX, linewidth=Grosor_linea, label=r'$\epsilon_{x}$')
#ax2.set_xlabel('Time[s]', fontsize=20)
ax2.set_ylabel(r'$\epsilon_{x}$ [m]', fontsize=28)
ax2.grid(True, linestyle='-.')
ax2.legend(fontsize='28')
ax2.tick_params(axis='x', labelsize=28)  # Tamaño de la fuente para los números del eje x
ax2.tick_params(axis='y', labelsize=28)  # Tamaño de la fuente para los números del eje y

# Graficar el error de seguimiento para el eje y en la tercera subgráfica
ax3.plot(tiempo, ErrorY, linewidth=Grosor_linea, label=r'$\epsilon_{y}$')
ax3.set_xlabel('Time[s]', fontsize=28)
ax3.set_ylabel(r'$\epsilon_{y}$ [m]', fontsize=28)
ax3.grid(True, linestyle='-.')
ax3.legend(fontsize='28')
ax3.tick_params(axis='x', labelsize=28)  # Tamaño de la fuente para los números del eje x
ax3.tick_params(axis='y', labelsize=28)  # Tamaño de la fuente para los números del eje y

# Ajustar el diseño de las subgráficas para evitar superposiciones
plt.tight_layout()

# Guardar la figura en un archivo PDF
plt.savefig("ErrorSeguimiento.pdf")

###############################################################################################################


###############################################################################################################


###############################################################################################################
###############################################################################################################
T = np.arange(len(u1))

num_elementos = len(u1)

# # Crear el vector de tiempo
time = np.linspace(0, duration_seconds, num_elementos)

print(len(u1), len(u2), len(u3))

fig, (ax7, ax8, ax9) = plt.subplots(3, 1, figsize=(12, 8))

# Grafica Z
ax7.plot(time, u3, linewidth=Grosor_linea, label=r'$z$')
ax7.plot(time, z, linewidth=Grosor_linea, label=r'$perturbacion_z$')
#ax4.set_xlabel('Time [s]', fontsize=28)
ax7.set_ylabel('u1', fontsize=28)
ax7.grid(True, linestyle='-.')
ax7.legend(fontsize='28') 
ax7.tick_params(axis='both', labelsize=28)

# Grafica X
ax8.plot(time, u1, linewidth=Grosor_linea, label=r'$x$')
ax8.plot(time, x_1, linewidth=Grosor_linea, label=r'$perturbacion_x$')
#ax5.set_xlabel('Time [s]', fontsize=28)
ax8.set_ylabel('u2', fontsize=28)
ax8.grid(True, linestyle='-.')
ax8.legend(loc='upper left', fontsize='28')
ax8.tick_params(axis='both', labelsize=28)

# Grafica Y
ax9.plot(time, u2, linewidth=Grosor_linea, label=r'$y$')
ax9.plot(time, y, linewidth=Grosor_linea, label=r'$perturbacion_y$')
ax9.set_xlabel('Time [s]', fontsize=28)
ax9.set_ylabel('u3', fontsize=28)
ax9.grid(True, linestyle='-.')
ax9.legend(loc='upper left', fontsize='28')
ax9.tick_params(axis='both', labelsize=28)

plt.tight_layout()
plt.savefig("Inputs.pdf")
#####==========================================================================================================

###############################################################################################################
# # Crear una figura con tres subgráficas
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

# # Graficar el error de seguimiento para el eje z en la primera subgráfica
# ax1.plot(time_vector[:-1], Gamma1, linewidth=Grosor_linea, label= "Derivative z")
# #ax1.set_xlabel('Time[s]', fontsize=20)
# ax1.set_ylabel(r'$\epsilon_{z}$ [m]', fontsize=28)
# ax1.grid(True, linestyle='-.')
# ax1.legend(fontsize='28')
# ax1.tick_params(axis='x', labelsize=28)  # Tamaño de la fuente para los números del eje x
# ax1.tick_params(axis='y', labelsize=28)  # Tamaño de la fuente para los números del eje y

# # Graficar el error de seguimiento para el eje x en la segunda subgráfica
# ax2.plot(time_vector[:-1], Gamma2, linewidth=Grosor_linea, label= "Derivative x")
# #ax2.set_xlabel('Time[s]', fontsize=20)
# ax2.set_ylabel(r'$\epsilon_{x}$ [m]', fontsize=28)
# ax2.grid(True, linestyle='-.')
# ax2.legend(fontsize='28')
# ax2.tick_params(axis='x', labelsize=28)  # Tamaño de la fuente para los números del eje x
# ax2.tick_params(axis='y', labelsize=28)  # Tamaño de la fuente para los números del eje y

# # Graficar el error de seguimiento para el eje y en la tercera subgráfica
# ax3.plot(time_vector[:-1], Gamma3, linewidth=Grosor_linea, label= "Derivative y")
# ax3.set_xlabel('Time[s]', fontsize=28)
# ax3.set_ylabel(r'$\epsilon_{y}$ [m]', fontsize=28)
# ax3.grid(True, linestyle='-.')
# ax3.legend(fontsize='28')
# ax3.tick_params(axis='x', labelsize=28)  # Tamaño de la fuente para los números del eje x
# ax3.tick_params(axis='y', labelsize=28)  # Tamaño de la fuente para los números del eje y

# # Ajustar el diseño de las subgráficas para evitar superposiciones
# plt.tight_layout()

# # Guardar la figura en un archivo PDF
# plt.savefig("Derivative_example.pdf")

################################################################################################################

fig = plt.figure()
ax8 =  fig.add_subplot(projection='3d')

Positions2 = np.array(ZValores).reshape(-1, 1)
Positions1 = np.array(ReferenciaZ).reshape(-1, 1)

ax8.plot(XValores, YValores, ZValores,linewidth=Grosor_linea, label='UAS trajectory')
ax8.plot(RefX, RefY, RefZ,"--",linewidth=Grosor_linea, label='Desired trajectory', color='red')
ax8.legend(loc='upper left', fontsize=14)

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


plt.savefig("MovimientoUAV.pdf")



plt.show()


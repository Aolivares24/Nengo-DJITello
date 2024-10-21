import rospy
import tf
from nav_msgs.msg import Odometry
import csv

import matplotlib.pyplot as plt
import math
import nengo
import numpy as np
from mpl_toolkits.mplot3d import Axes3D #axes3d
import plotly.graph_objects as go
from djitellopy import Tello
import logging
from scipy.integrate import trapz

# Deshabilitar los mensajes de información (INFO) de djitellopy
logging.getLogger('djitellopy').setLevel(logging.WARNING)
###################################################################################
###################################################################################
j = 1
u = []
u2 =[]
u3 = []
Tiempo = []
flag = 0
Inputs = []
Input2= []
Input3 = []
List_kp = []
List_ki = []
List_kd = []


global t
global Time_ini

def plot_xy(data, num_xticks=5, num_yticks=5, xmin=None, xmax=None, ymin=None, ymax=None):
    j = len(Inputs)    
    print("Aqui el tiempo ", len(t), j, i, T)
    Tiempo = np.arange(j)
    k = len(List_kp)
    Time = np.arange(k)  
    #print(data[Error_int][-1])
    
    with open('PosicionXYZ.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data[posicion])

    with open('ReferenciaZ.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data[ref])

    with open('ReferenciaXY.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data[refXY])
    
    with open('InputZ.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(Inputs)
    with open('InputX.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(Input2)
    with open('InputY.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(Input3)

    #plt.figure(figsize=(9, 10))
    Fig5, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    ax1.plot(t, data[posicion][:, 0], label="Posicion Z", linewidth=4, color='blue')
    ax2.plot(t, data[posicion][:, 1], label="Posicion X", linewidth=4, color='green')
    ax3.plot(t, data[posicion][:, 2], label="Posicion Y", linewidth=4, color='yellow')
    ax1.plot(t, data[ref], label="Referencia $y_{z}$ axis", color='red')
    ax2.plot(t, data[refXY][:, 0], label="Referencia $y_{x}$ axis", color='orange')
    ax3.plot(t, data[refXY][:, 1], label="Referencia $y_{y}$ axis", color='grey')
   
    # ax3.tick_params(axis='both', labelsize=28)
    ax1.grid(True)
    ax1.legend(fontsize=28, loc='upper left') 

    #ax2.set_xlabel("Time [s]", fontsize=32)
    ax2.set_ylabel("Position $y_{p}$ [m]", fontsize=28)
    ax2.grid(True)
    ax2.legend(fontsize=28, loc='upper left') 

    ax3.set_xlabel("Time [s]", fontsize=28)
    #ax3.set_ylabel("Position $y_{p}$ [m]", fontsize=32)
    ax3.grid(True)
    ax3.legend(fontsize=28, loc='upper left') 
    
        # Set the number of ticks on the x-axis and y-axis
    ax1.locator_params(axis='x', nbins=num_xticks)
    ax1.locator_params(axis='y', nbins=num_yticks)
    ax1.tick_params(axis='both', labelsize=28)
    ax2.tick_params(axis='both', labelsize=28)
    ax3.tick_params(axis='both', labelsize=28)
    plt.tight_layout()
    
    
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28) 

    Fig5.savefig("GraficaZ.pdf")
    
    fig1, ax2 = plt.subplots(figsize=(12, 8))
    ax2.plot(t, data[error][:, 0], label="Error Z", color='orange')
    ax2.plot(t, data[error][:, 1], label="Error X", color='blue')
    ax2.plot(t, data[error][:, 2], label="Error Y", color='red')
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Position $y_{p}$ [m]")
    ax2.grid(True, linestyle='-.')
    ax2.legend(fontsize='28')
    #plt.show()
    plt.savefig("Errores.pdf")


    fig1, ax3 = plt.subplots()
    ax3.plot(Tiempo, Inputs,"--", label='U de z')
    ax3.plot(Tiempo, Input2,"--", label='U de x')
    ax3.plot(Tiempo, Input3,"--", label='U de y')
    ax3.set_xlabel('Tiempo')
    ax3.set_ylabel('Input')
    ax3.grid(True, linestyle='-.')
    ax3.legend(fontsize='28')
    #plt.show()
    plt.savefig("GraficaURedondeo.pdf")

    fig2, ax4 = plt.subplots()
    ax4.plot(t, data[u_ax][:, 0],"--", label='U de z')
    ax4.plot(t, data[u_ax][:, 1],"--", label='U de x')
    ax4.plot(t, data[u_ax][:, 2],"--", label='U de y')
    ax4.set_xlabel('Tiempo')
    ax4.set_ylabel('Input')
    ax4.grid(True, linestyle='-.')
    ax4.legend(fontsize='28')
    #plt.show()
    plt.savefig("GraficaUNengo.pdf")

    fig4, (ax4, ax5, ax6) = plt.subplots(3, 1, figsize=(12, 8))
    ax4.plot(t, data[gradiente_kp],"--", label='gradiente kp')
    ax4.set_xlabel('Tiempo')
    ax4.set_ylabel('Input')
    ax4.grid(True, linestyle='-.')
    ax4.legend(fontsize='14')
    ax5.plot(t, data[gradiente_ki],"--", label='gradiente ki')
    ax5.set_xlabel('Tiempo')
    ax5.set_ylabel('Input')
    ax5.grid(True, linestyle='-.')
    ax5.legend(fontsize='14')
    ax6.plot(t, data[gradiente_kd],"--", label='gradiente kd')
    ax6.set_xlabel('Tiempo')
    ax6.set_ylabel('Input')
    ax6.grid(True, linestyle='-.')
    ax6.legend(fontsize='14')
    #plt.show()
    plt.savefig("GraficaKp.pdf")

    fig5, (ax4, ax5, ax6) = plt.subplots(3, 1, figsize=(12, 8))
    ax4.plot(Time, List_kp,"--", label='kp')
    ax4.set_xlabel('Tiempo')
    ax4.set_ylabel('Input')
    ax4.grid(True, linestyle='-.')
    ax4.legend(fontsize='14')
    ax5.plot(Time, List_ki,"--", label='ki')
    ax5.set_xlabel('Tiempo')
    ax5.set_ylabel('Input')
    ax5.grid(True, linestyle='-.')
    ax5.legend(fontsize='14')
    ax6.plot(Time, List_kd,"--", label='kd')
    ax6.set_xlabel('Tiempo')
    ax6.set_ylabel('Input')
    ax6.grid(True, linestyle='-.')
    ax6.legend(fontsize='14')
    plt.savefig("GraficaKp_Ki_Kd.pdf")

    with open('Kp.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(List_kp)
    with open('Ki.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(List_ki)
    with open('Kd.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(List_kd)
    MyTello.land()
    plt.show()

    
def Plot3D_XYZ(data):
    fig = plt.figure()
    ax8 =  fig.add_subplot(projection='3d')

    Grosor_linea = 4

    ZValores = data[posicion][:, 0]
    XValores = data[posicion][:, 1]
    YValores = data[posicion][:, 2]

    RefZ = data[ref]
    RefX = data[refXY][:, 0]
    RefY = data[refXY][:, 1]

    Positions2 = np.array(ZValores).reshape(-1, 1)
    Positions1 = np.array(RefZ).reshape(-1, 1)

    ax8.plot(XValores, YValores, Positions2,linewidth=Grosor_linea, label='UAS trajectory')
    ax8.plot(RefX, RefY, Positions1,"--",linewidth=Grosor_linea, label='Desired trajectory', color='red')
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


###################################################################################
rospy.init_node('myNode')
Frecuencia = 200 ##HZ
CicleNengo = (400)/Frecuencia
Extra = 10/Frecuencia
rate = rospy.Rate(Frecuencia) # ROS Rate at 100Hz



class ExternalInput(nengo.Node):
    Kp = [160, 60, 60]  # Declare Kp as a class variable
    Kd = [25, 15, 15] 
    Ki = [195, 195, 195]
    def __init__(self, name):
        self.pose = 0
        self.mu_Kp = 5e-3
        self.mu_Ki = 1e-4
        self. mu_Kd = 5e-3
        self.dest_sub = rospy.Subscriber(f'/mocap_node/tello_02/Odom', Odometry, self.callback, queue_size=1)      
        super(ExternalInput, self).__init__(label=name, output= self.tick, size_in=12, size_out=3)   
    def callback (self, msg):   
        self.pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]#, yaw_rad]
    def tick(self,t, valores):     
        uz = np.clip(round(valores[0]), -100, 100)
        ux = np.clip(round(valores[1]), -100, 100)
        uy = np.clip(round(valores[2]), -100, 100)
        Inputs.append(uz)
        Input2.append(ux)
        Input3.append(uy)
        List_kp.append(self.Kp)
        List_ki.append(self.Ki)
        List_kd.append(self.Kd)
        self.Kp = self.Kp + self.mu_Kp * valores[[3, 4, 5]]
        self.Ki = self.Ki + self.mu_Ki * valores[[6, 7, 8]]
        self.Kd = self.Kd + self.mu_Kd * valores[9:]
        MyTello.send_rc_control(int(ux), int(uy), int(uz), 0) 
        return self.pose[2], self.pose[0], self.pose[1]
    

class Models (object):
    def __init__(self, frecuancy):
        self.Frecuncy = frecuancy
        self.Kp = ExternalInput.Kp  # Use the class variables from ExternalInput
        self.Kd = ExternalInput.Kd
        self.ki_aux = ExternalInput.Ki
        self.Ki = [1, 1, 1]
        ##############################################
        rc = 1.0
        self.l = 2
        self.w = math.pi/30
        self.SetpointFinal = [1.5, rc , rc] ### Altura, radio de la flor, radio del circulo
        ##############################################
        self.J = 200 ####Numero de neuronas
        self.I = 3 ######Numero de salidas
        self.O = 3 ######Numero de entradas
        self.model = nengo.Network(label='drone')
        self.tau = 1/self.Frecuncy
        self.synaps = 0.01

        with self.model:
            ################################## ENTRADA DE OPTITRACK #################################################
            external_input = ExternalInput('name') ################# Se crea un nodo
            
            Input_encoder = nengo.Ensemble(n_neurons=self.J,dimensions=self.I, radius=2.5)
            nengo.Connection(external_input,Input_encoder)
            ################################### GENERACION Y CODIFICACION DE #########################################
            ################################### SEÑAL DE REFERENCIA ##################################################
            def reference_signals_xy_func(t):
                if t > 5 and t<15:
                    return[self.SetpointFinal[1], 0]
                if t >= 15 and t<25:
                    return [self.SetpointFinal[1], self.SetpointFinal[2]]
                if t >=25 and t<35:
                    return [-self.SetpointFinal[1], self.SetpointFinal[2]]
                if t>=35:
                    return [-self.SetpointFinal[1], 0]
                else:
                    return [0.0, 0.0]  # Signal is zero
            ##########################################################################################################

            Reference_SignalsZ = nengo.Node(lambda t: self.SetpointFinal[0]) ######Escalon en Z
            Encoded_Reference = nengo.Ensemble(n_neurons=self.J, dimensions=1, radius=2)
            nengo.Connection(Reference_SignalsZ,Encoded_Reference, synapse=None)

            Reference_SignalsXY = nengo.Node(reference_signals_xy_func)#[self.SetpointFinal[1] / (1 + math.exp(-(1*t-20))), self.SetpointFinal[2] / (1 + math.exp(-(1*t-30)))]) ######Sigmoide que va aseguir
            Encoded_ReferenceXY = nengo.Ensemble(n_neurons=self.J, dimensions=2, radius=2)
            nengo.Connection(Reference_SignalsXY,Encoded_ReferenceXY, synapse=None)

            #########################################################################################################
            #########################################################################################################
            Error = nengo.Ensemble(n_neurons=self.J, dimensions=self.I, radius = 3)
            nengo.Connection(Encoded_Reference,Error[0], transform=1, synapse = None)
            nengo.Connection(Encoded_ReferenceXY,Error[[1, 2]], transform=1, synapse = None)

            nengo.Connection(Input_encoder,Error,transform=-1, synapse = None)
            #########################################################################################################
            #####################################(Error_act - Error_pre)/ dt ########################################
            d_e_z = nengo.Ensemble(n_neurons=self.J, dimensions = 1, radius = 2)
            nengo.Connection(Error[0],d_e_z,synapse=0.05,transform= self.Frecuncy) #################### F = 30 Hz
            nengo.Connection(Error[0],d_e_z,synapse=0.1,transform= -self.Frecuncy) ###################################

            d_e_x = nengo.Ensemble(n_neurons=self.J, dimensions = 1, radius = 2)
            nengo.Connection(Error[1],d_e_x,synapse=0.05,transform= self.Frecuncy) #################### F = 30 Hz
            nengo.Connection(Error[1],d_e_x,synapse=0.1,transform= -self.Frecuncy) ###################################

            d_e_y = nengo.Ensemble(n_neurons=self.J, dimensions = 1, radius = 2)
            nengo.Connection(Error[2],d_e_y,synapse=0.05,transform= self.Frecuncy) #################### F = 30 Hz
            nengo.Connection(Error[2],d_e_y,synapse=0.1,transform= -self.Frecuncy) ###################################
            #########################################################################################################
            Error_integral = nengo.Ensemble(n_neurons=self.J, dimensions=self.I, radius= 4)
            nengo.Connection(Error_integral, Error_integral,transform=1, synapse=self.tau)
            nengo.Connection(Error[0], Error_integral[0], transform= self.tau*self.ki_aux[0], synapse=self.tau)
            nengo.Connection(Error[1], Error_integral[1], transform= self.tau*self.ki_aux[1], synapse=self.tau)
            nengo.Connection(Error[2], Error_integral[2], transform= self.tau*self.ki_aux[2], synapse=self.tau)
            #########################################################################################################
            #########################################################################################################

            controlEns = nengo.Ensemble(n_neurons= 800, dimensions=self.O, radius=100)
            nengo.Connection(Error,controlEns,transform= self.Kp, synapse=None)  ####### Ganancia proporcional
            nengo.Connection(d_e_z,controlEns[0],transform= self.Kd[0], synapse=None) ####### Ganacia derivativa
            nengo.Connection(d_e_x,controlEns[1],transform= self.Kd[1], synapse=None) ####### Ganacia derivativa
            nengo.Connection(d_e_y,controlEns[2],transform= self.Kd[1], synapse=None) ####### Ganacia derivativa
            nengo.Connection(Error_integral,controlEns,transform= self.Ki, synapse=None) ####### Ganancia integral

            ########################################################################################################
            ########################################################################################################
            def gradi_kp(t,x):
                return x[:3]*x[3:]
            def gradi_ki(t,x):
                return x[:3]*x[[3, 4, 5]]*x[6:]
            def gradi_kd(t,x):
                return x[:3]*x[[3, 4, 5]]*x[6:]
            Grad_kp = nengo.Node(gradi_kp, size_in=6, size_out=3)#nengo.Ensemble(n_neurons=self.J, dimensions = self.I, radius = 2)
            Grad_ki = nengo.Node(gradi_ki, size_in=9, size_out=3)#nengo.Ensemble(n_neurons=self.J, dimensions = self.I, radius = 2)
            Grad_kd = nengo.Node(gradi_kd, size_in=9, size_out=3)#nengo.Ensemble(n_neurons=self.J, dimensions = self.I, radius = 2)
            d_y = nengo.Ensemble(n_neurons=self.J, dimensions=self.I, radius=2)
            d_u = nengo.Ensemble(n_neurons=self.J, dimensions=self.I, radius=100)
            nengo.Connection(Input_encoder, d_y, synapse=0.05)
            nengo.Connection(Input_encoder, d_y, synapse=0.1, transform=-1)        
            nengo.Connection(controlEns, d_u, synapse=0.05)
            nengo.Connection(controlEns, d_u, synapse=0.1, transform=-1)
            def division(t,x):
                d_y = x[:3] 
                d_x = x[3:]
                if np.any(d_x != 0):
                    division = np.divide(d_y, d_x, out=np.zeros_like(d_y), where=(d_x != 0))
                else:
                    division = np.zeros_like(d_y)
                return division  # Avoid division by zero by adding a small value
            Gamma = nengo.Node(division, size_in=6, size_out=3)

            nengo.Connection(d_y, Gamma[:3], synapse=self.synaps)
            nengo.Connection(d_u, Gamma[3:], synapse=self.synaps)
            nengo.Connection(Gamma, Grad_kp[:3], synapse=self.synaps)
            nengo.Connection(Gamma, Grad_ki[:3], synapse=self.synaps)
            nengo.Connection(Gamma, Grad_kd[:3], synapse=self.synaps)
            nengo.Connection(Error, Grad_kp[3:],function=lambda x: x**2, synapse=self.synaps)
            nengo.Connection(Error, Grad_ki[[3, 4, 5]], synapse=self.synaps)
            nengo.Connection(Error_integral, Grad_ki[6:], synapse=self.synaps)
            nengo.Connection(Error, Grad_kd[[3, 4, 5]], synapse=self.synaps)
            nengo.Connection(d_e_z, Grad_kd[6], synapse=self.synaps)
            nengo.Connection(d_e_x, Grad_kd[7], synapse=self.synaps)
            nengo.Connection(d_e_y, Grad_kd[8], synapse=self.synaps)
            
            ########################################################################################################
            ########################################################################################################
            controlSignals = nengo.Node(size_in=self.O)
            nengo.Connection(controlEns, controlSignals, transform=1, synapse=0.1)
            nengo.Connection(controlSignals, external_input[:3], synapse=0.1)
            nengo.Connection(Grad_kp, external_input[[3, 4, 5]], synapse=0.1)
            nengo.Connection(Grad_ki, external_input[[6, 7, 8]], synapse=0.1)
            nengo.Connection(Grad_kd, external_input[9:], synapse=0.1)
            #########################################################################################################
            #########################################################################################################
            self.inputs = nengo.Probe(external_input[:3], synapse=0.1)
            self.ref_encod = nengo.Probe(Encoded_Reference, synapse=0.1)
            self.ref_encodXY = nengo.Probe(Encoded_ReferenceXY, synapse=0.1)
            self.error_probe = nengo.Probe(Error, synapse=0.1)
            self.SeñalesControl = nengo.Probe(controlEns, synapse=0.1)
            self.gradiente_Kp = nengo.Probe(Grad_kp)
            self.gradiente_Ki = nengo.Probe(Grad_ki)
            self.gradiente_Kd = nengo.Probe(Grad_kd)


    def SeñalInput(self):
        return self.SeñalesControl
    def Gradientes(self):
        return self.gradiente_Kp, self.gradiente_Ki, self.gradiente_Kd
    def get_model(self):
        return self.model
    def get_probe(self):
        return self.inputs, self.ref_encod, self.ref_encodXY, self.error_probe
    


m = Models(Frecuencia)
model = m.get_model
sim = nengo.Simulator(model(), dt=0.0025, optimize=True)

def inicializacion():
        MyTello = Tello()
        MyTello.connect()
        print(MyTello.get_battery())
        MyTello.streamoff()
        MyTello.takeoff()
        return MyTello

        


#################
MyTello = inicializacion()
#################
posicion, ref, refXY, error = m.get_probe()
u_ax = m.SeñalInput()
gradiente_kp, gradiente_ki, gradiente_kd = m.Gradientes()

T = 0 
#··································································································#
Tiempo_final = 20 ###############··································································#
#··································································································#
Time_ini = rospy.get_time()
def Times():
    T_actual = rospy.get_time()
    return T_actual - Time_ini

while not rospy.is_shutdown():
    i = 1
    T= Times()
    while i <= CicleNengo:
        sim.step()
        t = sim.trange()
        i= i+1
    if T> Tiempo_final + Extra:
        flag = 1
    if sim.data[posicion][-1][0]>1.8 or abs(sim.data[posicion][-1][1])>1.3 or abs(sim.data[posicion][-1][2])>1.3:
        MyTello.send_rc_control(0, 0 ,0, 0)#############################################################
        MyTello.land()
        print("")
        print("Fuera de rango")
        print("")
        print(sim.data[posicion][-1][0], sim.data[posicion][-1][1], sim.data[posicion][-1][2])
        break
    if T>1:
        if flag==1:
            MyTello.send_rc_control(0,0,0,0)
            plot_xy(sim.data, num_xticks=6, num_yticks=6, xmin=-1, xmax=31, ymin=-0.11, ymax=0.22)
            MyTello.land()
            # print(MyTello.get_battery())
            break
    rate.sleep()

###############################################################################################################
    ###########################################################################################################
plot_xy(sim.data, num_xticks=6, num_yticks=6, xmin=-1, xmax=31, ymin=-0.11, ymax=0.22)
#Plot3D_XYZ(sim.data)


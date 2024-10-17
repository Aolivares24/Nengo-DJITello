import rospy
import tf
from nav_msgs.msg import Odometry
import csv

import matplotlib.pyplot as plt
import math
import nengo
import numpy as np
import nengo_loihi
from sklearn.preprocessing import MinMaxScaler
from djitellopy import Tello
import logging

# Deshabilitar los mensajes de información (INFO) de djitellopy
logging.getLogger('djitellopy').setLevel(logging.WARNING)
###################################################################################
###################################################################################

rospy.init_node('myNode')
Frecuencia = 20 ##HZ

rate = rospy.Rate(Frecuencia) # ROS Rate at 100Hz
flag = 0

class ExternalInput(nengo.Node):
    def __init__( self, name ):
        self.pose = 0
        #self.MyTello = self.inicializacion() 
        self.dest_sub = rospy.Subscriber(f'/mocap_node/tello_02/Odom', Odometry, self.callback, queue_size=1)
        
        super(ExternalInput, self).__init__(label=name, output= self.tick, size_in=1, size_out=1)
    
    def callback (self, msg):
        # quaternion = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
    
        # euler = tf.transformations.euler_from_quaternion(quaternion)
    
        # yaw_rad = euler[2] - (math.pi/2)
        # if (yaw_rad < -(math.pi)):
        #     yaw_rad += (2 * math.pi)
    
        self.pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]#, yaw_rad]

    def tick(self,t, valores):
        
        u = int(round(valores[0]))
    
        return self.pose[2]
    

class Models (object):
    def __init__(self, frecuancy):
        self.Frecuncy = frecuancy
        self.Kp = 100 ######El 100 %
        self.Kd = 11.25 ############Esta reducido a 3/4
        self.Ki = 2
        self.ZFinal = 1.5
        self.J = 150 ####Numero de neuronas
        self.model = nengo.Network(label='drone')
        
        with self.model:
        
            ################################## ENTRADA DE OPTITRACK #################################################
            external_input = ExternalInput('name') ################# Se crea un nodo
            
            Input_encoder = nengo.Ensemble(n_neurons=self.J,dimensions=1, radius=2)
            nengo.Connection(external_input,Input_encoder)
            ################################### GENERACION Y CODIFICACION DE #########################################
            ################################### SEÑAL DE REFERENCIA ##################################################
            Reference_Signals = nengo.Node(lambda t: self.ZFinal / (1 + math.exp(-(0.7*t-5)))) ######Sigmoide que va aseguir
            Encoded_Reference = nengo.Ensemble(n_neurons=self.J, dimensions=1, radius=2)
            nengo.Connection(Reference_Signals,Encoded_Reference)
            #########################################################################################################
            #########################################################################################################
            Error = nengo.Ensemble(n_neurons=self.J, dimensions=1, radius = 3)
            err_syn = 0.005
            nengo.Connection(Encoded_Reference,Error, synapse = None)
            nengo.Connection(Input_encoder,Error,transform=-1, synapse = None)
            #########################################################################################################
            #####################################(Error_act - Error_pre)/ dt ########################################
            d_e_z = nengo.Ensemble(n_neurons=self.J, dimensions = 1, radius = 2)
            nengo.Connection(Error,d_e_z,synapse=0.05,transform= self.Frecuncy) #################### F = 30 Hz
            nengo.Connection(Error,d_e_z,synapse=0.1,transform= -self.Frecuncy) #####################################
            #########################################################################################################

            Error_integral = nengo.Ensemble(n_neurons=500, dimensions=1, radius=1.5)
            nengo.Connection(Error_integral, Error_integral,transform=1, synapse=0.01)
            nengo.Connection(Error, Error_integral, transform= 0.01, synapse=0.01)

            #########################################################################################################
            #########################################################################################################

            controlEns = nengo.Ensemble(n_neurons= self.J, dimensions=1, radius=100)
            nengo.Connection(Error,controlEns,transform=self.Kp)  ####### Ganancia proporcional
            nengo.Connection(d_e_z,controlEns,transform= self.Kd) ####### Ganacia derivativa
            nengo.Connection(Error_integral,controlEns,transform= self.Ki) ####### Ganancia integral

            ########################################################################################################
            controlSignals = nengo.Node(size_in=1)
            nengo.Connection(controlEns, controlSignals, transform=1)
            nengo.Connection(controlSignals, external_input, synapse=None)

            #########################################################################################################
            #########################################################################################################
            self.inputs = nengo.Probe(external_input, synapse=0.1)
            self.ref_encod = nengo.Probe(Encoded_Reference, synapse=0.1)
            self.error_probe = nengo.Probe(Error, synapse=0.1)
            self.derv_error = nengo.Probe(d_e_z, synapse=0.1)
            self.SeñalesControl = nengo.Probe(controlSignals)
            self.ErrorInt = nengo.Probe(Error_integral)
            #print(self.inputs)


    def SeñalInput(self):
        return self.SeñalesControl
    def get_model(self):
        return self.model

    def get_probe(self):
        return self.inputs, self.ref_encod, self.error_probe, self.SeñalesControl, self.derv_error, self.ErrorInt
    

global t

m = Models(Frecuencia)
model = m.get_model
sim = nengo.Simulator(model(), dt=0.002, optimize=True) #, dt=1/Frecuencia, optimize=False

def inicializacion():
        MyTello = Tello()
        MyTello.connect()
        MyTello.for_back_velocity = 0
        MyTello.left_rigth_velocity = 0
        MyTello.up_down_velocity = 0
        MyTello.speed = 0
        print(MyTello.get_battery())
        MyTello.streamoff()
        MyTello.streamon()
        MyTello.takeoff()
        return MyTello

j = 1
u = []
Inputs = []
Tiempo = []
global Time_ini
MyTello = inicializacion()
Time_ini = rospy.get_time()
def Times():
    T_actual = rospy.get_time()

    resta =  T_actual - Time_ini
    return resta

posicion, ref, error, señalControl, deriv_error, Error_int = m.get_probe()
u_ax = m.SeñalInput()

while not rospy.is_shutdown():
    i = 1
    T= Times()
    while i <= 500/Frecuencia:
        sim.step()
        t = sim.trange()
        u.append(sim.data[u_ax][-1][0])
        i= i+1

    Uz = round( np.sum(u)/ i-1)
    Inputs.append(Uz)
    #print(Uz)
    u = []
    Tiempo.append(j)
    if T>1:
        if flag==1:
            MyTello.send_rc_control(0,0,0,0)
            MyTello.land()
            print(MyTello.get_battery())
            break
    if T> 30 + 2*1/Frecuencia:
        MyTello.send_rc_control(0,0,0,0)
        MyTello.land()
        print(MyTello.get_battery())
        break

    j = j +1
    MyTello.send_rc_control(0 , 0, int(Uz), 0) #######################
    
    
    rate.sleep()




###############################################################################################################
    ###########################################################################################################
def plot_xy(data, num_xticks=5, num_yticks=5, xmin=None, xmax=None, ymin=None, ymax=None):
        
    print("Aqui el tiempo ", len(t), j)
    with open('PosicionZ.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(Inputs)

    plt.figure(figsize=(9, 10))
    plt.plot(t, data[posicion], label="Posicion", linewidth=4, color='blue')
    plt.plot(t, data[ref], label="Referencia $y_{p}$ axis", color='red')
    # plt.plot(t, data[error], label="Error", color='orange')
    # plt.plot(t, data[deriv_error], label="Error", color='green')
    #plt.plot(t, data[error_dec], label="Error_node", color='yellow')
    #plt.plot(t, data[u_ax], label="Señal de control", color='orange')
    plt.xlabel("Time [s]", fontsize=32)
    plt.ylabel("Position $y_{p}$ [m]", fontsize=32)
    plt.grid(True)
    plt.legend(fontsize=32, loc='upper right') 
    
        # Set the number of ticks on the x-axis and y-axis
    plt.locator_params(axis='x', nbins=num_xticks)
    plt.locator_params(axis='y', nbins=num_yticks)
    

    
        # Set the x and y axis limits if provided
    if xmin is not None and xmax is not None:
        plt.xlim(xmin, xmax)
        # if ymin is not None and ymax is not None:
        #     plt.ylim(ymin, ymax)
    
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28) 

    plt.savefig("GraficaZ.pdf")
    
    fig1, ax2 = plt.subplots()
    ax2.plot(t, data[error], label="Error", color='orange')
    ax2.plot(t, data[deriv_error], label="Error derivada", color='green')
    ax2.plot(t, data[Error_int], label="Error integral", color='blue')
    #ax2.plot(t, data[u_ax], label="Señal de control", color='orange')
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Position $y_{p}$ [m]")
    #ax2.set_ylim(0, 2)
    ax2.grid(True, linestyle='-.')
    ax2.legend(fontsize='14')
    #plt.show()
    plt.savefig("Errores.pdf")


    fig1, ax3 = plt.subplots()
    ax3.plot(Tiempo, Inputs,"--", label='U de z')
    #ax2.plot(t, data[u_ax], label="Señal de control", color='orange')
    ax3.set_xlabel('Tiempo')
    ax3.set_ylabel('Input')
    #ax2.set_ylim(0, 2)
    ax3.grid(True, linestyle='-.')
    ax3.legend(fontsize='14')
    #plt.show()
    plt.savefig("GraficaU1.pdf")

    plt.show()
        #file = '/home/omarg/control_quad_ws/src/nengo_examples/CCE 2023/ImagesCCE/' + 'Posy'
        #plt.savefig(file, format="pdf", bbox_inches='tight')   
    
plot_xy(sim.data, num_xticks=6, num_yticks=6, xmin=-1, xmax=31, ymin=-0.11, ymax=0.22)
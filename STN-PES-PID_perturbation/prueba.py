from djitellopy import Tello
from time import sleep

MyTello = Tello()


MyTello.connect()
# MyTello.for_back_velocity = 0
# MyTello.left_rigth_velocity = 0
# MyTello.up_down_velocity = 0
# MyTello.speed = 0
#     print(MyTello.get_battery())
MyTello.streamoff()


print(MyTello.get_battery())
MyTello.takeoff()

sleep(2)
MyTello.for_back_velocity = 0
MyTello.left_rigth_velocity = 0
MyTello.up_down_velocity = 0
MyTello.speed = 0
MyTello.land()
    
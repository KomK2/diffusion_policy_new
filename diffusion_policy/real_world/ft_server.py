
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import WrenchStamped
import numpy as np

latest_ft_data = None

def callback(data):
    global latest_ft_data
    latest_ft_data = [data.wrench.force.x, data.wrench.force.y, data.wrench.force.z,
                      data.wrench.torque.x, data.wrench.torque.y, data.wrench.torque.z]
    # print(latest_ft_data)

def service_server(request):
    global latest_ft_data
    # latest_ft_data+=1
    response = TriggerResponse()

    if latest_ft_data is not None:
        response.success = True
        response.message = str(latest_ft_data)
    else:
        response.success = False
        response.message = "No FT data available yet"

    # latest_ft_data+=1
    return response

def server():
    rospy.init_node('ft_data_service_server')
    rospy.Subscriber('/leptrino_force_torque_base/force_torque', WrenchStamped, callback)
    rospy.Service('get_latest_ft_data', Trigger, service_server)
    rospy.spin()
    # print("service running")

if __name__ == '__main__':
    server()

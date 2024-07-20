import rospy
from std_srvs.srv import Trigger
import time

def client():
    rospy.init_node('ft_data_service_client')
    rospy.wait_for_service('get_latest_ft_data')
    get_latest_ft_data = rospy.ServiceProxy('get_latest_ft_data', Trigger)
    response = get_latest_ft_data()
    
    if response.success:
        print(response.message[1:-1].tolist())
    else:
        rospy.logerr("Failed to get FT data: %s", response.message)

if __name__ == '__main__':
    while True:
        client()
        time.sleep(1)
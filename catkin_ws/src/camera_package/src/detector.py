# The following code is used to watch a video stream, detect Aruco markers, and use
# a set of markers to determine the posture of the camera in relation to the plane
# of markers.
#
# Assumes that all markers are on the same plane, for example on the same piece of paper
#
# Requires camera calibration (see the rest of the project for example calibration)

import numpy
import cv2
import cv2.aruco as aruco
import math
import rospy
from geometry_msgs.msg import Twist 
from geometry_msgs.msg import Pose 

# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)

CAMERA_MATRIX = numpy.array([[920.471743, 0.0, 652.8],[0.0, 927.3, 319.5], [0.0, 0.0, 1.0]])
DISTORTION_MATRIX = numpy.array([[0.06], [-0.1], [-0.006], [0.009], [0.0]])

cmd_vel_pub = rospy.Publisher('/model/pose', Pose, queue_size = 1)  


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = numpy.transpose(R)
    shouldBeIdentity = numpy.dot(Rt, R)
    I = numpy.identity(3, dtype = R.dtype)
    n = numpy.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return numpy.array([x, y, z])


def timer_callback(event):
    # Create vectors we'll be using for rotations and translations for postures
    rvecs, tvecs = None, None

    cam = cv2.VideoCapture(0)
    cam.set(3, 1280)
    cam.set(4, 720)

    while(cam.isOpened()):
        # Capturing each frame of our video stream
        ret, QueryImg = cam.read()
        if ret == True:
            # grayscale image
            gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
        
            # Detect Aruco markers
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
            
            # Make sure all 5 markers were detected before printing them out
            if ids is not None:
                
                # Print corners and ids to the console
                for i, corner in zip(ids, corners):
                    
                    rvecs, tvecs, points = aruco.estimatePoseSingleMarkers(corner, 0.06, CAMERA_MATRIX, DISTORTION_MATRIX)
                    
                    aruco.drawAxis(QueryImg, CAMERA_MATRIX, DISTORTION_MATRIX, rvecs, tvecs, 0.01)
                    rot_mat, jacobian = cv2.Rodrigues(rvecs)
                    print(f"{rot_mat=}\n")
                    euler_angles = rotationMatrixToEulerAngles(rot_mat)

                    x = -tvecs[0][0][1]
                    y = -tvecs[0][0][0]
                    psi = -euler_angles[2]

                    print(f"{rvecs=}\n{tvecs=}\n{euler_angles=}")
                    print(f"\n\n{x=}\n{y=}\n{psi=}")
                    # print('ID: {}; Corners: {}'.format(i, corner))

                    cmd_vel = Pose()
                    cmd_vel.position.x = x
                    cmd_vel.position.y = y
                    cmd_vel.orientation.z= psi
                    cmd_vel_pub.publish(cmd_vel)

                # Outline all of the markers detected in our image
                QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))

            # Display our image
            cv2.imshow('QueryImage', QueryImg)


        # Exit at the end of the video on the 'q' keypress
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    rospy.init_node('model')
    # rospy.Subscriber("/turtle1/cmd_vel", Twist, pose_callback)
    timer = rospy.Timer(rospy.Duration(0.01), timer_callback)
    rospy.spin()

import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'serial_com'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(
        exclude=['test']
    ),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))
        ),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='bme1234',
    maintainer_email='bme1234@todo.todo',
    description='ROS 2 serial communication and chassis control package',
    license='TODO: License declaration',
    tests_require=[
        'pytest',
    ],
    entry_points={
        'console_scripts': [
            'uart_node = serial_com.uart_node:main',
            'cmd_vel_to_rover_control = serial_com.cmd_vel_to_rover_control:main',
            'odom_node = serial_com.odom_node:main',
            'obstacle_avoidance = serial_com.obstacle_avoidance:main',
        ],
    },
)
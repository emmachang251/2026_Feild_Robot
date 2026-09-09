import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'level_a'


setup(
    name=package_name,
    version='0.0.1',

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
            os.path.join(
                'share',
                package_name,
                'launch'
            ),
            glob('launch/*.launch.py')
        ),
        (
            os.path.join(
                'share',
                package_name,
                'config'
            ),
            glob('config/*.yaml')
        ),
    ],

    install_requires=[
        'setuptools',
        'numpy',
    ],

    zip_safe=True,

    maintainer='bme1234',
    maintainer_email='bme1234@example.com',

    description=(
        'ROS 2 autonomous control package '
        'for competition level A'
    ),

    license='Apache-2.0',

    tests_require=[
        'pytest'
    ],

    entry_points={
    'console_scripts': [
        (
            'lane_wall_detector = '
            'level_a.lane_wall_detector:main'
        ),
        (
            'level_a_wall_safety = '
            'level_a.level_a_wall_safety:main'
        ),
        (
            'lane_change_controller = '
            'level_a.lane_change_controller:main'
        ),
        'dataset_capture = level_a.dataset_capture:main',
        'pig_detector = level_a.pig_detector:main',
        'manure_groove_detector = level_a.manure_groove_detector:main',
    ],
},
)
from setuptools import setup

setup(
    name='indoor_explorers',
    version='0.0.1',
    keywords='exploration, robotics, environment, multi-agent, rl, openaigym, openai-gym, gym',
    description='Exploration of unknown indoor areas using lidar',
    install_requires=[
        'gym>=0.9.6',
        'numpy>=1.19.2',
        'pygame>=2.0.0'
    ],
    include_package_data=True,
    py_modules=[]
)

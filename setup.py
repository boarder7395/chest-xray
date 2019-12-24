from setuptools import setup, find_packages
import logging
import subprocess

logging.basicConfig()
logger = logging.getLogger('Chest Xray Install')

def check_nvidia():
    """Utility function to check if GPU is available. Uses the nvidia-smi utility to 
    determine if a GPU is available.
    
    Returns:
        bool: Signifies whether a GPU is available.
    """
    try:
        process = subprocess.Popen(['nvidia-smi', '-L'], stdout=subprocess.PIPE)
    except:
        return False
    output, error = process.communicate()
    if error:
        return False
    logger.info('Installing Tensorflow-GPU...')
    return True

requirements = [
    'numpy==1.16.1',
    'tensorflow-gpu==2.0.0' if check_nvidia() else 'tensorflow==2.0.0',
    'pandas==0.25.3',
    'matplotlib==3.1.2',
    'pyspark==2.4.0',
    'pillow==6.2.1',
    'scipy==1.3.3',
    'opencv-python==4.1.2.30'
]

setup(
    name='chest_xray',
    version='0.0.0',
    python_requires='==3.6.9',
    packages=find_packages(include=['chest_xray',
                                    'chest_xray.*']),
    install_requires=requirements,
    )

import os

project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
version_file = os.path.join(project_folder, 'VERSION')

__version__ = open(version_file).read()

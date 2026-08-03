from setuptools import setup, find_packages

setup(name='ModernBookNLP', 
	version='1.0.0', 
	packages=find_packages(),
	py_modules=['booknlp'],
	include_package_data=True, 
	license="MIT",
install_requires=[
    "torch==2.6.0",
    "transformers==4.57.6",
    "spacy==3.8.14",
    "numpy",
    "pandas",
    "networkx",
    "torch-geometric==2.8.0",
    "huggingface_hub",
    "safetensors",
    "tqdm",
])

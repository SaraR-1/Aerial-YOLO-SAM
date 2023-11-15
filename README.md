# AerialML
This project aims to detect and segment building from satellite and aerial imagery.

Specifically:
- Import satellite or aerial images from the designated data directory.
- Conduct image preprocessing to optimize data quality (enhance contrast).
- Train a YOLOv8 model on xView dataset to detect major types of objects. A preprocessing on xView dataset is necessary to better represent our test data.
- Detect major types of objects such as buildings, cars or airplanes on our test dataset.
- Employ detection techniques to identify the precise building (or other object!) footprints. Here we use Meta's Segment Anything Model (SAM).
- Generate a GeoJSON file containing polygon representations of the detected the object footprints.
- Create new images with the object footprints superimposed for visualization.


## Installation
To use the code in this project, you'll need to create a Python environment, clone the repository and download the data using DVC.
```
conda create -n <envname> python=3.10
conda activate <envname>

pip install poetry
poetry install

git clone https://github.com/SaraR-1/AerialMI.git
dvc pull
```

Before using DVC, you'll first need to setup your AWS credential. To do so, create a file at `~/.aws/credentials`. The content of the file should look like this:
```
[default]
aws_access_key_id=foo
aws_secret_access_key=bar
```
Where:
- aws_access_key_id - The access key for your AWS account.
- aws_secret_access_key - The secret key for your AWS account.

## Run the code
To run the code you can use DVC as:
```
dvc repro
```
This will reproduce the complete (or partial) pipeline by running its stages as needed in the correct order.


# installation in your local pc

1. create a virtual venv
```terminal
python -m venv venv
```
2. activate the virtual environment

3. if you have cuda on your computer use the following command
```terminal
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
4. install the requirements
```terminal
pip install -r requirements.txt
```

# using ipynb file in google colab

## inference
1. upload the inference.ipynb, models_for_inference.zip files from Model_Pytorch/utils folder
2. Run and watch the weather prediction :)

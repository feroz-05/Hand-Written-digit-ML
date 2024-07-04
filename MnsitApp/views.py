from django.shortcuts import render, HttpResponse
from .forms import ImageUploadForm
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras 
# Create your views here.

model = keras.models.load_model("./MnsitModel.h5")

def mnsitModel(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = request.FILES['image']  # Fetch the image file in a variable
            #Model Building           
            image_path = image_file  # Replace with your image file path
            image = Image.open(image_path)
            # Convert the image to grayscale (if necessary)
            image = image.convert('L')  # 'L' mode converts the image to grayscale
            # Convert PIL image to numpy array
            checkImage = np.array(image)
             #checking weather the give Picture is 28 * 28 size
            pixel_array = str(np.array(image))
            if checkImage.shape == (28 , 28 ):
                InputChange = pixel_array.replace("]","")
                newInput = InputChange.replace("[","")
                intIput = [int(x) for x in newInput.split()]
                Input2D = np.array(intIput)
                Input2D = Input2D.reshape((1, 28, 28, 1))
                
                predicted = model.predict(Input2D)
                return render(request, 'success.html',{'pixel': np.argmax(predicted)})
            else:
                return HttpResponse("Image Size is not 28x28")
            #Now you can do something with the image file, e.g., save it or process it
    else:
        form = ImageUploadForm()
        return render(request, 'index.html', {'form': form})
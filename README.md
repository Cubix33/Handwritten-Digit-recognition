## Handwritten-Digit-recognition

# Working Procedure: 
●	Importing libraries such as numpy and pandas.
●	Import datasets containing handwritten images from tensorflow.keras
●	Loading the data using mnist attribute and classifying it into training and testing datasets.
●	Checking the amount of images in the dataset by using shape attribute.
●	Making the width and height of all the images in the dataset uniform and making them all grayscale.
●	Importing training and testing model from sklearn.model_selection.
●	Making the testing set size 10 per cent of the overall data.
●	Feature scaling the datasets to ensure that every data value gets equal weightage. Instead of using OneHotEncoder, we do this step manually.
●	Encoding all the categorical variables using keras from tensorflow.
●	Importing various models such as flatten, dense, sequential etc. from the models attribute of tensorflow.keras.
●	Since we use LeNet architecture in this project, we manually extract all these five layers using code.
●	Checking a summary of the finished model.
●	Minimizing losses in the model using loss function and using optimizer ‘adam.’
●	Fitting the model using the training datasets and 5 epochs.
●	Evaluating the accuracy score the model, (verbose=1).
# Learning Outcomes: 
●	We learnt the utility of libraries such as NumPy and Pandas and learnt how to check the shape of the dataset to get a deeper understanding of its dimensions.
●	Learnt how to process datasets and making the sizes of all the images uniform and converting them to grayscale. 
●	Learnt to implement the crucial step of feature scaling to ensure equal emphasis for every dataset value. 
●	Learnt how to divide the dataset into a training and testing set.
●	Understood the LeNet architecture and also learnt how to manually extract all the 5 layers in this architecture. Also briefly learnt the concept of neural network layers such as flatten, dense and sequential from Tensorflow Keras library.
●	Manually undertook the encoding of categorical variables instead of using shortcuts like OneHotEncoder. Instead we used the Keras library from Tensorflow to do this.
●	Briefly understood the usage and experimented with hyperparameters such as the number of epochs, etc. 
●	Learnt how to minimize training losses in the dataset by using a loss function(eg. Categorical crossentropy) and optimizers such as ‘adam’.


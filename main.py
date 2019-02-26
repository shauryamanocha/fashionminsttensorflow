import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
print("tf: "+tf.__version__)
print("keras: "+keras.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
train_images = train_images/255.0
test_images = test_images/255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images,train_labels,epochs = 5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
scale = 7
right,ttl = 0.0, 49.0
predictions = model.predict(test_images)
plt.figure(figsize=(14,8))
for i in range(scale*scale):
    # print(i)
    print("i predicted a "  +str(class_names[np.argmax(predictions[i])])+",it was actually a: "+str(class_names[test_labels[i]]))
    # print("it was actually a: "+str(class_names[test_labels[i]]))
    if np.argmax(predictions[i]) == test_labels[i]:
        right+=1.0
    plt.subplot(scale,scale,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(left=.1,right=.9,top=1,bottom=.1)
    plt.imshow(test_images[i])
    plt.xlabel("p: "+str(class_names[np.argmax(predictions[i])])+",a: "+ str(class_names[test_labels[i]]))
print("got "+str(right)+" right out of "+str(ttl))
print("sample accuracy of: "+str(100.0*float(right/ttl))+"%")


plt.show()
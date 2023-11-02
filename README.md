# Number-recognition
* 南華大學跨領域-人工智慧 期中報告<br>
* 11124131陳毅倫  11124129吳張傑<br>
# 目錄
* 準備資料<br>
* 實作方法<br>
# 準備資料
* 準備一個可以使用Google Colaboratory的帳號
* 請下載我們提供的:「Writing recognition.ipynb」
* 請下載我們提供的用於置放測試的檔案的資料夾「data_for_test」並上傳到Google Drive
# 實作方法
* 用googlecolab開啟我們提供的Writing recognition.ipynb檔案<br>
※記得要檢查一下 colab 的 Python版本 3.6以下的版本無法運行<br>
* 接下來，逐個運行程式碼，讓機器學習辨識數字<br>
* 導入數據包
```Python
! pip install tensorflow keras numpy mnist matplotlib //Python
```
![image](https://github.com/Yuexiaisnothere/Number-recognition/blob/main/example1.png)
```Python
import numpy as np
import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import keras
import keras.utils
from keras import utils as np_utils //Python
```
* 導入mnist數據集中對應數據
```Python
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels() //Python
```
* 將圖片像量化、訓練神經網路
```Python
train_images = (train_images/255)-0.5
test_images = (test_images/255)-0.5

train_images = train_images.reshape((-7,784))
test_images = test_images.reshape((-7,784))
print(train_images.shape)
print(test_images.shape) //Python
```
* 建立神經網路模型
```Python
model=Sequential()
model.add(Dense(64,activation="relu",input_dim=784))
model.add(Dense(64,activation="relu"))
model.add(Dense(10,activation="softmax"))
print(model.summary()) //Python
```
![image](https://github.com/Yuexiaisnothere/Number-recognition/blob/main/example2.png)
* 神經網路模型編譯及訓練
```Python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
from tensorflow.keras.utils import to_categorical
history=model.fit(train_images,to_categorical(train_labels),batch_size=32,epochs=5)
print(history.history.keys())
#print(plt.plot(history.history,["loss"]))
print(plt.plot(history.history["accuracy"])) //Python
```
![image](https://github.com/Yuexiaisnothere/Number-recognition/blob/main/example3.png)
* 評估
```Python
model.evaluate(
    test_images,to_categorical(test_labels)
) //Python
```
![](https://github.com/Yuexiaisnothere/Number-recognition/blob/main/predictions.png)
* 實行預測
```Python
predictions=model.predict(test_images[:5])
print(np.argmax(predictions,axis=1))
print(test_labels[:5]) //Python
```
![](https://github.com/Yuexiaisnothere/Number-recognition/blob/main/prediction.png)
* 看看在mnist中的圖片<br>
```Python
for i in range(0,5):
  first_image=test_images[i]
  first_image=np.array(first_image,dtype="float")
  pixels=first_image.reshape((28,28))
  plt.imshow(pixels,cmap="gray")
  plt.show() //Python
```
![image](https://github.com/Yuexiaisnothere/Number-recognition/blob/main/example4.png)

手寫數字辨識
----------
* 連結及綁定Google Drive、檢視連結資料夾可用於辨識的圖片檔案目錄
```Python
import os
from google.colab import drive
drive.mount("/content/drive")
path="/content/drive/MyDrive/colab/data_for_test"#要更換成data_for_text的位置
os.chdir(path)
os.listdir(path) //Python
```
![image](https://github.com/Yuexiaisnothere/Number-recognition/blob/main/example5.png)
* 模型預測
```Python
from PIL import Image
import numpy as np
import os //Python
```
```Python
img=Image.open("test-1.jpg").convert("1")#test-1 has other sample, like test-2... to test-5
img=np.resize(img,(28,28,1))
im2arr=np.array(img)
im2arr=im2arr.reshape(1,784)
y_pred=model.predict(im2arr)
print(np.argmax(y_pred,axis=1)) //Python
```
![image](https://github.com/Yuexiaisnothere/Number-recognition/blob/main/example6.png)

# 參考資料
*  [https://blog.csdn.net/weixin_43843172/article/details/109897787](https://blog.csdn.net/weixin_43843172/article/details/109897787)
*  [https://github.com/zhouwenxiaobupt/-/blob/master/handwrittenDigitPrediction.ipynb](https://github.com/zhouwenxiaobupt/-/blob/master/handwrittenDigitPrediction.ipynb)

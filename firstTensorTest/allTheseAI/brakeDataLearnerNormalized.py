import pandas as pd
from pandas.api.types import is_numeric_dtype
from keras import models
from keras import layers
from keras import callbacks
import numpy as np
import matplotlib.pyplot as plt

"""'anomalie', 'failure'"""
pd.DataFrame({'failure': {'false': 0, 'true': 1}})
relevantColumns = ['PedalForce', 'BrakeForce', 'SpeedBefore', 'SpeedAfter', 'Timestamp', 'useageCount',
                   'TimeTillFailure']
train_data = pd.read_csv("../data/OhneExtremeAbnutzung.csv")
test_data = pd.read_csv("../data/shortTestData.csv")
df = pd.DataFrame(train_data)
print("I loaded the data")

for column in train_data:
    if(df.dtypes[column] != 'int64'):
        print("I skipped")
        print(column)
        continue
    mean = np.mean(train_data[column])
    print(test_data[column])
    print("I computed the mean")
    print(mean)
    train_data[column] -= mean
    print("I subtracted the mean")
    std = np.std(train_data[column])
    print("I computed the std")
    print(std)
    train_data[column] /= std
    print("I divided the std")
    test_data[column] -= mean
    test_data[column] /= std

train_targets = train_data.TimeTillFailure
test_targets = test_data.TimeTillFailure

train_data = train_data.drop(["TimeTillFailure"], axis=1)
test_data = test_data.drop(["TimeTillFailure"], axis=1)

train_data['Number'] = [S.replace("brakes", "") for S in train_data['Number']]
test_data['Number'] = [S.replace("brakes", "") for S in test_data['Number']]

print("trainData after standartisation:")
print(train_data)

print("goalField:")
print(train_targets)

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

num_epochs = 14
breakPoint = 630751
partTrainData = train_data[:breakPoint]
partTrainTargets = train_targets[:breakPoint]
partValData = train_data[breakPoint:]
partValTargets = train_targets[breakPoint:]

model = build_model()
cb = callbacks.ModelCheckpoint("../data/learner.h5", monitor='val_mean_absolute_error', save_best_only=False, save_weights_only=False, mode='auto', period=1)
history = model.fit(partTrainData, partTrainTargets, validation_data=(partValData, partValTargets), epochs=num_epochs,
                    callbacks=[cb])
testMSEScore, testMAEScore = model.evaluate(test_data, test_targets)
print(testMAEScore)
print(testMSEScore)
print(history)
print(history.history)
print(history.history['val_mean_absolute_error'])
averageMaeHistory = history.history['val_mean_absolute_error']
plt.plot(range(1, len(averageMaeHistory)+1), averageMaeHistory)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

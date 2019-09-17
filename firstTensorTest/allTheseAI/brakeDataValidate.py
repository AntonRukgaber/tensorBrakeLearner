import pandas as pd
from keras import models
from keras import layers
from keras import callbacks
import numpy as np
import matplotlib.pyplot as plt

pd.DataFrame({'failure': {'false': 0, 'true': 1}})
test_data = pd.read_csv("../data/validation.csv")
"""test_targets = test_data.TimeTillFailure"""
test_data['Number'] = [S.replace("brakesStory", "1") for S in test_data['Number']]
df = pd.DataFrame(test_data)

print(df)
allmeans = [5.564049364030128, 7.663193864587143, 92.4472072075881, 92.44737314953589, 157686.16691434648,
            42284.601416953985, 24732.180688585097]
allstd = [15.075395742815441, 19.972874019378647, 63.4303766785348, 63.43024920927179, 91039.86252588991,
          24673.76417542862, 21335.225921597485]
i = 0

for column in test_data:
    if(df.dtypes[column] != 'int64'):
        print("I skipped")
        print(column)
        continue
    test_data[column] -= allmeans[i]
    test_data[column] /= allstd[i]
    print(allmeans[i])
    print(allstd[i])
    i += 1

print(i)
"""test_data_predict = test_data.drop(["TimeTillFailure"], axis=1)

print(test_data_predict)
"""
model = models.load_model("../data/learnerNormalizedIguess.h5")
results = model.predict(test_data)

resultsNorm = (results*allstd[i])+allmeans[i]
resultsWr = pd.DataFrame(resultsNorm)
resultsWr.to_csv("../data/results.csv")

print(results)
print(resultsNorm)

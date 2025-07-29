import matplotlib.pyplot as mpl
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
data = pd.read_csv("""path here""") # path to dataset

print(data.info())
print("\n","-------------------","\n")
print(data.isnull().sum())


x_multi = data[["TV", "Radio", "Newspaper"]]
y_multi = data["Sales"]
x_train, x_test, y_train, y_test = tts(x_multi, y_multi, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print(f"Score:  {(score*100):.5f}")
y_pred = model.predict(x_test)

plt3 = mpl.scatter(y_test, y_pred)
mpl.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt3.figure.show()


mpl.show()

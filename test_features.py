import numpy as np
import time
gt0 = np.load("gt0.npy")
gt1 = np.load("gt1.npy")
pred0 = np.load("preds0.npy")
pred1 = np.load("pred1.npy")

if gt0.all() == gt1.all():
    print("GTs are equal")

if pred0.all() == pred1.all():
    print("Preds are equal")

# sum elements in pred0
for i in range(len(pred0)):
    if pred0[i] == 0:
        print("hola")

print(len(pred1))

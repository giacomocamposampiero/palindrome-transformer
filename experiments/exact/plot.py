import pandas as pd
import matplotlib.pyplot as plt 

data = pd.read_csv("data/results.csv")
fig = plt.figure()
fig.set_size_inches(8, 5)
plt.plot(range(2, 200), data["valacc"].values, "-", color="k")
plt.xlabel("String length", fontsize=18, labelpad=10)
plt.ylabel("Accuracy", fontsize=18, labelpad=10)
plt.ylim(bottom=-0.1, top=1.1)
plt.tight_layout()
plt.savefig("palindrome.pdf", transparent=True)
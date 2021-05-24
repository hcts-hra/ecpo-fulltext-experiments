import os
import matplotlib.pyplot as plt

for file in ["d299-b1.logs","d299-b2.logs","d299-b4.logs","d299-b8.logs","d299-b14.logs"]:
    with open(file) as f:
        lines = [line.split(",") for line in f.readlines()]
        times = [float(line[-1].split(":")[-1])/3600 for line in lines]
        val_accs = [float(line[-2].split(":")[-1]) for line in lines]
        plt.plot(times, val_accs, label="batch size = "+file.split(".")[0].split("b")[1])
        plt.legend(loc="right")

plt.xlim(0,3.5)
plt.xlabel("training time (h)")
plt.ylabel("accuracy on validation set (%)")
plt.savefig('logs.png', bbox_inches='tight')
plt.show()

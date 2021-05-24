import os
import matplotlib.pyplot as plt

with open("d299-b4.logs") as f:
    lines = [line.split(",") for line in f.readlines()]

epochs = [int(line[0].split(":")[-1]) for line in lines]
train_losses = [float(line[1].split(":")[-1]) for line in lines]
val_losses = [float(line[2].split(":")[-1]) for line in lines]
val_accs = [float(line[-2].split(":")[-1]) for line in lines]

fig, ax1 = plt.subplots()

ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')
ax1.plot(epochs, train_losses, label="training loss", color="royalblue") # label here
ax1.plot(epochs, val_losses, label="validation loss", color="cornflowerblue") # label here
ax1.legend(loc="right", bbox_to_anchor=(1,0.5))

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('validation accuracy') # we already handled the x-label with ax1
ax2.plot(epochs, val_accs, label="validation accuracy", color="tab:red")
ax2.legend(loc="right", bbox_to_anchor=(1,0.4))

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.xlim(0,80)
plt.savefig('b4_logs.png', bbox_inches='tight')
plt.show()

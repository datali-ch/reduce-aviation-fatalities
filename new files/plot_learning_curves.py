# Plots log loss and accuracy 
import matplotlib.pyplot as plt

loss = curr_stats[:,0]
accuracy = curr_stats[:,1]

fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
fig.set_size_inches(30, 15, forward=True)
ax.plot(range(1,len(loss)+1), loss)
plt.ylabel('Log loss')
plt.xlabel('Epochs')
fig.savefig('plots/log_loss.png')  # save the figure to file
plt.close(fig) 

fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
fig.set_size_inches(30, 15, forward=True)
ax.plot(range(1,len(loss)+1), accuracy)
plt.ylabel('Log loss')
plt.xlabel('Epochs')
fig.savefig('plots/accuracy.png')  # save the figure to file
plt.close(fig) 

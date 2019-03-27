# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 22:01:04 2019

@author: surowka

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

"""

# Plots respiration data -> whole dataset needed
subset = train_df.loc[(train_df['crew'] == 8) & (train_df['experiment'] == 'CA') & (train_df['seat'] == 0)]
subset.sort_values(by='time')


from biosppy.signals import ecg, resp
y=subset['r']
out = resp.resp(y,sampling_rate=256, show=False)

fig, ax = plt.subplots( nrows=1, ncols=1 ) 
ax.plot(out['resp_rate_ts'], out['resp_rate'])
fig.set_size_inches(15, 8, forward=True)
plt.ylabel('Respiratory frequency [Hz]')
plt.xlabel('Time [s]');
#mp.savefig('plots/ts_respiration_frequency.png')

# Respiration over time
fig, ax = plt.subplots( nrows=1, ncols=1 ) 
ax.plot(subset['time'], subset['r'], '.')
fig.set_size_inches(15, 8, forward=True)
plt.xlabel('Time [s]');
plt.ylabel('Respiration signal [microvolts]')
#mp.savefig('plots/ts_chest_movement.png')

# Event over time
fig, ax = plt.subplots( nrows=1, ncols=1 ) 
fig.set_size_inches(15, 8, forward=True)
myData = np.array(subset['event'])
myTime = np.array(subset['time'])
ax.plot(myTime, myData, '.')
plt.xlabel('Time [s]');
plt.ylabel('Event')
#mp.savefig('plots/ts_event.png')
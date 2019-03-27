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
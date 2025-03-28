import matplotlib.pyplot as plt

time_initGMM = [(t/50.0) * 1000 for t in [0.136046, 0.423723, 1.105070]]
datasets = ["small", "medium", "large"]

time_init_learn = [(t/50.0) * 1000 for t in [0.083506, 0.328811, 1.343713]]

time_weights = [(t/50.0) * 1000 for t in [0.010456, 0.040998, 0.160449]]

time_graph_seg = [(t/50) * 1000 for t in [1.18193, 6.200943, 33.707372]]

fig, axes = plt.subplots(2, 2, figsize=(6,6))
# First subplot - GMM initialization time
axes[0, 0].plot(datasets, time_initGMM, marker='o', linestyle='-')
axes[0, 0].set_title('GMM Initialization Time (KMeans)')
axes[0, 0].set_xlabel('Dataset')
axes[0, 0].set_ylabel('Average Time / Image (ms)')

# Second subplot - Initialization and Learning time
axes[0, 1].plot(datasets, time_init_learn, marker='o', linestyle='-')
axes[0, 1].set_title('Initialization and Learning Time')
axes[0, 1].set_xlabel('Dataset')
axes[0, 1].set_ylabel('Average Time / Image (ms)')

# Third subplot - Weights update time
axes[1, 0].plot(datasets, time_weights, marker='o', linestyle='-')
axes[1, 0].set_title('Weights Update Time')
axes[1, 0].set_xlabel('Dataset')
axes[1, 0].set_ylabel('Average Time / Image (ms)')

# Fourth subplot - Graph and Segmentation time
axes[1, 1].plot(datasets, time_graph_seg, marker='o', linestyle='-')
axes[1, 1].set_title('Graph and Segmentation Time')
axes[1, 1].set_xlabel('Dataset')
axes[1, 1].set_ylabel('Average Time / Image (ms)')

# Adjust layout
plt.tight_layout()
plt.show()

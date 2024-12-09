import matplotlib.pyplot as plt

# Data
max_seq_lens = [10, 20, 30, 50, 100]
golden_times = [0.9409, 1.7542, 9.6020, 6.7162, 26.0974]
customized_times = [0.4003, 1.0953, 2.0268, 4.2576, 13.0434]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(max_seq_lens, golden_times, marker='o', label="Golden Greedy Decoding Without KV Cache")
plt.plot(max_seq_lens, customized_times, marker='s', label="Customized Greedy Decoding")
plt.title("Time Taken for Different Sequence Lengths", fontsize=14)
plt.xlabel("Max Sequence Length", fontsize=12)
plt.ylabel("Time (seconds)", fontsize=12)
plt.xticks(max_seq_lens)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("result_1_2.png")
# Display
plt.show()

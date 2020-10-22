import numpy as np

from sklearn.preprocessing import StandardScaler

# z-score standardization
# z = (x - 平均值) / 標準差
test_data_array = np.array([60000, 50000])
mean = test_data_array.mean()
std = test_data_array.std()
z_score_normalization = (test_data_array - mean) / std
print(f"mean={mean}, std={std}, z_score={z_score_normalization}")

"""sklearn"""
scale = StandardScaler()


# min-max normalization
# x = ((x - min) / (max - min)) * (new_max - new_min) + new_min

# normalization by decimal scaling

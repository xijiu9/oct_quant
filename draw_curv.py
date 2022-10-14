import matplotlib.pyplot as plt
import numpy as np
x200 = np.arange(200)

# 可以大概理解为一个full-precision的model训练过程
full_precision = [32.96, 41.44, 49.81, 47.88, 57.87, 60.77, 70.47, 58.56, 72.54, 73.05, 71.23, 78.28, 75.08, 74.81, 79.1, 79.79,
     79.32, 81.61, 81.96, 80.18, 82.25, 82.54, 81.84, 83.33, 84.52, 80.29, 83.47, 83.4, 85.51, 83.69, 84.01, 85.51,
     84.84, 81.85, 82.17, 79.91, 85.38, 82.7, 84.58, 84.04, 87.22, 86.48, 87.23, 85.79, 87.95, 85.61, 86.72, 85.68,
     86.08, 88.25, 87.11, 86.93, 85.4, 86.92, 86.42, 87.34, 87.71, 87.14, 88.17, 88.63, 87.29, 87.6, 87.58, 87.63,
     87.31, 89.15, 88.45, 87.55, 86.28, 88.81, 88.02, 87.61, 85.76, 88.9, 87.69, 87.31, 86.93, 88.32, 88.91, 87.6,
     87.82, 88.26, 89.05, 87.35, 87.61, 87.14, 87.14, 88.67, 89.23, 89.06, 88.84, 88.94, 89.16, 88.51, 86.74, 87.49,
     88.42, 88.21, 89.79, 87.5, 88.59, 88.47, 89.42, 88.62, 89.65, 89.32, 89.98, 89.45, 89.39, 90.24, 90.43, 89.91,
     89.22, 90.02, 90.25, 89.22, 89.54, 89.91, 89.64, 89.37, 90.38, 90.11, 90.07, 90.27, 89.47, 89.91, 89.36, 90.54,
     89.9, 89.68, 90.17, 90.69, 90.75, 90.48, 90.4, 91.04, 90.37, 90.85, 90.41, 91.22, 90.15, 90.84, 91.23, 90.8,
     91.17, 91.1, 90.91, 90.83, 91.56, 91.52, 91.64, 91.47, 91.56, 91.53, 91.65, 91.33, 91.71, 91.84, 91.96, 91.85,
     91.3, 91.73, 92.25, 92.01, 92.12, 92.06, 92.14, 92.03, 92.33, 92.34, 92.21, 92.1, 92.46, 92.58, 92.58, 92.5,
     92.22, 92.78, 92.46, 92.84, 92.82, 92.65, 92.84, 92.67, 92.66, 92.67, 92.73, 92.79, 92.72, 92.6, 92.88, 92.59,
     92.94, 92.92, 92.93, 92.64, 92.74, 92.9, 92.83, 92.8]

# 一个从full-precision出发，全4bit的结果，最后能到93.5
all_4bit_full_ckpt_200 = [92.93, 92.07, 90.5, 90.05, 89.57, 87.69, 88.39, 86.91, 86.71, 85.28, 88.44, 84.47, 88.13, 85.73, 86.29, 86.24, 85.45, 85.68, 85.32, 85.05, 85.21, 87.3, 86.06, 86.2, 87.1, 88.34, 82.28, 87.42, 88.53, 88.74, 88.76, 85.55, 80.51, 87.89, 89.13, 86.41, 88.11, 84.3, 87.28, 85.61, 86.12, 87.88, 87.23, 87.93, 86.5, 88.42, 87.53, 89.21, 87.64, 86.9, 88.53, 89.12, 87.26, 86.77, 86.52, 88.52, 87.11, 86.43, 86.73, 88.22, 88.4, 88.67, 86.4, 89.37, 89.21, 89.3, 88.85, 88.49, 88.86, 87.43, 88.77, 89.19, 89.58, 87.94, 88.06, 87.82, 85.38, 89.42, 89.9, 88.93, 89.58, 88.99, 89.36, 89.54, 88.91, 89.65, 89.0, 89.15, 88.67, 89.85, 87.13, 89.06, 90.26, 88.5, 89.7, 89.18, 89.59, 90.1, 90.3, 89.06, 89.57, 90.31, 90.36, 88.83, 90.77, 89.64, 89.99, 90.25, 90.45, 90.26, 90.84, 90.72, 90.08, 90.28, 90.6, 89.14, 89.49, 89.83, 89.93, 90.7, 91.0, 90.01, 90.48, 90.86, 90.77, 89.68, 90.53, 90.79, 90.71, 90.66, 91.08, 90.9, 91.29, 91.0, 90.28, 90.71, 90.59, 91.22, 91.43, 90.87, 91.79, 91.38, 91.88, 91.68, 92.23, 91.82, 91.82, 91.27, 91.32, 91.7, 92.29, 91.36, 92.21, 92.4, 92.12, 91.9, 92.34, 92.28, 92.35, 92.44, 92.36, 92.83, 92.43, 92.56, 92.73, 92.79, 92.7, 92.86, 92.9, 93.07, 92.83, 92.82, 93.01, 93.02, 92.94, 92.88, 92.98, 93.16, 93.33, 93.21, 93.11, 93.19, 93.15, 93.13, 93.18, 93.23, 93.35, 93.09, 93.34, 93.21, 93.53, 93.53, 93.04, 93.26, 93.4, 93.31, 93.47, 93.35, 93.35, 93.1]

# 先训练30 epoch full-precision，再170epoch 纯4bit
all_4bit_full_ckpt_170 = [80.56, 79.51, 80.5, 80.39, 81.32, 83.27, 80.47, 83.04, 83.13, 83.62, 81.18, 84.02, 79.27, 84.99, 78.44, 82.08, 75.57, 85.38, 82.18, 85.68, 81.52, 86.47, 85.8, 85.12, 86.01, 87.76, 84.81, 85.93, 85.87, 86.28, 82.67, 87.53, 85.48, 87.37, 85.1, 85.68, 86.76, 86.46, 83.59, 87.71, 87.44, 85.06, 87.67, 86.23, 86.35, 88.16, 86.55, 88.87, 87.32, 83.83, 87.68, 48.82, 86.27, 86.6, 88.74, 87.33, 87.9, 88.82, 88.23, 86.49, 87.1, 88.8, 86.35, 88.88, 88.96, 88.45, 88.56, 88.68, 88.33, 89.22, 89.03, 89.73, 88.77, 88.74, 87.85, 87.55, 89.45, 87.82, 88.92, 89.39, 89.75, 89.26, 89.41, 89.95, 89.62, 89.27, 89.99, 89.76, 90.28, 89.26, 88.88, 90.14, 89.64, 90.09, 90.22, 90.02, 90.17, 89.52, 89.03, 91.4, 90.05, 90.2, 90.28, 90.66, 90.54, 90.81, 89.48, 90.33, 90.89, 90.69, 90.25, 90.42, 90.85, 91.11, 91.06, 90.97, 90.02, 91.19, 91.49, 90.77, 90.77, 91.34, 91.24, 91.6, 91.52, 91.66, 91.5, 91.77, 91.85, 91.89, 91.86, 91.79, 91.72, 92.15, 92.05, 92.0, 91.7, 92.04, 91.86, 92.25, 92.63, 92.47, 92.26, 92.26, 92.11, 92.44, 92.38, 92.52, 92.32, 92.5, 92.46, 92.46, 92.52, 92.54, 92.51, 92.36, 92.5, 92.59, 92.71, 92.64, 92.6, 92.75, 92.77, 92.6, 92.72, 92.53, 92.75, 92.69, 92.87, 92.75]
x170 = np.arange(30, 200)

all_4bit_full_ckpt_170_clip_001 = [81.27, 77.67, 80.65, 82.69, 82.54, 84.65, 85.12, 85.64, 84.73, 85.36, 85.48, 83.03, 86.58, 86.35, 84.43, 83.79, 85.47, 86.7, 86.2, 80.66, 83.18, 86.62, 86.91, 87.04, 83.61, 86.46, 83.0, 87.97, 88.18, 87.05, 85.58, 86.79, 85.96, 87.58, 87.1, 86.87, 87.31, 88.2, 87.14, 87.77, 86.44, 86.96, 87.35, 87.91, 87.14, 86.97, 88.02, 89.15, 88.16, 87.89, 87.41, 87.77, 88.24, 86.41, 88.12, 87.92, 87.93, 87.43, 88.3, 88.97, 87.83, 89.03, 87.67, 88.27, 88.48, 87.88, 88.53, 87.34, 87.99, 89.95, 88.89, 89.47, 89.29, 89.64, 88.65, 89.5, 90.11, 89.03, 88.89, 88.82, 90.31, 90.29, 88.43, 89.3, 89.89, 89.14, 89.35, 89.06, 88.93, 89.89, 90.45, 90.37, 90.25, 89.39, 88.74, 90.27, 90.68, 90.52, 90.7, 89.81, 90.29, 90.95, 90.86, 90.91, 91.07, 90.89, 90.52, 90.62, 90.93, 90.81, 90.68, 90.85, 91.45, 90.66, 91.29, 91.14, 91.29, 91.44, 91.61, 91.65, 91.75, 91.56, 91.92, 91.95, 91.68, 91.62, 92.1, 91.89, 91.96, 91.49, 92.13, 92.15, 91.68, 91.89, 91.96, 92.26, 92.3, 92.04, 92.3, 92.18, 92.55, 92.27, 92.62, 92.43, 92.3, 92.71, 92.67, 92.77, 92.43, 92.54, 92.71, 92.82, 92.68, 92.87, 92.99, 92.94, 92.67, 92.58, 92.84, 92.75, 93.05, 92.82, 92.86, 92.73, 92.71, 93.08, 92.99, 93.07, 92.85, 93.17]

# 先训练100 epoch full-precision，再100epoch 纯4bit
all_4bit_full_ckpt_100 = [88.13, 86.89, 88.42, 88.8, 88.69, 89.29, 87.33, 88.57, 88.86, 89.77, 88.28, 90.5, 88.44, 89.72, 89.74, 89.84, 90.69, 89.29, 90.22, 88.72, 90.84, 90.45, 90.18, 89.54, 89.72, 90.15, 90.49, 89.06, 89.53, 90.27, 90.42, 90.53, 90.49, 90.6, 89.75, 91.05, 90.59, 90.8, 91.13, 90.37, 90.67, 90.85, 91.17, 90.6, 91.07, 90.95, 90.45, 91.17, 91.26, 91.22, 91.81, 91.67, 91.54,
 91.76, 91.29, 91.1, 91.27, 91.32, 92.06, 91.69, 92.23, 92.13, 92.26, 92.43, 92.06, 92.27, 92.23, 92.42, 92.42, 92.28, 92.38, 92.66, 92.47, 92.17, 92.57, 92.57, 92.72, 92.44, 92.88, 92.5, 92.47, 92.65, 92.52, 92.88, 92.84, 92.51, 92.65, 92.65, 92.73, 92.86, 93.0, 92.73, 92.85, 92.75, 92.76, 92.61, 92.87, 92.67, 92.8, 92.6]
x100 = np.arange(100, 200)

# clip 0.01 all 4bit
all_4bit_clip_001 = [22.95, 34.59, 39.64, 45.5, 50.95, 53.26, 56.13, 59.99, 66.66, 69.38, 69.15, 71.12, 72.18, 70.65, 71.62, 75.24, 75.31, 75.92, 77.12, 75.16, 74.42, 76.86, 79.11, 79.25, 81.2, 74.45, 81.25, 80.64, 82.12, 82.85, 81.18, 82.78, 83.81, 78.43, 82.7, 82.22, 80.93, 80.87, 84.35, 81.39, 84.08, 83.8, 82.82, 84.57, 80.68, 84.18, 85.54, 84.93, 83.28, 86.12, 84.97, 85.88, 85.64, 86.78, 86.76, 85.14, 85.8, 81.15, 86.43, 85.62, 86.0, 85.16, 85.73, 87.47, 87.9, 84.6, 86.07, 86.76, 87.6, 85.27, 87.21, 87.83, 86.25, 87.22, 87.07, 87.25, 87.0, 87.28, 87.0, 86.15, 87.47, 87.47, 87.93, 87.51, 87.1, 86.21, 87.4, 88.18, 88.16, 88.18, 87.43, 88.57, 85.96, 86.72, 88.04, 87.9, 88.11, 86.69, 85.98, 88.06, 89.05, 89.07, 88.48, 88.07, 88.91, 89.48, 88.98, 88.43, 89.21, 88.0, 89.11, 89.5, 88.43, 88.94, 89.13, 88.67, 89.43, 89.68, 89.46, 89.78, 89.29, 89.71, 89.98, 88.89, 89.6, 89.65, 89.89, 90.27, 90.34, 89.51, 89.49, 90.24, 90.38,
89.65, 90.13, 90.23, 89.61, 90.34, 90.7, 90.7, 90.19, 90.09, 89.92, 90.4, 90.43, 91.05, 90.36, 90.47, 91.13, 91.04, 90.51, 90.73, 91.09, 91.33, 90.75, 91.06, 91.23, 91.48, 91.39, 90.9, 91.28, 91.3, 91.4, 91.54, 91.24, 91.64, 91.33, 91.78, 91.43, 91.56, 91.66, 91.51, 91.57, 91.94, 92.02, 91.83, 91.83, 91.69, 91.91, 91.91, 91.72, 91.95, 92.15, 91.91, 92.16, 91.99, 91.65, 91.91, 92.08, 92.08, 92.05, 91.99, 92.1, 92.05, 91.83, 92.22, 91.93, 91.95, 91.93, 92.23]

plt.figure()
plt.locator_params(axis='y', nbins=14)
plt.ylim(80, 94)
# plt.scatter(x200, full_precision, s=1, label='full_precision')
plt.scatter(x170, all_4bit_full_ckpt_170, s=1, label='all_4bit_full_ckpt_170')
# plt.scatter(x100, all_4bit_full_ckpt_100, s=1, label='all_4bit_full_ckpt_100')
# plt.scatter(x200, all_4bit_full_ckpt_200, s=1, 'label'=all_4bit_full_ckpt_200)
plt.scatter(x200, all_4bit_clip_001, s=1, label='all_4bit_clip_001')
plt.scatter(x170, all_4bit_full_ckpt_170_clip_001, s=1, label='all_4bit_full_ckpt_170_clip_001')

plt.legend()
plt.savefig('results/curve.png')
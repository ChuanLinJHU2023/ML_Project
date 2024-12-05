import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Read the CSV file
file_path = 'grid_search.csv'  # Update this path as necessary
data = pd.read_csv(file_path)

# Convert hidden_units from string representation of list to actual tuple, then to string
data['hidden_units'] = data['hidden_units'].apply(lambda x: str(tuple(eval(x))))

# Set the style for seaborn
sns.set(style='whitegrid')

# 1. Line Plot for Test Accuracy based on Number of Hidden Units
plt.figure(figsize=(12, 6))
unique_units = data['hidden_units'].unique()
for units in unique_units:
    subset = data[data['hidden_units'] == units]
    plt.plot(subset['learning_rate'], subset['test_accuracy'], marker='o', label=f'Hidden Units: {units}')
plt.title('Test Accuracy vs Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Test Accuracy')
plt.legend(title='Hidden Units')
plt.grid()
plt.show()

# 2. Heatmap for Number of Hidden Units and Learning Rates
heatmap_data = data.pivot_table(values='test_accuracy', index='hidden_units', columns='learning_rate')
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.3f', cbar=True)
plt.title('Heatmap of Test Accuracy by Hidden Units and Learning Rates')
plt.xlabel('Learning Rate')
plt.ylabel('Hidden Units')
plt.show()

# 3. Boxplot to show distribution of accuracies for each hidden unit configuration
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='hidden_units', y='test_accuracy')
plt.title('Boxplot of Test Accuracy per Configuration of Hidden Units')
plt.xlabel('Hidden Units')
plt.ylabel('Test Accuracy')
plt.xticks(rotation=45)
plt.grid()
plt.show()

# 4. Facet Grid to explore accuracy variations across configurations
g = sns.FacetGrid(data, col='hidden_units', col_wrap=3, height=4, aspect=1)
g.map(sns.scatterplot, 'learning_rate', 'test_accuracy')
g.set_axis_labels('Learning Rate', 'Test Accuracy')
g.set_titles(col_template='Hidden Units: {col_name}')
g.add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Accuracy Variations by Hidden Units')
plt.show()

# 5. 3D Surface Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Prepare grid data
learning_rates = np.sort(data['learning_rate'].unique())
hidden_units_list = np.sort(data['hidden_units'].unique())  # Sort string representations
accuracy_matrix = np.zeros((len(hidden_units_list), len(learning_rates)))

for i, hu in enumerate(hidden_units_list):
    for j, lr in enumerate(learning_rates):
        accuracy_matrix[i, j] = data[(data['hidden_units'] == hu) & (data['learning_rate'] == lr)]['test_accuracy'].mean()

# Create mesh grid
X, Y = np.meshgrid(learning_rates, hidden_units_list)
Z = accuracy_matrix

# Plot surface
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Hidden Units')
ax.set_zlabel('Test Accuracy')
plt.title('3D Surface Plot of Test Accuracy')
plt.show()
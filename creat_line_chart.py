import matplotlib.pyplot as plt

# Step 1: Read data from the text file
with open('Dice_Scores_Memo_differentInputSize.txt', 'r') as file:
    lines = file.readlines()

# Step 2: Convert lines to a list of numbers (y-values)
y_values = [float(line.strip()) for line in lines]

# Step 3: Create x-values as the line numbers (starting from 1)
x_values = ['224*224 (default size)', '0.1 Scale', '0.2 Scale', '0.3 Scale', '0.4 Scale', '0.5 Scale', '0.6 Scale', '0.7 Scale', '0.8 Scale', '0.9 Scale', '1.0 Scale']

# Step 4: Plot the data
plt.plot(x_values, y_values, marker='o', linestyle='-')

# Step 5: Add labels for the axes
plt.xlabel('Input Data Size')  # Label for the x-axis
plt.ylabel('Dice Score')        # Label for the y-axis
# plt.title('Line Chart from Data')  # Add a title (optional)

# Step 6: Show the plot
plt.show()

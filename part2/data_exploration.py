import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('src/housing.csv')

plt.figure()
df['median_house_value'].hist(bins=50)
plt.title('Distribution of Median House Value (More Detailed Bins)')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.show()

plt.savefig('src/median_house_value_distribution.png', dpi=300, bbox_inches='tight')

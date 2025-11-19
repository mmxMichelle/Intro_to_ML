import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('src/housing.csv')

plt.figure()
df['median_house_value'].hist(bins=50)
plt.title('Distribution of Median House Value')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')

plt.savefig('src/median_house_value_distribution.png', dpi=300, bbox_inches='tight')

plt.show()

import pandas as pd
df = pd.read_hdf('data/reviews.h5', key='reviews')
print(df)
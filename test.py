
import polars as pl
   # def returns(self):
    #     stock = self.get_stock()
    #     returns = stock.pct_change().dropna()
    #     return self.returns


df = pl.DataFrame({

    'AAPL': [150, 155, 160, 158],
    'GOOGL': [2800, 2850, 2900, 2950],
    'MSFT': [300, 305, 310, 315]
})

print(df.with_columns(
    pl.col('AAPL').pct_change().alias('AAPL_pct_change'),
    pl.col('GOOGL').pct_change().alias('GOOGL_pct_change'),
    pl.col('MSFT').pct_change().alias('MSFT_pct_change')
))
from constants import ttingo_api_key
from  ttingo_api import retrieve_stock
import DLP

stock = retrieve_stock(['TSLA'], "2023-01-01", "2023-10-01", ttingo_api_key, dataframe = True, save_file = True)


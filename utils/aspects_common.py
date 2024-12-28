# Ben Kabongo
# December 2024

import pandas as pd
from torch.utils.data import Dataset
from typing import Any


class AspectRatingsDataset(Dataset):
    
        def __init__(self, config: Any, data_df: pd.DataFrame):
            super().__init__()
            self.data_df = data_df
            self.config = config
    
        def __len__(self) -> int:
            return len(self.data_df)

        def __getitem__(self, index):
            row = self.data_df.iloc[index]
            user_id = row["user_id"]
            item_id = row["item_id"]
            overall_rating = row["rating"]
            aspects_ratings = [row[aspect] for aspect in self.config.aspects]

            _out = {
                "user_id": user_id,
                "item_id": item_id,
                "overall_rating": overall_rating,
                "aspects_ratings": aspects_ratings
            }
            return _out
        
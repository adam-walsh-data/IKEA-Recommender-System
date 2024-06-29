import pandas as pd
import numpy as np
import random
import pickle

# Sample data (replace this with your session data)
data = {
    'session_id': [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
    'item_id': [101, 102, 103, 201, 202, 203, 204, 301, 302, 303],
    'click': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
}

# Create a DataFrame from the sample data
df_sessions = pd.DataFrame(data)

# Sort sessions by session_id and time_stamp if available
df_sessions = df_sessions.sort_values(by=['session_id'])

# Group sessions by session_id
session_groups = df_sessions.groupby('session_id')

# List to store processed session data
df_groups = []

# Function to create slates for each session
def create_slates(session_group):
    item_id_list = session_group['item_id'].tolist()
    click_list = session_group['click'].tolist()
    length = len(item_id_list)
    slate_id_c, slate_click_c, slate_pos_c = [], [], []
    for i in range(length):
        low = max(i - 19, 0)
        high = min(i, length - 20)
        l = random.randint(low, high)
        r = l + 19
        slate_id_c.append(item_id_list[l:i] + item_id_list[i+1:r+1])
        slate_click_c.append(click_list[l:i] + click_list[i+1:r+1])
        slate_pos_c.append(i - l)
    session_group.insert(session_group.shape[1], 'slate_id', slate_id_c)
    session_group.insert(session_group.shape[1], 'slate_click', slate_click_c)
    session_group.insert(session_group.shape[1], 'slate_pos', slate_pos_c)
    return session_group

# Apply create_slates function to each session group
for session_id, session_group in session_groups:
    processed_group = create_slates(session_group)
    df_groups.append(processed_group)

# Combine processed groups into final DataFrame
res_df = pd.concat(df_groups)

# Save the result as a pickle file
with open('web_sessions_slate_data.pkl', 'wb') as f:
    pickle.dump(res_df, f)

# Show the result (for demonstration)
print(res_df)
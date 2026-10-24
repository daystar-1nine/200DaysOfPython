# Pandas Indexing, Selection, and Memory Semantics

Pandas structures tabular data using DataFrames (2D tables) and Series (1D labelled arrays). Data selection distinguishes between label-based indexing with .loc[] and integer position-based indexing with .iloc[]. Using boolean masks allows intuitive row filtering based on complex conditional logic.

Chained assignment (such as df[mask]['col'] = val) triggers SettingWithCopyWarning because it is ambiguous whether a view or an in-memory copy was returned by the intermediate slice. Explicit assignment using .loc[mask, 'col'] = val guarantees reliable in-place mutation and avoids silent data corruption.

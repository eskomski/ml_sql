# ml_sql

This is a demonstration project meant to illustrate how to incorporate relational databases into an experiment/analysis pipeline for machine learning. The use case presented here is simple hyperparameter tuning with one worker, but this can very easily be extended to other use cases with multiple workers.

Nothing in this repository is meant to be particularly groundbreaking or sophisticated; it uses a very simple schema with no tricks or optimizations. This is simply meant to demonstrate how employing the very basics of databases can yield more efficient and organized experimentation and analysis compared to using flat files. Further, Python provides very simple interfaces for interacting with databases—SQLite3 support is built into the Python standard library, and Pandas provides support for reading relations and query results into DataFrames—so fitting these components into existing code is fairly straightforward.

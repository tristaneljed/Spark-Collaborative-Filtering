# Spark Collaborative Filtering

- The file "server.py" starts a CherryPy server running a Flask "app.py" to start a RESTful web server wrapping a Spark-based "engine.py" context.

# How to use

- Start the server running : ./start_server.sh 
- To add a specefic creative information for a certain user : curl --data-binary @user_ratings.file http://127.0.0.1:5432/0/creatives
- To predict number of successes for a given creative of a certain user : curl -i -X GET http://127.0.0.1:5432/0/creatives/20
- To get a top list of creatives of a certain user : curl -i -X GET http://127.0.0.1:5432/0/creatives/top/3
- To get a list of least-rated creatives of a certain user : curl -i -X GET http://127.0.0.1:5432/0/creatives/last/3

# Dataset

v1. 
- creatives.csv : creativeID, creativeName, 
- creatives_rating.csv : advertiserID, creativeID, Rating, Timestamp (Rating made by many users)
- user_ratings.file : creativeID, Rating (Rating made by me as a user)

# Collaborative Filtering


In Collaborative filtering we make predictions (filtering) about the interests of a user by collecting preferences from many users (collaborating). The underlying assumption is that if a user X has the same opinion as a user Y on an issue, X is more likely to have Y's opinion on a different issue x than to have the opinion on x of a user chosen randomly.

Spark MLlib library for Machine Learning provides a Collaborative Filtering implementation by using Alternating Least Squares. The implementation in MLlib has the following parameters:
- *numBlocks* is the number of blocks used to parallelize computation (set to -1 to auto-configure).
- *rank* is the number of latent factors in the model.
- *iterations* is the number of iterations to run.
- *lambda* specifies the regularization parameter in ALS.
- *implicitPrefs* specifies whether to use the explicit feedback ALS variant or one adapted for implicit feedback data.
- *alpha* is a parameter applicable to the implicit feedback variant of ALS that governs the baseline confidence in preference observations.

# Selecting ALS parameters

- In order to determine the best ALS parameters, we need first to split the dataset into train, validation, and test datasets. Then we can proceed with the training phase. We choose the best rank according to RMSE (root-mean-square error).

# movie-predictor

Predict rating that a user would give to a movie (for example, on Netflix). This can be used to recommend movies to users. 

Input: training data file and testing data file each containing 3 columns (user, movie, rating)

Outut: rmse value and histogram of error from {1,2,3,4,5} using each predictor (baseline_predictor_error.png and improved_predictor_error.png) 

Baseline prediction: The baseline predictor takes the average of all ratings and adds a user bias and movie bias. The user bias and movie bias are chosen such that it minimises the RMSE on the training data. A user may have a bias if he always rates movies higher or lower than the average. A movie could have a bias if it always tends to receive higher or lower ratings than others. 

Improved predictor: The neighbourhood predictor calculates the cosine similarity between a movie-movie pair and considers the movies (neighbours) that are most similar to it. It then adds the weighted sum of the errors from the neighbours to the baseline prediction. 2 movies are similar if they were rated similarly by many users, so a movie would be rated similar to its neighbours that were rated by a common user.

Based on implementation by https://www.netflixprize.com/assets/GrandPrize2009_BPC_BellKor.pdf and explanation by https://github.com/13lheytens/Netflix-Prize

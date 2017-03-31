import os
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import MatrixFactorizationModel


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_counts_and_averages(creativeID_and_nbr_success):
    """Given a tuple (creativeID, nbr_success_iterable)
    return (creativeID, (nbr_success_count, nbr_success_avg))
    """
    nbr_success = len(creativeID_and_nbr_success[1])
    return creativeID_and_nbr_success[0], (nbr_success, float(sum(x for x in creativeID_and_nbr_success[1]))/nbr_success)


class CreativeRecommendationEngine:
    """A creative recommendation engine
    """

    def __count_and_average_success(self):
        """Update the creative's nbr_success from 
        the current data self.nbr_success_RDD
        """
        logger.info("Counting Creative nbr_success...")
        creative_ID_with_nbr_success_RDD = self.nbr_success_RDD.map(lambda x: (x[1], x[2])).groupByKey()
        creative_ID_with_avg_nbr_success_RDD = creative_ID_with_nbr_success_RDD.map(get_counts_and_averages)
        self.creatives_nbr_success_counts_RDD = creative_ID_with_avg_nbr_success_RDD.map(lambda x: (x[0], x[1][0]))
		

    def __train_model(self):
		"""Train the ALS model with the current dataset
		"""
		logger.info("Training the ALS model...")
		self.model = ALS.train(self.nbr_success_RDD, self.rank, seed=self.seed,
				iterations=self.iterations, lambda_=self.regularization_parameter)
		logger.info("ALS model built!")
	
    def __predict_nbr_success(self, user_and_creative_RDD):
		"""Gets predictions for a given (userID, creativeID) formatted RDD
		Returns: an RDD with format (creativeName, creative_Nbr_Success, Nbr_Success)
		"""
		predicted_RDD = self.model.predictAll(user_and_creative_RDD)
		predicted_nbr_success_RDD = predicted_RDD.map(lambda x: (x.product, x.rating))
		predicted_creative_Nbr_Success_creativeName_and_count_RDD = \
			predicted_nbr_success_RDD.join(self.creativeID_RDD).join(self.creatives_nbr_success_counts_RDD)
		predicted_creative_Nbr_Success_creativeName_and_count_RDD = \
			predicted_creative_Nbr_Success_creativeName_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))
		return predicted_creative_Nbr_Success_creativeName_and_count_RDD

    def add_nbr_success(self, nbr_success):
        """Add creative nbr_success in the format (userID, userID, Nbr_Success)
        """
        # Convert nbr_success to an RDD
        new_nbr_success_RDD = self.sc.parallelize(nbr_success)
        # Add new nbr_success to the existing ones
        self.nbr_success_RDD = self.nbr_success_RDD.union(new_nbr_success_RDD)
        # Re-compute creative nbr_success count
        self.__count_and_average_success()
        # Re-train the ALS model with the new nbr_success
        self.__train_model()
        
        return nbr_success
		
    def get_nbr_success_for_creative_ids(self, user_id, creative_ids):
        """Given a user_id and a list of creative_ids, predict nbr_success for them 
        """
        requested_creatives_RDD = self.sc.parallelize(creative_ids).map(lambda x: (user_id, x))
        # Get predicted nbr_success
        ratings = self.__predict_nbr_success(requested_creatives_RDD).collect()

        return ratings
		
    def get_top_nbr_success(self, user_id, creatives_count):
		"""Recommend up to creatives_count top unrated creatives to user_id
		"""
		# Get pairs of (userID, creativeID) for user_id unrated creatives
		user_unrated_creatives_RDD = self.creatives_RDD.filter(lambda nbr_success_data: not nbr_success_data[1]==user_id).map(lambda x: (user_id, x[0]))
		# Get predicted nbr_success
		ratings = self.__predict_nbr_success(user_unrated_creatives_RDD).filter(lambda r: r[2]>=50).takeOrdered(creatives_count, key=lambda x: -x[1])
		return ratings
	
    def get_last_nbr_success(self, user_id, creatives_count):
		"""Recommend up to creatives_count top unrated creatives to user_id
		"""
		# Get pairs of (userID, creativeID) for user_id unrated creatives
		user_unrated_creatives_RDD = self.creatives_RDD.filter(lambda nbr_success_data: not nbr_success_data[1]==user_id).map(lambda x: (user_id, x[0]))
		# Get predicted nbr_success
		ratings = self.__predict_nbr_success(user_unrated_creatives_RDD).filter(lambda r: r[2]<20).takeOrdered(creatives_count, key=lambda x: x[1])
		return ratings
	
    def __init__(self, sc, dataset_path):
			
		logger.info("Starting up the Creative Recommendation Engine:")
		self.sc = sc
		# Load nbr_success data for later use
		logger.info("Loading nbr_success data")
		nbr_success_file_path = os.path.join(dataset_path, 'creatives_rating.csv')
		nbr_success_raw_RDD = self.sc.textFile(nbr_success_file_path)
		nbr_success_raw_data_header = nbr_success_raw_RDD.take(1)[0]
		self.nbr_success_RDD = nbr_success_raw_RDD.filter(lambda line: line!=nbr_success_raw_data_header)\
			.map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()
		# Load creatives data for later use
		logger.info("Loading Creatives data...")
		creatives_file_path = os.path.join(dataset_path, 'creatives.csv')
		creatives_raw_RDD = self.sc.textFile(creatives_file_path)
		creatives_raw_data_header = creatives_raw_RDD.take(1)[0]
		self.creatives_RDD = creatives_raw_RDD.filter(lambda line: line!=creatives_raw_data_header)\
			.map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1])).cache()
		self.creativeName_RDD = self.creatives_RDD.map(lambda x: (int(x[0]),x[1])).cache()
		self.creativeID_RDD = self.creatives_RDD.map(lambda x: (int(x[0]),x[0])).cache()
		# Pre-calculate creatives nbr_success counts
		self.__count_and_average_success()

		# Train the model
		self.rank = 6
		self.seed = 5L
		self.iterations = 10
		self.regularization_parameter = 0.1
		self.__train_model()
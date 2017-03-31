from flask import Blueprint
main = Blueprint('main', __name__)
 
import json
from engine import CreativeRecommendationEngine
 
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
from flask import Flask, request

@main.route("/<int:user_id>/creatives/top/<int:count>", methods=["GET"])
def top_creatives(user_id, count):
    logger.debug("User %s TOP creatives requested", user_id)
    top_creatives = creative_recommendation_engine.get_top_nbr_success(user_id,count)
    return json.dumps(top_creatives)

@main.route("/<int:user_id>/creatives/last/<int:count>", methods=["GET"])
def last_creatives(user_id, count):
    logger.debug("User %s LAST creatives requested", user_id)
    last_creatives = creative_recommendation_engine.get_last_nbr_success(user_id,count)
    return json.dumps(last_creatives)
	
@main.route("/<int:user_id>/creatives/<int:creative_id>", methods=["GET"])
def creative_ratings(user_id, creative_id):
    logger.debug("User %s rating requested for creative %s", user_id, creative_id)
    ratings = creative_recommendation_engine.get_nbr_success_for_creative_ids(user_id, [creative_id])
    return json.dumps(ratings)
	
@main.route("/<int:user_id>/creatives", methods = ["POST"])
def add_creative_ratings(user_id):
    # get the ratings from the Flask POST request object
    ratings_list = request.form.keys()[0].strip().split("\n")
    ratings_list = map(lambda x: x.split(","), ratings_list)
    # create a list with the format required by the engine (user_id, creative_id, nbr_success)
    nbr_success = map(lambda x: (user_id, int(x[0]), float(x[1])), ratings_list)
    # add them to the model using then engine API
    creative_recommendation_engine.add_nbr_success(nbr_success)
	
    return json.dumps(nbr_success)
	
def create_app(spark_context, dataset_path):

		global creative_recommendation_engine
		creative_recommendation_engine = CreativeRecommendationEngine(spark_context, dataset_path)    
		app = Flask(__name__)
		app.register_blueprint(main)
		return app
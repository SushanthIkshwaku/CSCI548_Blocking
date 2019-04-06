import abc

class Blocking(abc.ABC):

	# Any shared data strcutures or methods should be defined as part of the parent class.
	
	# A list of shared arguments should be defined for each of the following methods and replace (or precede) *args.
	
	# The output of each of the following methods should be defined clearly and shared between all methods implemented by members of the group. 
	
	def __init__(self, blocking_module):
		'''
		Initialise object
		Args:
			blocking_module: String that mentions the blocking module being used.
		'''
		self.blocking_module = blocking_module

	@classmethod
	@abc.abstractmethod
	def read_dataset(filepath_list, *args, **kwargs):
		'''
		Accepts datasets as csv file path and returns a List of pandas DataFrame.
		Args:
			filepath_list: List of string path to the dataset
			*args: Additional arguments can be mentioned.

		Returns: List of Pandas DataFrame
		'''
		pass

	@classmethod
	@abc.abstractmethod
	def train(dataframe_list, *args, **kwargs):
		'''
		Accepts list of pandas dataframe to train the model on and returns the trained model
		Args:
			dataframe_list: List of pandas dataframe
			*args: Additional hyper parameters required for training

		Returns: Trained tensorflow model
		'''
		pass

	@classmethod
	@abc.abstractmethod
	def predict(model, dataframe_list, *args, **kwargs):
		'''
		Given a trained model and pandas dataframe of datasets, predict() returns a list of pairs of
		related ids as a list of tuples.
		Args:
			model: tensorflow model used to predict.
			dataframe_list: List of pandas dataframe
			*args: Additional arguments

		Returns: List of pairs of elements related in the dataset. List of tuples.
		'''
		pass

	@classmethod
	@abc.abstractmethod
	def evaluate(groundtruth, related_ids, *args, **kwargs):
		'''
		Given a ground truth path and list of pairs of related entities of the dataset, evaluate() returns the Precision,
		Recall and F1_Score metrics.
		Args:
			groundtruth: String path to the ground truth data
			related_ids: List of tuple which contains pairs of related entities.
			*args: If more than 1 Dataset, string dataset path can be mentioned as additional arguments

		Returns: Precision, Recall, Block_size_reduction
		'''
		pass

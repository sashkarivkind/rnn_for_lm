import ast
import sys
import time
class hyper_params:
	def __init__(self):
		self.SAVE_PATH="../savedStates/"
		self.CORPUS_PATH = "../corpus//"
		self.CORPUS_NAME = "warpeace_input_Karpathy.txt"
		self.THIS_RUN_NAME = sys.argv[0]+'_noname_'+str(int(time.time()))
		self.THIS_RUN_SUFFIX=''
		self.TRAIN_LIM = [0, 0.8]
		self.VALIDATION_LIM = [0.8, 0.9]
		self.N_HIDDEN = 512
		self.LEARNING_RATE = 2e-4
		self.X_sc = 1
		self.BATCH_SIZE = 100
		self.EPSILON = 1e-7
		self.num_steps = 100
		self.using_gru = 0
		self.num_epochs = 100
		self.detailed_layers = None
		self.cont_prev_run=None
		self.cont_iter=None
		self.hidden_layers=3
		self.layer_type=[0]+ ['rnn','gru','rnn']#['gru']*self.hidden_layers
		self.num_layers = self.hidden_layers #legacy redundancy to be discarded in future release
		self.verbose = 1
		self.dropout = {'hh':0.98 , 'ih':0.95}
		self.theta_to_exclude =[]
		self.layer_size = [0]+[self.N_HIDDEN]*self.num_layers

	def parse_from_command_line(self,command_line):
		print('\n\n\ parsing command line \n\n')
		for command in command_line:
			if command.find('=')>-1:
				param_name,param_val = command.split('=')
				print('debug:', param_val)
				param_val = ast.literal_eval(param_val)
				self.__dict__[param_name]=param_val
	def as_dict(self):
		return self.__dict__

	def print_params(self):
		for key in sorted(self.as_dict().keys()):
			print(key,'=',self.as_dict()[key])

 

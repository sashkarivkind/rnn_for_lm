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
		self.LEARNING_RATE = 2e-3
		self.X_sc = 1
		self.BATCH_SIZE = 100
		self.EPSILON = 1e-7
		self.num_steps = 100
		self.using_gru = 0
		self.num_epochs = 100
		self.cont_prev_run=None
		self.cont_iter=None
		self.verbose = 1
		self.dropout = {'hh':0.95 , 'ih':1.0}
		self.theta_to_exclude =[]

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

 

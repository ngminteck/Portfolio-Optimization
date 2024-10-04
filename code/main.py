from directory_manager import *
#from main_data_collection import *
from main_training import *
from main_evaluate import *

make_all_directory()
#main_data_collection()
PCA = True
main_training(PCA)
main_evaluate(PCA)




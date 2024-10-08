from directory_manager import *
#from main_data_collection import *
from main_training import *
from main_evaluate import *

make_all_directory()
#main_data_collection()
PCA = False
improve_current_model_with_6_hours_training = False
main_training(PCA, improve_current_model_with_6_hours_training)
main_evaluate(PCA)




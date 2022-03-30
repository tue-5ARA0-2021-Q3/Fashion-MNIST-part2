""" Module to run src files """
from src import train_eval

def run():
    """
    Function to run the machine learning model
    """
    learner = train_eval.Learner()
    learner.train_model()
    #learner.save_model("")

if __name__ == '__main__':
    run()
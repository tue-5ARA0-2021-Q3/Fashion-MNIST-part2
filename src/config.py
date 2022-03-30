# ====================================================
# CFG
# ====================================================
class CFG:
    """
    Configuration settings for the training model
    """
    epochs = 5
    optimizer = 'adam'
    activation = 'relu'
    debug = True
    model_name = 'model' + '_e' + str(epochs)


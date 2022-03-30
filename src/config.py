# ====================================================
# CFG
# ====================================================
class CFG:
    """
    Configuration settings for the training model
    """
    epochs = 15
    optimizer = 'adam'
    activation = 'relu'
    debug = True
    model_name = 'model' + '_e' + str(epochs)


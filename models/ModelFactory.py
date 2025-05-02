from models.VisualTransformer import VisualTransformer


def CreateModelFromHyperParameters(hyperparameters):
    model = VisualTransformer(
        patch_size = hyperparameters['patch_size'], 
        embedding_size = hyperparameters['embedding_size'], 
        num_classes = hyperparameters['num_classes'],
        num_heads = hyperparameters['num_heads'],
        mlp_dim = hyperparameters['mlp_dim'],
        dropout  = hyperparameters['dropout'])
    return model

def CreateModelFromCheckPoint(checkpoint):
    print("Loading model from checkpoint")
    hyperparameters = checkpoint['hyperparameters']
    model = CreateModelFromHyperParameters(hyperparameters)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print("Model Loaded")
    return model
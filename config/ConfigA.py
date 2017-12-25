# CONFIG FOR ACTOR BRAIN IN ACTOR-CRITIC
configA = {
    "inputs_dim": 8,
    "hidden_dim": 50,
    "outputs_dim": 4,    # action space probability
    "GAMMA": 0.99,
    "learning_rate": 0.00025, #0.001/2,
    "weight_decay": 0.0001,
    "betas": (0.001, 0.9)
    
}

1. python train.py, which would
    Read training data from Input directory
    Train models.
    Save your models to Output 
2. python inference.py, which would
    Read test data from Input
    Load models from Output/Models/Igor/{lang}/*bin , Output/Models/Moiz/*h5 , Input/Ujjwal/Data/step-[2-3]/*h5
    Use models to make predictions on new samples
    Save your predictions to Output/Models/Igor/{lang}/*probs.csv Output/Predictions/Moiz/*csv Input/Ujjwal/Data/*tta.csv
	Blend predictions from all models and save end prediction file to Output/Predictions/submission.csv

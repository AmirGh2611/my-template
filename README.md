template/
├── data/
│ ├── init.py
│ ├── dataset.py # data class
│ └── loader.py # reading data from hard disk and split
├── model/
│ ├── init.py
│ └── custom_model.py # deep learning model
├── training/
│ ├── init.py
│ ├── loop.py # raw pytorch training loop for optuna
│ └── engine.py # ignite trainer/evaluator setup for final
├── optimization/
│ ├── init.py
│ ├── build.py # build_model(trial), build_loaders(trial)
│ └── objective.py # optuna objective with pruning
├── config/
│ └── config.py # const variables
├── utils/
│ ├── init.py
│ └── seed.py # set_seed()
├── outputs/
│ ├── checkpoints/
│ └── optuna_study.db # optuna result
├── experiment.ipynb # run optuna and find the best hyperparameters
└── train.py # final train with best hyperparameters

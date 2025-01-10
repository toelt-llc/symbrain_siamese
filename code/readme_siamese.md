# Folder content 

 - *create_dataset* notebook : tests on dataset import to pytorch and siamese-ready format. Functions then implemented in *utils.py*

 ___
 - *siamese_torch_test.py* : current script used for the model training loop. Several architectures changes ([models.py](code/siamese_torch/siamese_models.py)), data loading and train/eval loop. 
 - *siamese_torch_eval_sizes.py*: Based on current best torch test version. Contains the loop to compare train/eval sizes. Results used in [plots](code/siamese_torch/plots/plots.ipynb)

<!-- Currently need to comment/uncomment models depending on desired pre-trained backbone training,  -->
Improvements of scripts are still planned, this is mostly in development and debug status.  
**usage** : `python3 siamese_torch.py` (requires the right python env)  
Various scripts options are available, to change training hyperparameters and logs.  Training is [wandb](https://wandb.ai/) compatible with the --wanbd arg.
 - **python env :**   
 `conda create --name <envname> python=3.9` and then  
 `pip install -r requirements.txt` (heavy environment, might need some cleaning)

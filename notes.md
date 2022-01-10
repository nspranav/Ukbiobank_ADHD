# Notes for predicting Age of the UkBiobank datset

## Network with 1 FC layer 
- Started with plain network that has 1 FC layer
- The Batch norm was followed after the activation
    ### Results
    1. If the learning rate was high there was the problem of vanishing gradient

## Plain Networks with small learning rate
- Experimented with 3 smaller learning rates with the same 
    ### Results
    1. Worked well with lr = 0.0007 but was oveerfitting. - Job 5261663

## Network with added drop out on the fc layer
- With the smaller learning rate and the dropout on the C layer.
### Results
    1. Looks like underfitting 

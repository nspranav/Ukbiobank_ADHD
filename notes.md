# Notes for predicting Age of the UkBiobank datset

![Age Desc](images/age_description.png) 

# Network with 1 FC layer 
- Started with plain network that has 1 FC layer
- The Batch norm was followed after the activation
    ### <span style = 'color: green'> Results </span>
        1. If the learning rate was high there was the problem of vanishing gradient

# Plain Networks with small learning rate
- Experimented with 3 smaller learning rates with the same 
    
    ### <span style = 'color: green'> Results </span>
    
    > <span style = 'color: black'> Worked well with lr = 0.0007 but was oveerfitting. - Job 5261664. However we have reached the minima at 70th epoch <span>

# Network with added drop out on the fc layer
- With the smaller learning rate and the dropout on the FC layer.
    ## <span style = 'color: green'> Results </span>
        1. Looks like underfitting 

# Reduce the dropout to 10% 

- Job 5263681 with learning rate 0.001 
    ## <span style = 'color: green'> Results </span>
        1. Training error has reduced compared to the 20% dropout   


# Trying model from paper [brain age](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7610710/)

- Did not use their sigmoid layer. Instead used the FC layers for regression.

- Using MAE was able to fit model with error of 3 years [5335292](out5335292.out) line 25


# Model with 2 FC layers
- Did not improve the MAE. ref: [5393704](out5393704.out)


# Try the same model with all the samples
- Was able to reduce the error to 2.5 without overfitting. Ref: [5415909](out5415909.out). Line 40 epoch 37.

# Different learning rates for CONV layers and FC layers

- <b> The minimum error reached before overfitting was 3 epoch 25 of [5416056](out5416056.out) </b>

### 1. Adding dropout to the First conv layer
    - No change from the model without dropout. 
    - Only 12000 samples were used

### 2. Adding droput to only 2nd layer: Job [5418772](out5418772.out)
     - No change from without dropout

### 3. Adding dropout to middle 3 layers: Job [5429162](out5429162.out)

### 4. Adding dropout to all the layers: Job [5429170](out5429170.out)

<br>


# Start transfer learning with the gender
    - Fix the Conv layers and only train the FC layer with sigmoid

## 1. Without transfer learning 
    - The accuracy is 98%

## 2. With transfer learning
<br> 

#### Job [5458898](out5458898.out) run with 20k data : <span style = 'color:red'> Stopped due to timeout </span>
    - Accuracy was improving in both training and testing

<br>

#### Job [5458900](out5458900.out) run with all data : <span style = 'color:red'> Stopped due to timeout </span>
    - Accuracy was improving in both training and testing

<br/>

#### Job [5460432](out5460432.out) distributed learning for faster training : <span style = 'color:green'> Done </span>
    - Max accuracy reached is 0.80

#### Job [5503087](gpu4_5503087.txt) Distributed learning with last first 4 layers fixed : <span style='color:green;font-weight: bold;'> Done </span> 
    - Accuracy of almost 99% percent

<br/>

# Starting transfer learning with the cognitive scores (Number of Digits remembered)

## With transfer learning 

<br>

#### Job [5526067](gpu4_5526067.txt) Distributed learning with all layers fixed except FC layer: <span style='color:green;font-weight: bold;'> Done </span>
    - MSE of 2.3 for train and 2.4 for validation

#### Job [5704714](gpu4_5704714.txt) Distributed learning with 4 CONV layers fixed: <span style="color:green; font-weight: bold"> Done </span> 
    - MSE of 2.2 on training and 2.4 on validation

#### Job [5720439](gpu4_5720439.txt) Distributed learning with 4 CONV layers fixed; Last CV layer has lr=1e-3 and FC layer has lr=1e-2: <span style="color:Green; font-weight: bold"> Done </span> 
    - MSE of 1.4 for train and 1.7 for validation

#### Job [5932653](gpu4_5932653.txt) Using classification with scores of 4 to 9
    - 
Running pyBKT test...

Step 1: Loading fixed data from 'fixed\_data.csv'...  
Loaded dataset with 500 rows  
   user\_id   skill\_name  correct  order\_id  
0        0  basic\_skill        1         0  
1        0  basic\_skill        0         1  
2        0  basic\_skill        1         2  
3        0  basic\_skill        0         3  
4        0  basic\_skill        1         4

Step 2: Training BKT model with modified parameters...

Step 3: Verifying model parameters...  
Learned parameters:  
                              value  
skill       param   class  
basic\_skill prior   default 0.98079  
            learns  default 0.02174  
            guesses default 0.49991  
            slips   default 0.50000  
            forgets default 0.00000  
Could not extract individual parameters: 'param'

Step 4: Making predictions...  
Prediction examples:  
   user\_id   skill\_name  correct  order\_id  correct\_predictions  state\_predictions  
0        0  basic\_skill        1         0              0.50000            0.98079  
1        0  basic\_skill        0         1              0.50000            0.98121  
2        0  basic\_skill        1         2              0.50000            0.98162  
3        0  basic\_skill        0         3              0.50000            0.98202  
4        0  basic\_skill        1         4              0.50000            0.98241

Step 5: Evaluating model performance...  
Updated Prediction Accuracy: 0.5000

Step 6: Visualizing learning curves for multiple students...  
Learning curves plot saved as 'pyBKT\_learning\_curves.png'  

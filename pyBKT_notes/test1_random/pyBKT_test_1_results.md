Running pyBKT test...  
Step 1: Creating synthetic data...  
Created dataset with 500 rows  
   user\_id   skill\_name  correct  order\_id  
0        0  basic\_skill        0         0  
1        0  basic\_skill        1         1  
2        0  basic\_skill        1         2  
3        0  basic\_skill        0         3  
4        0  basic\_skill        1         4

Step 2: Training BKT model...

Step 3: Verifying model parameters...  
Learned parameters:  
                              value  
skill       param   class  
basic\_skill prior   default 0.37134  
            learns  default 0.07744  
            guesses default 0.19618  
            slips   default 0.13581  
            forgets default 0.00000  
Could not extract individual parameters: 'skill'

Step 4: Making predictions...  
Prediction examples:  
   user\_id   skill\_name  correct  order\_id  correct\_predictions  state\_predictions  
0        0  basic\_skill        0         0              0.44424            0.37134  
1        0  basic\_skill        1         1              0.30383            0.16116  
2        0  basic\_skill        1         2              0.53040            0.50032  
3        0  basic\_skill        0         3              0.75029            0.82950  
4        0  basic\_skill        1         4              0.52594            0.49365

Step 5: Evaluating model performance...  
Prediction accuracy: 0.7320

Step 6: Visualizing learning curve...  
Learning curve plot saved as 'pyBKT\_test\_plot.png'

Test completed successfully\!

SUCCESS: pyBKT is correctly installed and functioning\!  

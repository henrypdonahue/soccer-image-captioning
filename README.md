# CS 1470 Assignment 3
## Accuracy
- My model accuracy is 62.02%, this is from my `HW3_CNN` notebook. I put a model for both task 1 and 2 in the notebook to check. Althought task both tasks sometimes get much lower, I am unsure why there is this variation.  
## Comments on code
- There is a bug in my HW3_CNN notebook in the `wrapping up` section. The `history = args.model.fit(` line says `AttributeError: 'NoneType' object has no attribute 'model'`. I could not figure out how to fix this.
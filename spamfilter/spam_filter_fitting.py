import torch
import MInteractionModel
import tools
import pandas as pd
import numpy as np

# Create model of interaction order 3 and read in train set
mod3 = MInteractionModel.MInteractionModel(order=3)
mod3.add_dataset("../data/cleansed_spam_train.csv")

# read in test set
test = pd.read_csv("../data/cleansed_spam_test.csv")

# start optimization with 1000 iterations and regularization coefficient 0.05
mod3.Q = mod3.torch_optimize(1000, a=0.05)

# save tensor to file
tools.save_3Dtens_to_file(mod3.Q, filename="spam3dtens.txt", digits=12)


##### EVALUATION - how many mails of the test set are categorized correctly?
correct, not_correct, true_pos, false_pos, false_neg, true_neg = 0,0,0,0,0,0
for i in range(0,len(test)):
    a = np.array(test)[i].copy()
    b = np.array(test)[i].copy()
    a[-1] = 1
    b[-1] = 0
    
    isspam = float(mod3.funcvalue(a, normalize=False))/(float(mod3.funcvalue(a, normalize=False)) + float(mod3.funcvalue(b, normalize=False)))     
    
    if isspam > 0.5:
        if np.array(test)[i][-1] == a[-1]:
            true_pos += 1
        else:
            false_pos +=1
    
    else:
        if np.array(test)[i][-1] == b[-1]:
            false_neg += 1
        else:
            true_neg +=1            
    
correct = true_pos + false_neg
not_correct = false_pos + true_neg

print("true_pos: ", true_pos)
print("false_neg: ", false_neg)
print("false_pos: ", false_pos)
print("true_neg: ", true_neg)


print("correct: ", correct)
print("not correct: ", not_correct)
print("Score: ", (correct / (correct + not_correct)))

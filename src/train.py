from torch import nn
import torch
from torch import optim
import numpy as np
from model import MatchPredictor

model = MatchPredictor()

model.train()


dataset = np.loadtxt("match_data.csv", delimiter = ",", dtype=np.float32, skiprows = 1)
np.random.shuffle(dataset)
labels = dataset[:, -2:]
data = dataset[:, :-2]
datalength = len(dataset)
train_size = int(0.8 * datalength)


training_data = torch.tensor(data[0:train_size], dtype=torch.float32)
testing_data = torch.tensor(data[train_size : datalength], dtype=torch.float32)
training_label = torch.tensor(labels[0:train_size], dtype=torch.float32)
testing_label = torch.tensor(labels[train_size : datalength], dtype=torch.float32)

batch_size = 10
num_epoch = 6
num_batches = len(training_data) // batch_size

lossfunction = nn.SmoothL1Loss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, weight_decay = 0.0005)


# returns tuple:
# tuple.0 ==> whether the scores were within 10%
# tuple.1 ==> whether the winner was correctly predicted
def compare(result, label):
    print(tuple(result.detach().numpy()))
    score_1_res, score_2_res = tuple(result.detach().numpy())
    score_1_actual, score_2_actual = tuple(label.detach().numpy())
    actual_winner = score_1_res > score_1_actual
    predicted_winner = score_1_res > score_2_res
    print(f"{score_1_res}, {score_2_res} ==> {score_1_actual, score_2_actual}")

    if (score_1_actual * 0.9 <= score_1_res <= score_1_actual * 1.1) and \
       (score_2_actual * 0.9 <= score_2_res <= score_2_actual * 1.1):
        if actual_winner == predicted_winner:
            return (True, True)
        else:
            return (True, False)
        
    else:
        if actual_winner == predicted_winner:
            return (False, True)
        else:
            return (False, False)
        

for epoch in range(num_epoch):
    running_loss = 0.0
    model.train()
    for i in range(num_batches):
        starting_index = i * batch_size
        ending_index = starting_index + batch_size
        if ending_index > len(training_data):
            ending_index = len(training_data)
        
        batch_data = training_data[starting_index:ending_index]
        batch_labels = training_label[starting_index:ending_index]

        result = model(batch_data)
        loss = lossfunction(result, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0 and i != 0:
            print(f"Batch {i}: Loss: {running_loss/100:.4f}")
            running_loss = 0

    print(f"=== Epoch {epoch+1}/{num_epoch} ===")

    model.eval()
    res = model(testing_data)
    correct_win_pred = 0
    correct_thresh_pred = 0

    for row_res, row_label in zip(res, training_label):
        (correct_thresh, correct_win) = compare(row_res, row_label)

        if correct_win:
            correct_win_pred += 1
        
        if correct_thresh:
            correct_thresh_pred += 1
    
    print(f"Win prediction accuracy: {correct_win_pred/len(testing_data):.4f}")
    print(f"10% threshold accuracy: {correct_thresh_pred/len(testing_data):.4f}")



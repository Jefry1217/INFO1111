import torch
import random


data = {
    '0-60' : [],
    'price' : [],
    'range_km' : [],
    'top_speed' : [],
    'model' : []
}
model_to_int = {
    's' : 0,
    'x' : 1,
    '3' : 2,
    'y' : 3 
}

# creating database:
for i in range(5000):
    models = 'sx3y'
    model = random.randint(0, 3)

    model = models[model]

    zero_to_60 = 0
    price = 0
    range_km = 0

    if model == 's':
        zero_to_60 = random.randint(2, 3)
        range_km = random.randint(600, 650)
        price = random.randint(100, 120)
        top_speed = random.randint(240, 260)

    elif model == '3':
        zero_to_60 = random.randint(4, 5)
        range_km = random.randint(500, 549)
        price = random.randint(64, 80)
        top_speed = random.randint(200, 220)

    elif model == 'x':
        zero_to_60 = random.randint(3, 4)
        range_km = random.randint(550, 599)
        price = random.randint(120, 140)
        top_speed = random.randint(220, 240)

    elif model == 'y':
        zero_to_60 = random.randint(5, 6)
        range_km = random.randint(450, 499)
        price = random.randint(80, 100)
        top_speed = random.randint(180, 200)
    
    data['0-60'].append(zero_to_60)
    data['price'].append(price)
    data['range_km'].append(range_km)
    data['model'].append(model_to_int[model])
    data['top_speed'].append(top_speed)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer = torch.nn.Linear(4, 7)
        self.output_layer = torch.nn.Linear(7, 1)

    def forward(self, x):
        hidden = torch.relu(self.hidden_layer(x))
        output = self.output_layer(hidden)
        return output
    
net = Net()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=0.00005)
torch.backends.cudnn.enabled = False

def run_tests(max, above_10, above_15):
    with torch.no_grad():
        correct = 0
        for i in range(20):
            idx = random.randint(0,4999)
            inputs = []
            inputs.append(data['0-60'][idx])
            inputs.append(data['price'][idx])
            inputs.append(data['range_km'][idx])
            inputs.append(data['top_speed'][idx])

            input = torch.tensor(inputs, dtype=torch.float)
            
            expected = data['model'][idx]
            output = net(input).detach().numpy()[0]
            # print(f"Expected: {data['model'][idx]}, Received: {output}")
            if round(output) == expected:
                correct += 1
        print(f'Correct: {correct}/20')
    if correct > 15:
        above_15 += 1
    if correct > 10:
        above_10 += 1
    if correct > max:
        max = correct
    return max, above_10, above_15

max = 0
above_10 = 0
above_15 = 0
for epoch in range(50):

    optimizer.zero_grad()
    running_loss = 0.0
    n = 2000
    for i in range(n):

        inputs = []
        inputs.append(data['0-60'][i])
        inputs.append(data['price'][i])
        inputs.append(data['range_km'][i])
        inputs.append(data['top_speed'][i])

        input = torch.tensor(inputs, dtype=torch.float)
        label = data['model'][i]
        label = torch.tensor([label], dtype=torch.float)

        output = net(input)
        
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch{epoch + 1}, Loss: {running_loss / n}')
    max, above_10, above_15 = run_tests(max, above_10, above_15)

print(f'max: {max}/20, above 10: {above_10}, above 15: {above_15}')
    
    

# Reading the data and pre-processing
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


file_paths = ['./AI COE coding challenge/cc_info.csv', './AI COE coding challenge/transactions.csv']

def read_data():
    """
        Data Reading : Function to read data for Model
    """
    # cc info
    cc = pd.read_csv('./AI COE coding challenge/cc_info.csv')
    # print(cc.head(5))
    # transactions
    transaction = pd.read_csv('./AI COE coding challenge/transactions.csv')
    # print(transaction.head(5))
    # merging both and creating a master dataset
    master = pd.merge(cc, transaction, on='credit_card', how='right')
    # print(master.head(5))

    master.set_index("date", inplace=True)
    master.index = pd.to_datetime(master.index)
    return master.sort_index()

def preprocess_data(df):
    """
        Data pre-processing : Function to preprocess data for Model
    """
    # print(df.columns)
    cols = ['credit_card_limit','transaction_dollar_amount','Long','Lat','zipcode']
    dummy_cols = ['city','state']
    # one_hot_encoding for categorical columns
    one_hot_encoded_cols = pd.get_dummies(df[dummy_cols], prefix='f')
    # concatenating and applyting scaler
    df1 = pd.concat([df[cols],one_hot_encoded_cols], axis=1)
    scaled_data = MinMaxScaler(feature_range = (0, 1))
    data_scaled_ = scaled_data.fit_transform(df1)
    # print(data_scaled_.shape)
    return data_scaled_



# Approach - The AutoEncoderNet will encode the data into a latent-space and decode the feature back to original. 
# My expectation is that the net will learn the features of non-fraud transactions and the input will be similar to output when applied. 
# For fraud, since it is unexpecte, input and output of the model will be significantly different 

class AutoEncoderNet(torch.nn.Module):
    def __init__(self, n_features):
        super(AutoEncoderNet, self).__init__()
        #initial parameters
        self.n_features = n_features
        #encoder
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(in_features=self.n_features, out_features=32),
                        torch.nn.Dropout(p=0.2),
        )
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(in_features=32, out_features=16),
                        torch.nn.BatchNorm1d(16),
        )
        self.layer3 = torch.nn.Sequential(torch.nn.Linear(in_features=16, out_features=8),
                        torch.nn.BatchNorm1d(8),
        )
        #latent space
        self.layer4 = torch.nn.Sequential(torch.nn.Linear(in_features=8, out_features=8))
        self.layer5 = torch.nn.Sequential(torch.nn.Linear(in_features=8, out_features=8))
        #decoder
        self.layer6 = torch.nn.Sequential(torch.nn.Linear(in_features=8, out_features=16),
                        torch.nn.Dropout(p=0.2),
        )
        self.layer7 = torch.nn.Sequential(torch.nn.Linear(in_features=16, out_features=32),
        )
        self.layer8 = torch.nn.Sequential(torch.nn.Linear(in_features=32, out_features=self.n_features)
        )


    def forward(self, x):
        # forward pass
        step1 = torch.nn.functional.leaky_relu(self.layer1(x))
        step2 = torch.nn.functional.leaky_relu(self.layer2(step1))
        step3 = torch.nn.functional.leaky_relu(self.layer3(step2))
        step4 = torch.nn.functional.leaky_relu(self.layer4(step3))
        step5 = torch.nn.functional.leaky_relu(self.layer5(step4))
        step6 = torch.nn.functional.leaky_relu(self.layer6(step5))
        step7 = torch.nn.functional.leaky_relu(self.layer7(step6))
        x = torch.sigmoid(self.layer8(step7))
        return x

def testNet():
    #testing the defined architechture
    data_shape = (4, 164)
    test_net = AutoEncoderNet(data_shape[1])
    #random sample
    sample = torch.rand(data_shape)
    test_net.eval()
    a = test_net.forward(sample)
    print(a.shape) #logging

if __name__ == '__main__':
    # step 1 
    df = read_data()
    df = preprocess_data(df)
    n_features = df.shape[1]
    batch_size = 32

    # testNet()
    #train & test-split
    train_data, test_data = train_test_split(df, test_size=0.2)

    data = torch.Tensor(train_data)
    # creating dataset
    train_dataset = TensorDataset(data) 
    # creating dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    
    #testing dataloader
    for batch in iter(train_dataloader):
        print(batch[0].shape)
        break

    data = torch.Tensor(test_data)
    # creating dataset
    test_dataset = TensorDataset(data) 
    # creating dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    
    #initalizing the network
    net = AutoEncoderNet(n_features)
    #defining loss
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train=False
    if train:
        #training the network
        print("Training Started")
        for epoch in range(1):

            running_loss = 0.0
            for i, data in enumerate(train_dataloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data[0]
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('Epoch > %d, Batch Passed : %5d, loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

    #saving the model
    PATH = './fraud_autoencoder.pth'
    torch.save(net.state_dict(), PATH)

    net.load_state_dict(torch.load(PATH))
    for batch in iter(test_dataloader):
        outputs = net(batch[0])

    #finding the threshold on the train_dataset
    total_mse = []
    with torch.no_grad():
        for data in train_dataloader:
            inputs = data[0]
            outputs = net(inputs)
            mse = criterion(outputs, inputs)
            total_mse.append(mse.sum().item())
    cut_off = np.percentile(total_mse, 95)

    # any transaction with mse higher than this cut-off will be sent for fraud-investigation
    # model evaluation on samples from test-dataset
    total_mse = []
    with torch.no_grad():
        for data in test_dataloader:
            inputs = data[0]
            outputs = net(inputs)
            mse = criterion(outputs, inputs)
            total_mse.append(mse.item())
    
    print(total_mse)
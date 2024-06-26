import torch
import torch.nn as nn
import torch.nn.functional as F

# Construct a model with one layer
class Model_MLP(nn.Module):
    
    def __init__(self, layer_size_list, non_linearity, initial_weights, type = 'regression'):
        super().__init__()
        input_size = layer_size_list[0]
        output_size = layer_size_list[-1]
        hidden_size_list = layer_size_list[1:-1]
        self.num_layers = len(hidden_size_list) + 1
        self.type = type
        if non_linearity == 'relu':
            self.non_linearity = torch.relu
        elif non_linearity == 'sigmoid':
            self.non_linearity = torch.sigmoid
        elif non_linearity == 'tanh':
            self.non_linearity = torch.tanh
        elif callable(non_linearity):
            self.non_linearity = non_linearity
        else:
            self.non_linearity = None
        self.initial_weights = initial_weights
        
        if self.num_layers == 1:
            self.l1 = nn.Linear(input_size, output_size)
        else:
            self.l1 = nn.Linear(input_size, hidden_size_list[0])
            for i in range(self.num_layers-2):
                setattr(self, 'l{}'.format(i+2), nn.Linear(hidden_size_list[i], hidden_size_list[i+1]))
            setattr(self, 'l{}'.format(self.num_layers), nn.Linear(hidden_size_list[-1], output_size))

        #initialize weights and biases
        for i in range(self.num_layers):
            if initial_weights == 'normal':
                getattr(self, 'l{}'.format(i+1)).weight.data.normal_(0.0, 1)
            elif initial_weights == 'xavier':
                nn.init.xavier_normal_(getattr(self, 'l{}'.format(i+1)).weight)
            elif initial_weights == 'kaiming':
                nn.init.kaiming_normal_(getattr(self, 'l{}'.format(i+1)).weight,nonlinearity=non_linearity)
            elif initial_weights == 'zero':
                getattr(self, 'l{}'.format(i+1)).weight.data.fill_(0.0)
            elif initial_weights == 'one':
                getattr(self, 'l{}'.format(i+1)).weight.data.fill_(1.0)
            elif initial_weights == 'uniform':
                getattr(self, 'l{}'.format(i+1)).weight.data.uniform_(-1.0, 1.0)
            elif initial_weights == 'xavier_uniform':
                nn.init.xavier_uniform_(getattr(self, 'l{}'.format(i+1)).weight)
            elif initial_weights == 'kaiming_uniform':
                nn.init.kaiming_uniform_(getattr(self, 'l{}'.format(i+1)).weight,nonlinearity=non_linearity)
            getattr(self, 'l{}'.format(i+1)).bias.data.fill_(0.0)
        
    def forward(self, inputs):
        if self.num_layers == 1:
            outputs = getattr(self, 'l1')(inputs)
        else:
            outputs = getattr(self, 'l1')(inputs)
            if self.non_linearity is not None:
                outputs = self.non_linearity(outputs)
            for i in range(self.num_layers-2):
                outputs = getattr(self, 'l{}'.format(i+2))(outputs)
                if self.non_linearity is not None:
                    outputs = self.non_linearity(outputs)
            outputs = getattr(self, 'l{}'.format(self.num_layers))(outputs)

        if self.type == "classification":
            outputs = nn.LogSoftmax(dim=1)(outputs)

        return outputs

# Construct a model with one layer
class Quadratic_MLP(nn.Module):
    
    def __init__(self, layer_size_list, non_linearity, initial_weights, type = 'regression'):
        super().__init__()
        self.input_size = layer_size_list[0]
        self.output_size = layer_size_list[-1]
        self.hidden_size_list = layer_size_list[1:-1]
        self.num_layers = len(self.hidden_size_list) + 1
        self.type = type
        if non_linearity == 'relu':
            self.non_linearity = torch.relu
        elif non_linearity == 'sigmoid':
            self.non_linearity = torch.sigmoid
        elif non_linearity == 'tanh':
            self.non_linearity = torch.tanh
        elif callable(non_linearity):
            self.non_linearity = non_linearity
        else:
            self.non_linearity = None
        self.initial_weights = initial_weights
        
        if self.num_layers == 1:
            self.q1 = nn.Parameter(torch.zeros(self.input_size, self.output_size * self.input_size))
            self.l1 = nn.Linear(self.input_size, self.output_size)
        else:
            self.q1 = nn.Parameter(torch.zeros(self.input_size, self.hidden_size_list[0] * self.input_size))
            self.l1 = nn.Linear(self.input_size, self.hidden_size_list[0])
            for i in range(self.num_layers-2):
                setattr(self, 'l{}'.format(i+2), nn.Linear(self.hidden_size_list[i], self.hidden_size_list[i+1]))
            setattr(self, 'l{}'.format(self.num_layers), nn.Linear(self.hidden_size_list[-1], self.output_size))

        #initialize weights and biases
        if initial_weights == 'normal':
            getattr(self, 'q{}'.format(1)).data.normal_(0.0, 1)
        elif initial_weights == 'xavier':
            nn.init.xavier_normal_(getattr(self, 'q{}'.format(1)))
        elif initial_weights == 'kaiming':
            nn.init.kaiming_normal_(getattr(self, 'q{}'.format(1)),nonlinearity=non_linearity)
        elif initial_weights == 'zero':
            getattr(self, 'q{}'.format(1)).data.fill_(0.0)
        elif initial_weights == 'one':
            getattr(self, 'q{}'.format(1)).data.fill_(1.0)
        elif initial_weights == 'uniform':
            getattr(self, 'q{}'.format(1)).data.uniform_(-1.0, 1.0)
        elif initial_weights == 'xavier_uniform':
            nn.init.xavier_uniform_(getattr(self, 'q{}'.format(1)))
        elif initial_weights == 'kaiming_uniform':
            nn.init.kaiming_uniform_(getattr(self, 'q{}'.format(1)),nonlinearity=non_linearity)

        for i in range(1,self.num_layers):
            if initial_weights == 'normal':
                getattr(self, 'l{}'.format(i)).weight.data.normal_(0.0, 1)
            elif initial_weights == 'xavier':
                nn.init.xavier_normal_(getattr(self, 'l{}'.format(i)).weight)
            elif initial_weights == 'kaiming':
                nn.init.kaiming_normal_(getattr(self, 'l{}'.format(i)).weight,nonlinearity=non_linearity)
            elif initial_weights == 'zero':
                getattr(self, 'l{}'.format(i)).weight.data.fill_(0.0)
            elif initial_weights == 'one':
                getattr(self, 'l{}'.format(i)).weight.data.fill_(1.0)
            elif initial_weights == 'uniform':
                getattr(self, 'l{}'.format(i)).weight.data.uniform_(-1.0, 1.0)
            elif initial_weights == 'xavier_uniform':
                nn.init.xavier_uniform_(getattr(self, 'l{}'.format(i)).weight)
            elif initial_weights == 'kaiming_uniform':
                nn.init.kaiming_uniform_(getattr(self, 'l{}'.format(i)).weight,nonlinearity=non_linearity)
            getattr(self, 'l{}'.format(i)).bias.data.fill_(0.0)
        
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        if self.num_layers == 1:
            input_dim = inputs.shape[1]
            qx = torch.matmul(inputs, self.q1).reshape((batch_size, input_dim, self.output_size))
            qxx = torch.bmm(inputs.unsqueeze(-2),qx).squeeze(-2)
            linear_part = getattr(self, 'l{}'.format(1))(inputs)
            outputs = qxx + linear_part
        else:
            input_dim = inputs.shape[1]
            qx = torch.matmul(inputs, self.q1).reshape((batch_size, input_dim, self.hidden_size_list[0]))
            qxx = torch.bmm(inputs.unsqueeze(-2),qx).squeeze(-2)
            linear_part = getattr(self, 'l{}'.format(1))(inputs)
            outputs = qxx + linear_part
            if self.non_linearity is not None:
                outputs = self.non_linearity(outputs)
            for i in range(self.num_layers-2):
                outputs = getattr(self, 'l{}'.format(i+2))(outputs)
                if self.non_linearity is not None:
                    outputs = self.non_linearity(outputs)
            outputs = getattr(self, 'l{}'.format(self.num_layers))(outputs)

        if self.type == "classification":
            outputs = nn.LogSoftmax(dim=1)(outputs)

        return outputs
    
# Construct a model with one layer
class Polynomial_MLP(nn.Module):
    
    def __init__(self, degree, layer_size_list, non_linearity, initial_weights, type = 'regression'):
        super().__init__()
        self.list_terms = ['linear','quadratic','cubic','quartic','quintic','sextic','septic','octic','nonic','decic']
        self.degree = degree
        self.input_size = layer_size_list[0]
        self.output_size = layer_size_list[-1]
        self.hidden_size_list = layer_size_list[1:-1]
        self.num_layers = len(self.hidden_size_list) + 1
        self.type = type
        if non_linearity == 'relu':
            self.non_linearity = torch.relu
        elif non_linearity == 'sigmoid':
            self.non_linearity = torch.sigmoid
        elif non_linearity == 'tanh':
            self.non_linearity = torch.tanh
        elif callable(non_linearity):
            self.non_linearity = non_linearity
        else:
            self.non_linearity = None
        self.initial_weights = initial_weights
        
        if self.num_layers == 1:
            setattr(self, self.list_terms[0] + '{}'.format(1), nn.Linear(self.input_size, self.output_size))
            for d in range(2,self.degree+1):
                setattr(self, self.list_terms[d]+'{}'.format(1), nn.Parameter(torch.zeros(self.input_size,  self.output_size*self.input_size**(d-1))))

        else:
            setattr(self, self.list_terms[0] + '{}'.format(1), nn.Linear(self.input_size, self.hidden_size_list[0]))
            print(self.list_terms[0]+'{}'.format(1))
            for d in range(2,self.degree+1):
                print(self.list_terms[d-1]+'{}'.format(1))
                setattr(self, self.list_terms[d-1]+'{}'.format(1), nn.Parameter(torch.zeros(self.input_size,  self.hidden_size_list[0]*self.input_size**(d-1))))
            for i in range(1,self.num_layers-1):
                print( self.list_terms[0] + '{}'.format(i+1))
                setattr(self, self.list_terms[0] + '{}'.format(i+1), nn.Linear(self.hidden_size_list[i-1], self.hidden_size_list[i]))
            print(self.list_terms[0]+'{}'.format(self.num_layers))
            setattr(self, self.list_terms[0] + '{}'.format(self.num_layers), nn.Linear(self.hidden_size_list[-1], self.output_size))

        #initialize weights and biases
        if initial_weights == 'normal':
            for d in range(2,self.degree+1):
                getattr(self, self.list_terms[d-1]+'{}'.format(1)).data.normal_(0.0, 1)
        elif initial_weights == 'xavier':
            for d in range(2,self.degree+1):
                nn.init.xavier_normal_(getattr(self, self.list_terms[d-1]+'{}'.format(1)))
        elif initial_weights == 'kaiming':
            for d in range(2,self.degree+1):
                nn.init.kaiming_normal_(getattr(self, self.list_terms[d-1]+'{}'.format(1)),nonlinearity=non_linearity)
        elif initial_weights == 'zero':
            for d in range(2,self.degree+1):
                getattr(self, self.list_terms[d-1]+'{}'.format(1)).data.fill_(0.0)
        elif initial_weights == 'one':
            for d in range(2,self.degree+1):
                getattr(self, self.list_terms[d-1]+'{}'.format(1)).data.fill_(1.0)
        elif initial_weights == 'uniform':
            for d in range(2,self.degree+1):
                getattr(self, self.list_terms[d-1]+'{}'.format(1)).data.uniform_(-1.0, 1.0)
        elif initial_weights == 'xavier_uniform':
            for d in range(2,self.degree+1):
                nn.init.xavier_uniform_(getattr(self, self.list_terms[d-1]+'{}'.format(1)))
        elif initial_weights == 'kaiming_uniform':
            for d in range(2,self.degree+1):
                nn.init.kaiming_uniform_(getattr(self, self.list_terms[d-1]+'{}'.format(1)),nonlinearity=non_linearity)

        for i in range(1,self.num_layers):
            if initial_weights == 'normal':
                getattr(self, self.list_terms[0] + '{}'.format(i)).weight.data.normal_(0.0, 1)
            elif initial_weights == 'xavier':
                nn.init.xavier_normal_(getattr(self, self.list_terms[0] + '{}'.format(i)).weight)
            elif initial_weights == 'kaiming':
                nn.init.kaiming_normal_(getattr(self, self.list_terms[0] + '{}'.format(i)).weight,nonlinearity=non_linearity)
            elif initial_weights == 'zero':
                getattr(self, self.list_terms[0] + '{}'.format(i)).weight.data.fill_(0.0)
            elif initial_weights == 'one':
                getattr(self, self.list_terms[0] + '{}'.format(i)).weight.data.fill_(1.0)
            elif initial_weights == 'uniform':
                getattr(self, self.list_terms[0] + '{}'.format(i)).weight.data.uniform_(-1.0, 1.0)
            elif initial_weights == 'xavier_uniform':
                nn.init.xavier_uniform_(getattr(self, self.list_terms[0] + '{}'.format(i)).weight)
            elif initial_weights == 'kaiming_uniform':
                nn.init.kaiming_uniform_(getattr(self, self.list_terms[0] + '{}'.format(i)).weight,nonlinearity=non_linearity)
            getattr(self, self.list_terms[0] + '{}'.format(i)).bias.data.fill_(0.0)
        
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        if self.num_layers == 1:
            input_dim = inputs.shape[1]
            outputs = getattr(self, self.list_terms[0]+'{}'.format(1))(inputs) #linear part 
            x = inputs.unsqueeze(-2)
            for d in range(2,self.degree+1):
                q = torch.matmul(inputs, getattr(self, self.list_terms[d-1]+'{}'.format(1))).reshape((batch_size, input_dim, self.output_size*self.input_size**(d-2)))
                for i in range(3,d+1):
                    q = torch.bmm(x,q).reshape((batch_size, input_dim, self.output_size*self.input_size**(d-i)))
                q = torch.bmm(x,q).squeeze(-2)
                outputs += q
        else:
            input_dim = inputs.shape[1]
            outputs = getattr(self, self.list_terms[0]+'{}'.format(1))(inputs) #linear part 
            x = inputs.unsqueeze(-2) 
            for d in range(2,self.degree+1):
                q = torch.matmul(inputs, getattr(self, self.list_terms[d-1]+'{}'.format(1))).reshape((batch_size, input_dim, self.hidden_size_list[0]*self.input_size**(d-2)))
                for i in range(3,d+1):
                    q = torch.bmm(x,q).reshape((batch_size, input_dim, self.hidden_size_list[0]*self.input_size**(d-i)))
                q = torch.bmm(x,q).squeeze(-2)
                outputs += q
            if self.non_linearity is not None:
                outputs = self.non_linearity(outputs)
            for i in range(self.num_layers-2):
                outputs = getattr(self, self.list_terms[0]+'{}'.format(i+2))(outputs)
                if self.non_linearity is not None:
                    outputs = self.non_linearity(outputs)
            outputs = getattr(self, self.list_terms[0]+'{}'.format(self.num_layers))(outputs)

        if self.type == "classification":
            outputs = nn.LogSoftmax(dim=1)(outputs)

        return outputs

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc_1 = nn.Linear(320, 50)
        self.fc_2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc_1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc_2(x)
        return F.log_softmax(x, dim=1)

class PolynomialConvNet(nn.Module):
    def __init__(self,fc_model):
        super(PolynomialConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc_model = fc_model

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.fc_model(x)
        return x

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc_1 = nn.Linear(320, 50)
        self.fc_2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc_1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc_2(x)
        return F.log_softmax(x, dim=1)

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.T = 653
        
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints. 
        self.fc1 = nn.Linear(4*2*7, 1)
        

    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        print(x.shape)
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)
        print(x.shape)
        
        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        
        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)
        
        # FC Layer
        x = x.view(-1, 4*2*7)
        x = F.sigmoid(self.fc1(x))
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    
    
class Corr_MLP(nn.Module):
    
    def __init__(self, n_channels, n_filters, nperseg, noverlap, seq_length, layer_size_list, F1, F2, pool_size, non_linearity, initialization):
        super().__init__()

        # Compute the number of layers and their dimensions
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.seq_length = seq_length
        self.output_size = layer_size_list[-1]
        self.layer_size_list = layer_size_list
        self.layer_size_list.insert(0, n_filters*n_channels**2)
        self.num_layers = len(self.layer_size_list) - 1

        # Define the non-linearity function
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
        
        # Define the layers
        self.conv_block = nn.Sequential(
            nn.Conv2d(1,F1,(1,int(seq_length/2+1)),stride=1,padding = 'same',bias=False),
            nn.BatchNorm2d(F1, False),
            nn.Conv2d(F1,F2,(n_channels,1),padding = 0,groups=F1),
            nn.BatchNorm2d(F2,False),
            nn.ELU(),
            #nn.AvgPool2d((1,8)),
            nn.Dropout2d(0.1),
            nn.Conv2d(F2,F2,(1,16),bias=False,padding='same'),
            nn.BatchNorm2d(F2,False),
            nn.ELU(),
            #nn.AvgPool2d((1,pool_size)),
            nn.Dropout2d(0.1)
        )

        self.q1 = nn.Parameter(torch.zeros(self.n_filters,F2)) #self.nperseg))
        for i in range(self.num_layers):
            setattr(self, 'l{}'.format(i+1), nn.Linear(self.layer_size_list[i], self.layer_size_list[i+1]))

        #initialize weights and biases
        self.initialize_weights(initialization,non_linearity)

    def initialize_weights(self,initialization, non_linearity):
        #initialize weights and biases
        if initialization == 'normal':
            getattr(self, 'q{}'.format(1)).data.normal_(0.0, 1)
        elif initialization == 'xavier':
            nn.init.xavier_normal_(getattr(self, 'q{}'.format(1)))
        elif initialization == 'kaiming':
            nn.init.kaiming_normal_(getattr(self, 'q{}'.format(1)),nonlinearity=non_linearity)
        elif initialization == 'zero':
            getattr(self, 'q{}'.format(1)).data.fill_(0.0)
        elif initialization == 'one':
            getattr(self, 'q{}'.format(1)).data.fill_(1.0)
        elif initialization == 'uniform':
            getattr(self, 'q{}'.format(1)).data.uniform_(-1.0, 1.0)
        elif initialization == 'xavier_uniform':
            nn.init.xavier_uniform_(getattr(self, 'q{}'.format(1)))
        elif initialization == 'kaiming_uniform':
            nn.init.kaiming_uniform_(getattr(self, 'q{}'.format(1)),nonlinearity=non_linearity)

        for i in range(1,self.num_layers):
            if initialization == 'normal':
                getattr(self, 'l{}'.format(i)).weight.data.normal_(0.0, 1)
            elif initialization == 'xavier':
                nn.init.xavier_normal_(getattr(self, 'l{}'.format(i)).weight)
            elif initialization == 'kaiming':
                nn.init.kaiming_normal_(getattr(self, 'l{}'.format(i)).weight,nonlinearity=non_linearity)
            elif initialization == 'zero':
                getattr(self, 'l{}'.format(i)).weight.data.fill_(0.0)
            elif initialization == 'one':
                getattr(self, 'l{}'.format(i)).weight.data.fill_(1.0)
            elif initialization == 'uniform':
                getattr(self, 'l{}'.format(i)).weight.data.uniform_(-1.0, 1.0)
            elif initialization == 'xavier_uniform':
                nn.init.xavier_uniform_(getattr(self, 'l{}'.format(i)).weight)
            elif initialization == 'kaiming_uniform':
                nn.init.kaiming_uniform_(getattr(self, 'l{}'.format(i)).weight,nonlinearity=non_linearity)
            getattr(self, 'l{}'.format(i)).bias.data.fill_(0.0)
        
    def forward(self, inputs):
        batch_size, _, n_channels, seq_length = inputs.size()
        ind = 0
        probs = []

        while ind + self.nperseg <= seq_length:
            x = inputs[:, :, ind:ind+self.nperseg]

            x = self.conv_block(x)
            x = x.squeeze(2)
            x = x.permute(0,2,1)

            xA =torch.einsum('ijk,lk->iljk', x, self.q1)
            xA = xA.reshape(batch_size, self.n_channels*self.n_filters, self.nperseg)
            xAx = torch.bmm(xA, x.permute(0,2,1)).reshape(batch_size, self.n_filters, self.n_channels, self.n_channels)
            output = xAx.reshape(batch_size, self.n_filters*self.n_channels**2)

            if self.non_linearity is not None:
                outputs = self.non_linearity(output)

            for i in range(self.num_layers):
                outputs = getattr(self, 'l{}'.format(i+1))(outputs)
                if self.non_linearity is not None and i < self.num_layers-1:
                    outputs = self.non_linearity(outputs)

            outputs = nn.LogSoftmax(dim=1)(outputs).unsqueeze(1)
            probs.append(outputs)
            ind += self.nperseg - self.noverlap
        
        probs = torch.cat(probs, dim=1)
        probs = torch.mean(probs, dim=1)

        return probs
    
    
# Construct a model with one layer
class Quadra_MLP(nn.Module):
    
    def __init__(self, n_channels, n_filters, nperseg, noverlap, seq_length, layer_size_list, non_linearity, initialization):
        super().__init__()

        # Compute the number of layers and their dimensions
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.seq_length = seq_length
        self.output_size = layer_size_list[-1]
        self.layer_size_list = layer_size_list
        self.layer_size_list.insert(0, n_filters*n_channels**2)
        self.num_layers = len(self.layer_size_list) - 1

        # Define the non-linearity function
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
        
        # Define the layers
        self.q1 = nn.Parameter(torch.zeros(n_filters, self.nperseg, self.nperseg))
        for i in range(self.num_layers):
            setattr(self, 'l{}'.format(i+1), nn.Linear(self.layer_size_list[i], self.layer_size_list[i+1]))

        #initialize weights and biases
        self.initialize_weights(initialization,non_linearity)

    def initialize_weights(self,initialization, non_linearity):
        #initialize weights and biases
        if initialization == 'normal':
            getattr(self, 'q{}'.format(1)).data.normal_(0.0, 1)
        elif initialization == 'xavier':
            nn.init.xavier_normal_(getattr(self, 'q{}'.format(1)))
        elif initialization == 'kaiming':
            nn.init.kaiming_normal_(getattr(self, 'q{}'.format(1)),nonlinearity=non_linearity)
        elif initialization == 'zero':
            getattr(self, 'q{}'.format(1)).data.fill_(0.0)
        elif initialization == 'one':
            getattr(self, 'q{}'.format(1)).data.fill_(1.0)
        elif initialization == 'uniform':
            getattr(self, 'q{}'.format(1)).data.uniform_(-1.0, 1.0)
        elif initialization == 'xavier_uniform':
            nn.init.xavier_uniform_(getattr(self, 'q{}'.format(1)))
        elif initialization == 'kaiming_uniform':
            nn.init.kaiming_uniform_(getattr(self, 'q{}'.format(1)),nonlinearity=non_linearity)

        for i in range(1,self.num_layers):
            if initialization == 'normal':
                getattr(self, 'l{}'.format(i)).weight.data.normal_(0.0, 1)
            elif initialization == 'xavier':
                nn.init.xavier_normal_(getattr(self, 'l{}'.format(i)).weight)
            elif initialization == 'kaiming':
                nn.init.kaiming_normal_(getattr(self, 'l{}'.format(i)).weight,nonlinearity=non_linearity)
            elif initialization == 'zero':
                getattr(self, 'l{}'.format(i)).weight.data.fill_(0.0)
            elif initialization == 'one':
                getattr(self, 'l{}'.format(i)).weight.data.fill_(1.0)
            elif initialization == 'uniform':
                getattr(self, 'l{}'.format(i)).weight.data.uniform_(-1.0, 1.0)
            elif initialization == 'xavier_uniform':
                nn.init.xavier_uniform_(getattr(self, 'l{}'.format(i)).weight)
            elif initialization == 'kaiming_uniform':
                nn.init.kaiming_uniform_(getattr(self, 'l{}'.format(i)).weight,nonlinearity=non_linearity)
            getattr(self, 'l{}'.format(i)).bias.data.fill_(0.0)
        
    def forward(self, inputs):
        batch_size, n_channels, seq_length = inputs.size()
        ind = 0
        probs = []

        while ind + self.nperseg <= seq_length:
            x = inputs[:, :, ind:ind+self.nperseg]
            xA =torch.einsum('ijk,nkk->injk', x, self.q1)
            xAx = torch.einsum('ijkl,inm->ijkm', xA, x.transpose(1,2))
            outputs = xAx.reshape(batch_size, self.n_filters*self.n_channels**2)

            if self.non_linearity is not None:
                outputs = self.non_linearity(outputs)

            for i in range(self.num_layers):
                outputs = getattr(self, 'l{}'.format(i+1))(outputs)
                if self.non_linearity is not None and i < self.num_layers-1:
                    outputs = self.non_linearity(outputs)

            outputs = nn.LogSoftmax(dim=1)(outputs).unsqueeze(1)
            probs.append(outputs)
            ind += self.nperseg - self.noverlap
        
        probs = torch.cat(probs, dim=1)
        probs = torch.mean(probs, dim=1)

        return probs

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, layer_size_list, d_model, num_heads, num_layers, d_ff, nperseg, noverlap, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Linear(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, nperseg)
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.stepsize = nperseg - noverlap

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.hidden_size = layer_size_list[0]
        self.output_size = layer_size_list[-1]
        
        self.fc1 = nn.Linear(d_model, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, src):
        batch_size, seq_length, n_channels = src.size()
        ind = 0
        probs = []

        while ind + self.nperseg <= seq_length:
            x = src[:, ind:ind+self.nperseg, :]
            x_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(x)))

            enc_output = x_embedded
            for enc_layer in self.encoder_layers:
                enc_output = enc_layer(enc_output)
            
            output = torch.max(enc_output, dim=1).values
            output = self.fc1(output)
            output = self.fc2(self.relu(output))
            output = self.softmax(output).unsqueeze(1)
            probs.append(output)
            ind += self.stepsize

        probs = torch.concatenate(probs, dim=1)
        probs = torch.mean(probs, dim=1)

        return probs


class ATCNet(nn.Module):
    r'''
    ATCNet: An attention-based temporal convolutional network forEEG-based motor imagery classiﬁcation.For more details ,please refer to the following information:

    - Paper: H. Altaheri, G. Muhammad and M. Alsulaiman, "Physics-Informed Attention Temporal Convolutional Network for EEG-Based Motor Imagery Classification," in IEEE Transactions on Industrial Informatics, vol. 19, no. 2, pp. 2249-2258, Feb. 2023, doi: 10.1109/TII.2022.3197419.
    - URL: https://github.com/Altaheri/EEG-ATCNet

    .. code-block:: python
        
        import torch
        
        from torcheeg.models import ATCNet

        model = ATCNet(in_channels=1,
                       num_classes=4,
                       num_windows=3,
                       num_electrodes=22,
                       chunk_size=128)

        input = torch.rand(2, 1, 22, 128) # (batch_size, in_channels, num_electrodes,chunk_size) 
        output = model(input)

    Args:
        in_channels (int): The number of channels of the signal corresponding to each electrode. If the original signal is used as input, in_channels is set to 1; if the original signal is split into multiple sub-bands, in_channels is set to the number of bands. (default: :obj:`1`)
        num_electrodes (int): The number of electrodes. (default: :obj:`32`)
        num_classes (int): The number of classes to predict. (default: :obj:`4`)
        num_windows (int): The number of sliding windows after conv block. (default: :obj:`3`)
        num_electrodes (int): The number of electrodes if the input is EEG signal. (default: :obj:`22`)
        conv_pool_size (int):  The size of the second average pooling layer kernel in the conv block. (default: :obj:`7`)
        F1 (int): The channel size of the temporal feature maps in conv block. (default: :obj:`16`)
        D (int): The number of second conv layer's filters linked to each temporal feature map in the previous layer in conv block. (default: :obj:`2`)
        tcn_kernel_size (int): The size of conv layers kernel in the TCN block. (default: :obj:`4`)
        tcn_depth (int): The times of TCN loop. (default: :obj:`2`)
        chunk_size (int): The Number of data points included in each EEG chunk. (default: :obj:`1125`)
    '''
    def __init__(self,
                    in_channels: int = 1,
                    num_classes: int = 4,
                    num_windows: int = 3,
                    num_electrodes: int = 22,
                    conv_pool_size: int = 7,
                    F1: int = 16,
                    D: int =2,
                    tcn_kernel_size: int = 4,
                    tcn_depth: int = 2,
                    chunk_size: int = 1125,
                    filter_size: int = 64
                    ):  
        super(ATCNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_windows = num_windows
        self.num_electrodes = num_electrodes
        self.pool_size = conv_pool_size
        self.F1 = F1
        self.D = D
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_depth = tcn_depth
        self.chunk_size = chunk_size
        F2 = F1*D

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels,F1,(1,filter_size),stride=1,padding = 'same',bias=False),
            nn.BatchNorm2d(F1, False),
            nn.Conv2d(F1,F2,(num_electrodes,1),padding = 0,groups=F1),
            nn.BatchNorm2d(F2,False),
            nn.ELU(),
            nn.AvgPool2d((1,8)),
            nn.Dropout2d(0.1),
            nn.Conv2d(F2,F2,(1,16),bias=False,padding='same'),
            nn.BatchNorm2d(F2,False),
            nn.ELU(),
            nn.AvgPool2d((1,self.pool_size)),
            nn.Dropout2d(0.1)
        )
        self.__build_model()
        
    def __build_model(self):
        with torch.no_grad():
            x = torch.zeros(2,self.in_channels,self.num_electrodes,self.chunk_size)
            x = self.conv_block(x)
            x = x[:,:,-1,:]
            x = x.permute(0,2,1)
            self.__chan_dim,self.__embed_dim = x.shape[1:]
            self.win_len = self.__chan_dim - self.num_windows +1

            for i in range(self.num_windows):
                st = i 
                end = x.shape[1]  -self.num_windows+i+1
                x2 = x[:,st:end,:]

                self.__add_msa(i)
                x2_= self.get_submodule("msa"+str(i))(x2,x2,x2)[0]
                self.__add_msa_drop(i) 
                x2_ = self.get_submodule("msa_drop"+str(i))(x2)
                x2 = torch.add(x2,x2_)
                
                for j in range(self.tcn_depth):
                    self.__add_tcn((i+1)*j,x2.shape[1])
                    out = self.get_submodule("tcn"+str( (i+1)*j ))(x2)
                    if x2.shape[1] != out.shape[1]: 
                        self.__add_recov(i)                   
                        x2 = self.get_submodule("re"+str(i))(x2)
                    x2 = torch.add(x2,out)
                    x2 = nn.ELU()(x2) 
                x2 = x2[:,-1,:]
                self.__dense_dim = x2.shape[-1]
                self.__add_dense(i)
                x2 = self.get_submodule("dense"+str(i))(x2)

   
    def __add_msa(self,index:int):
        
        self.add_module('msa'+str(index),nn.MultiheadAttention(
                                         embed_dim=self.__embed_dim,
                                         num_heads=2,
                                         batch_first=True))
    def __add_msa_drop(self,index):
        self.add_module('msa_drop'+str(index),nn.Dropout(0.3))

    def __add_tcn(self,index:int,num_electrodes:int):
        self.add_module('tcn'+str(index), 
           nn.Sequential(
            nn.Conv1d(num_electrodes,32,self.tcn_kernel_size,padding='same'),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Conv1d(32,32,self.tcn_kernel_size,padding = 'same'),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3) )
        )

    def __add_recov(self,index:int):
        self.add_module('re'+str(index),nn.Conv1d(self.win_len,32,4,padding='same'))

    def __add_dense(self, index:int):
        self.add_module('dense'+str(index),nn.Linear(self.__dense_dim,self.num_classes))

    def forward(self,x):
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 22, 1125]`. Here, :obj:`n` corresponds to the batch size, :obj:`22` corresponds to :obj:`num_electrodes`, and :obj:`1125` corresponds to :obj:`chunk_size`.

        Returns:
            torch.Tensor[size of batch, number of classes]: The predicted probability that the samples belong to the classes.
        '''
        x = self.conv_block(x)
        x = x[:,:,-1,:]
        x = x.permute(0,2,1)

        
        for i in range(self.num_windows):
            st = i 
            end = x.shape[1] -self.num_windows+i+1
            x2 = x[:,st:end,:]
            x2_= self.get_submodule("msa"+str(i))(x2,x2,x2)[0] 
            x2_ = self.get_submodule("msa_drop"+str(i))(x2)
            x2 = torch.add(x2,x2_)
            

            for j in range(self.tcn_depth):
               out = self.get_submodule("tcn"+str( (i+1)*j ))(x2)
               if x2.shape[1] != out.shape[1]:                 
                    x2 = self.get_submodule("re"+str(i))(x2)
               x2 = torch.add(x2,out)
               x2 = nn.ELU()(x2) 
            x2 = x2[:,-1,:]
            x2 = self.get_submodule("dense"+str(i))(x2)
            if i == 0:
                sw_concat = x2
            else:
                sw_concat =sw_concat.add(x2)

        x = sw_concat/self.num_windows
        x = nn.Softmax(dim=1)(x)
        return x

class DeepFourierTransform(torch.nn.Module):

    def __init__(self, nperseg, noverlap, seq_length, mapping_size=100, output_dim = 2, scale=10):
        super().__init__()

        self.nperseg = nperseg
        self.noverlap = noverlap
        self.seq_length = seq_length

        self.fc_cos = nn.Linear(nperseg, mapping_size)
        self.fc_sin = nn.Linear(nperseg, mapping_size)
        self.fc = nn.Linear(2*mapping_size, output_dim)

    def forward(self, input):

        batch_size, seq_length = input.size()
        ind = 0
        fft_list = []

        while ind + self.nperseg <= seq_length:
            x = input[:, ind:ind+self.nperseg]
            cos_x = torch.cos(self.fc_cos(x))
            sin_x = torch.sin(self.fc_sin(x))
            x = torch.cat([cos_x, sin_x], dim=1)
            x = self.fc(x)
            fft_list.append(x.unsqueeze(1))
            ind += self.nperseg - self.noverlap
        
        fft_list = torch.cat(fft_list, dim=1)
        fft = torch.mean(fft_list, dim=1)
        probs = nn.LogSoftmax(dim=1)(fft)

        return probs
    

class DeepWelchTransform(torch.nn.Module):

    def __init__(self, nperseg, noverlap, seq_length, mapping_size=100, output_dim = 2, scale=10):
        super().__init__()

        self.nperseg = nperseg
        self.noverlap = noverlap
        self.seq_length = seq_length

        self.freqs = nn.Parameter(torch.arange(0, 1, nperseg).float())
        self.fc = nn.Linear(1, 1)

    def forward(self, input):

        batch_size, seq_length = input.size()
        ind = 0
        fft_list = []

        while ind + self.nperseg <= seq_length:
            x = input[:, ind:ind+self.nperseg]
            v = torch.exp(- 2 * math.pi * 1j * self.freqs)
            x = torch.einsum('ij,j->i', x, v)
            fft_list.append(x.unsqueeze(1))
            ind += self.nperseg - self.noverlap
        
        fft_list = torch.cat(fft_list, dim=1)
        fft = torch.mean(fft_list, dim=1).unsqueeze(1)
        psd = torch.abs(fft)**2
        psd = self.fc(psd)

        return psd
import numpy as np
import torch
from torch.autograd import Variable
from SimpleNeuralNet import Net
from scipy import optimize

class DG:
    
    def __init__(self, num_epochs, learning_rate, H, D, 
                 alpha = 1.0, beta = 1000):
        """
        A pytorch implementation of Deep Networks for Global Optimization [1]. This module performs Bayesian Linear Regression with basis function extracted from a
        neural network.
        
        [1] J. Snoek, O. Rippel, K. Swersky, R. Kiros, N. Satish, 
            N. Sundaram, M.~M.~A. Patwary, Prabhat, R.~P. Adams
            Scalable Bayesian Optimization Using Deep Neural Networks
            Proc. of ICML'15
            
        Parameters
        ----------

            
        """
        self.X = None
        self.Y = None
        self.network = None
        self.alpha = alpha
        self.beta = beta
        self.init_learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.H = H # the neural number of the middle layers
        self.D = D # size of the last hidden layer
        
    def train(self, X, Y):
        """
        Trains the model on the provided data.
        The training data base can be enriched.
        Parameters
        ----------
        X: np.ndarray (N, D)
            Input datapoints. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        self.X: torch float tensor of the normalized input(X)
        Y: np.ndarray (N, T)
            The corresponding target values.
            The dimensionality of Y is (N, T), where N has to
            match the number of points of X and T is the number of objectives
        self.Y: torch float tensor of the normalized Y
        """
        # Normalize inputs        
        (normX, normY) = self.normalize(X, Y)
        self.X = Variable(torch.from_numpy(normX).float())
        self.Y = Variable(torch.from_numpy(normY).float(), requires_grad=False)
        features = X.shape[1]
        targets = Y.shape[1]
        self.network = Net(features, self.H, self.D, targets) # [Ramneet-Singh]: Modified this to handle multiple outputs
        loss_fn = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.init_learning_rate)
        for t in range(self.num_epochs):
            y_pred = self.network(self.X)
            #print(y_pred.shape)
            #print(self.Y.shape)
            # [Ramneet-Singh]: Changing this to handle multiple outputs
            loss = loss_fn(y_pred.view(-1, targets), self.Y.view(-1, targets))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.phi = self.network.PHI(self.X).data

        # Find the hyperparameters through optimization for each output variable
        for i in range(targets):
            res = optimize.fmin(self.marginal_log_likelihood_wrapper(i), np.random.rand(2))
            self.hypers[2*i], self.hypers[2*i+1] = np.exp(res[0]), np.exp(res[1]) 
        # res = optimize.fmin(self.marginal_log_likelihood, np.random.rand(2))
        # self.hypers = [np.exp(res[0]), np.exp(res[1])]
        return(self.hypers)
    
    def marginal_log_likelihood_wrapper(self, output_idx):
        def marginal_log_likelihood(theta): # theta are the hyperparameters to be optimized
            #print(theta)
            #print(type(theta))
            if np.any((-5 > np.array(theta))) + np.any((np.array(theta) > 10)):
                return -1e25
            alpha = np.exp(theta[0]) # it is not clear why here we calculate the exponential
            beta = np.exp(theta[1])
            Ydata = self.Y.data[:, output_idx] # for the bayesian part, we do not need Y to be a variable anymore
            D = self.X.size()[1]
            N = self.X.size()[0]
            Identity = torch.eye(self.phi.size()[1])
            self.phi_T = torch.transpose(self.phi, 0, 1)
            self.K = torch.addmm(Identity, self.phi_T, self.phi, beta=beta, alpha=alpha)
            self.K_inverse = torch.inverse(self.K)
            m = beta*torch.mm(self.K_inverse, self.phi_T)
            self.m[:, output_idx] = torch.mv(m, Ydata)
            mll = (D/2.)*np.log(alpha)
            mll += (N/2.)*np.log(beta)
            mll -= (N/2.) * np.log(2*np.pi)
            mll -= (beta/2.)* torch.norm(Ydata - torch.mv(self.phi, self.m[:, output_idx]),2)
            mll -= (alpha/2.) * torch.dot(self.m[:, output_idx],self.m[:, output_idx])
            Knumpy = self.K.numpy() # convert K to numpy for determinant calculation
            mll -= 0.5*np.log(np.linalg.det(Knumpy))
            return -mll
        return marginal_log_likelihood
    
    def predict(self, xtest):
        mx = Variable(torch.from_numpy(np.array(self._mx)).float())
        sx = Variable(torch.from_numpy(np.array(self._sx)).float())
        xtest = (xtest - mx)/sx
        phi_test = self.network.PHI(xtest).data
        phi_T = torch.transpose(phi_test, 0, 1)
        # Predict for each output variable
        y_mean = []
        y_var = []
        for i in range(len(self._my)):
            self.marginal_log_likelihood_wrapper(i)(self.hypers[2*i : 2*i+2])
            mean = np.dot(phi_test.numpy(), self.m[:, i])
            mean = mean*self._sy[i] + self._my[i]
            var = np.diag(np.dot(phi_test.numpy(),np.dot(self.K_inverse.numpy(), phi_T.numpy())))+(1./self.hypers[2*i + 1])
            y_mean.append(mean)
            y_var.append(var)

        return y_mean, y_var
    
    # [Ramneet-Singh]: Modifying to handle multiple outputs
    def normalize(self, x, y):
        col_x=x.shape[1]
        row_x=x.shape[0]
        mx=list()
        sx=list()
        for i in range(col_x):
            mx.append(np.mean(x[:,i]))
            sx.append(np.std(x[:,i],ddof=1))
        self._mx=mx
        self._sx=sx
        mx_mat=np.mat(np.zeros((row_x,col_x)))
        sx_mat=np.mat(np.zeros((row_x,col_x)))
        for i in range(row_x):
            mx_mat[i,:]=mx
            sx_mat[i,:]=sx
        x_nom=(x-mx_mat)/sx_mat

        col_y=y.shape[1]
        row_y=y.shape[0]
        my=list()
        sy=list()
        for i in range(col_y):
            my.append(np.mean(y[:,i]))
            sy.append(np.std(y[:,i],ddof=1))
        self._my=my
        self._sy=sy
        my_mat=np.mat(np.zeros((row_y,col_y)))
        sy_mat=np.mat(np.zeros((row_y,col_y)))
        for i in range(row_y):
            my_mat[i,:]=my
            sy_mat[i,:]=sy
        y_nom=(y-my_mat)/sy_mat

        return x_nom,y_nom
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# def marginal_log_likelihood(self, theta): # theta are the hyperparameters to be optimized
#             #print(theta)
#             #print(type(theta))
#             if np.any((-5 > np.array(theta))) + np.any((np.array(theta) > 10)):
#                 return -1e25
#             alpha = np.exp(theta[0]) # it is not clear why here we calculate the exponential
#             beta = np.exp(theta[1])
#             Ydata = self.Y.data # for the bayesian part, we do not need Y to be a variable anymore
#             D = self.X.size()[1]
#             N = self.X.size()[0]
#             Identity = torch.eye(self.phi.size()[1])
#             self.phi_T = torch.transpose(self.phi, 0, 1)
#             self.K = torch.addmm(beta, Identity, alpha, self.phi_T, self.phi)
#             self.K_inverse = torch.inverse(self.K)
#             m = beta*torch.mm(self.K_inverse, self.phi_T)
#             # [Ramneet-Singh]: Changed to handle multiple targets in output
#             self.m = torch.mm(m, Ydata)
#             mll = (D/2.)*np.log(alpha)*np.ones(Ydata.shape[1])
#             mll += (N/2.)*np.log(beta)*np.ones(Ydata.shape[1])
#             mll -= (N/2.) * np.log(2*np.pi)*np.ones(Ydata.shape[1])
#             mll -= (beta/2.)* torch.linalg.norm(Ydata - torch.mm(self.phi, self.m), dim=0)
#             mll -= (alpha/2.) * (torch.matmul(torch.transpose(self.m, 0, 1), self.m).diag())
#             Knumpy = self.K.numpy() # convert K to numpy for determinant calculation
#             mll -= 0.5*np.log(np.linalg.det(Knumpy))*np.ones(Ydata.shape[1])
#             return -mll
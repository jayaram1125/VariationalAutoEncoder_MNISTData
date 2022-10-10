import gzip
import pickle
import torch
import torch.nn as nn
import torch.distributions


def encoder_net(input_dim = 784 ,hidden_dim=200,latent_dim = 100):
	model =  nn.Sequential(
		nn.Linear(input_dim,hidden_dim),
		nn.ReLU(inplace=True),
		nn.Linear(hidden_dim,hidden_dim),
		nn.ReLU(inplace=True),
		nn.Linear(hidden_dim,latent_dim*2)
	)
	return model

def decoder_net(latent_dim = 100,hidden_dim=200,output_dim = 784):
	model =  nn.Sequential(
		nn.Linear(latent_dim,hidden_dim),
		nn.ReLU(inplace=True),
		nn.Linear(hidden_dim,hidden_dim),
		nn.ReLU(inplace=True),
		nn.Linear(hidden_dim,output_dim)
	)
	return model


class Encoder(nn.Module):
	def __init__ (self,latent_dim):
		super().__init__()
		self.network = encoder_net()
		self.latent_dim = latent_dim 
		self.softplus = torch.nn.Softplus()

	def forward(self,x):
		encoder_net_logits = self.network(x) 		
		mu = encoder_net_logits[:,:self.latent_dim]
		sigma = self.softplus(encoder_net_logits[:,self.latent_dim:])
		return mu,sigma

class Decoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.network = decoder_net()

	def forward(self,x):
		decoder_net_logits = self.network(x) 
		return decoder_net_logits

class VAEDataAug:
	def __init__(self):
		self.latent_dim = 100
		self.encoder_model = Encoder(self.latent_dim)
		self.decoder_model = Decoder()

	def load_and_split_mnist_data(self):
		filename = 'mnist.pkl.gz'
		f = gzip.open(filename, 'rb')
		
		uf = pickle._Unpickler(f)
		uf.encoding = 'latin1'

		training_data, validation_data, test_data = uf.load()

		f.close()

		self.train_input, self.train_label = training_data
		self.validation_input, self.validation_label = validation_data
		self.test_input, self.test_label = test_data


	def calculate_kl_divergence(self,mu,sigma):
		# 0.5*[−∑(log𝜎^2) -n +∑𝜎^2+∑𝜇^2]
		# 𝜇 is mean and 𝜎 is the standard deviation of distribution q_z  
		return 0.5*(-torch.sum(torch.log(torch.pow(sigma,2))) -self.latent_dim + torch.sum(torch.pow(sigma,2)) + torch.sum(torch.pow(mu,2)))

	def execute(self,print_frequency=1000,set_grad_val = True):
		with torch.set_grad_enabled(set_grad_val):

			p_z = torch.distributions.normal.Normal(torch.zeros(self.latent_dim,dtype=torch.float32),torch.ones(self.latent_dim,dtype=torch.float32))

			if set_grad_val:
				data = self.train_input
				optimizer = torch.optim.Adam(list(self.encoder_model.parameters()) + list(self.decoder_model.parameters()),lr=0.001)
			else:
				data = self.test_input	

			for i in range(0,len(data)):
				mu,sigma = self.encoder_model(torch.from_numpy(data[i]).type(torch.float32).reshape(1,784))

				q_z = torch.distributions.normal.Normal(mu,sigma)

				# KL divergence between q_z and latent prior p_z,where p_z is multivariate gaussian
				#distribution with mean =0 and standard deviation =1

				kl_divergence = self.calculate_kl_divergence(mu,sigma) 
				decoder_net_logits = self.decoder_model(q_z.sample())

				p_x_given_z = torch.distributions.bernoulli.Bernoulli(logits = decoder_net_logits)
				

				data_tensor = torch.from_numpy(data[i]).type(torch.float32).reshape(1,784)
				data_tensor_out = (data_tensor>0.0).float()
				
				expected_loglikelihood = torch.sum(p_x_given_z.log_prob(data_tensor_out)) 

				elbo = expected_loglikelihood-kl_divergence

				if set_grad_val:
					loss = -elbo 
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

				if i%print_frequency ==0 or i == len(data)-1:
					print("Iteration:%d,Elbo=%f,Expected Loglikelihood=%f"%(i,elbo,expected_loglikelihood))

	def train(self):
		print("\n Train")
		self.execute(print_frequency=5000,set_grad_val =True)


	def test(self):	
		print("\n Test")
		self.execute(print_frequency=1000,set_grad_val =False)


if __name__ == "__main__":
	obj = VAEDataAug()
	obj.load_and_split_mnist_data()
	
	print("Train data set size = %d"%(len(obj.train_input)))
	print("Test data set size = %d"%(len(obj.test_input)))

	obj.train()
	obj.test()





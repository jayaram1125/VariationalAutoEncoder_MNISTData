import gzip
import pickle
import torch
import torch.nn as nn



def encoder_net(input_dim = 784 ,hidden_dim=200,latent_dim = 100):
	model =  nn.Sequential(
		nn.Linear(input_dim,hidden_dim),
		nn.ReLU(inplace=True),
		nn.Linear(hidden_dim,hidden_dim),
		nn.ReLU(inplace=True),
		nn.Linear(hidden_dim,latent_dim)
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
	def __init__ (self):
		pass

	def forward(self,x):
		self.encoder_net_logits = encoder_net(x) 
		mu = self.encoder_net_logits 
		sigma = torch.nn.Softplus(self.encoder_net_logits)
		return mu,sigma


class Decoder(nn.Module):
	def __init__(self):
		pass

	def forward(self):
		self.decoder_net_logits = decoder_net()
		return self.decoder_net_logits

class NormalLogProb(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, loc, scale, z):
		var = torch.pow(scale, 2)
		return -0.5 * torch.log(2 * np.pi * var) - torch.pow(z - loc, 2) / (2 * var)

class BernoulliLogProb(nn.Module):
	def __init__(self):
		super().__init__()
		self.bce_with_logits = nn.BCEWithLogitsLoss(reduction="none")

	def forward(self, logits,target):
		# bernoulli log prob is equivalent to negative binary cross entropy
		return self.bce_with_logits(logits,target)


class VAEDataAug:

	def __init__(self):
		pass

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

	def execute(self,print_frequency=1000,set_grad_val = True):
		with torch.set_grad_enabled(set_grad_val):
			
			latent_dim = 100 
			
			encoder_model = Encoder()
			decoder_model = Decoder()
			
			bernoulli_log_prob = BernoulliLogProb() 
			normal_log_prob = NormalLogProb()

			p_z = normal_log_prob(torch.zeros(latent_dim,dtype=torch.float32),sigma,torch.ones(latent_dim,dtype=torch.float32))
	
			
			if set_grad_val:
				data = self.train_input
			else:
				data = self.test_input	
				optimizer = torch.optim.Adam(list(encoder_model.parameters()) + list(decoder_model.parameters()),lr=0.001)


			for i in range(0,len(data)):
				
				mu,sigma = encoder_model(data[i])
				q_z = normal_log_prob(mu,sigma)


				decoder_net_logits = decoder_model(q_z.sample())

				p_x_given_z = bce_logits(self.decoder_net_logits)
				
				predicted_samples = p_x_given_z.sample()

				kl_divergence =  torch.mean(q_z * (torch.log(q_z) - p_z))

				elbo = loglikelihood-kl_divergence

				if set_grad_val:
					loss = -elbo 
					optimizer.zero_grad()
					loss.backward()


			if i%print_frequency ==0:
				print("\nelbo=%f,loglikelihood=%f"%(elbo,loglikelihood))
				print(elbo)


	def train(self)
		self.execute(print_frequency=5000,set_grad_val =True)


	def test(self):	
		self.execute(print_frequency=1000,set_grad_val =False)


if __name__ == "__main__":
	obj = VAEDataAug()
	obj.load_and_split_mnist_data()

	print(len(obj.train_input))
	print(len(obj.validation_input))
	print(len(obj.test_input))

	print(len(obj.train_label))
	print(len(obj.validation_label))
	print(len(obj.test_label))


	obj.train()
	obj.test()





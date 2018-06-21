'''
# EM Algorithm

'''

import numpy as np

def EMAlgorithm(rolls, theta_A=None, theta_B=None, maxiter=10):

	# inital guess 
	theta_A = theta_A or random.random()
	theta_B = theta_B or random.random()
	thetas = [(theta_A, theta_B)]


	# Iterate till convergence 
	for c in range(maxiter):
		print("After Iteration #%d:\t%0.2f %0.2f" % (c, theta_A, theta_B))
		heads_A, tails_A, heads_B, tails_B = E(rolls, theta_A, theta_B)
		theta_A, theta_B = M(heads_A, tails_A, heads_B, tails_B)

	thetas.append((theta_A,theta_B))    
	return thetas, (theta_A,theta_B)


def E(rolls, theta_A, theta_B):
	'''
	Expectation Step: 
	- calculate liklihoods based on inital estimates
	- calculate the P(A) and P(B) to get expected  head and tail count for A and B
	'''

	heads_A, tails_A = 0,0
	heads_B, tails_B = 0,0
	for trial in rolls:
		likelihood_A = likelihood(trial, theta_A)
		likelihood_B = likelihood(trial, theta_B)
		p_A = likelihood_A / (likelihood_A + likelihood_B)
		p_B = likelihood_B / (likelihood_A + likelihood_B)
		heads_A += p_A * trial.count("H")
		tails_A += p_A * trial.count("T")
		heads_B += p_B * trial.count("H")
		tails_B += p_B * trial.count("T") 
	return heads_A, tails_A, heads_B, tails_B


def M(heads_A, tails_A, heads_B, tails_B):
	'''
	Maximization Step (based on formula):

	thetaA = number of heads for coin A / total heads of coin A and coin B
	thetaB = number of heads for coin B / total heads of coin A and coin B

	where thetaA and thetaB are the biases
	'''
	theta_A = heads_A / (heads_A + tails_A)
	theta_B = heads_B / (heads_B + tails_B)
	return theta_A, theta_B


def likelihood(roll, bias):
	'''
	Coin Liklihood calculations

	- calculating the numerator for each coin 
	- used for calculating P(Z_A | E) and P(Z_B | E)
	'''

	numHeads = roll.count("H")
	flips = len(roll)
	return pow(bias, numHeads) * pow(1-bias, flips-numHeads)



if __name__ == '__main__':

	rolls = [ "TTTTT", "HHHHH", "HHTHH", 
          "THTTT", "HTTTT" ]
	thetas, _ = EMAlgorithm(rolls, 0.5, 0.6, maxiter=6)

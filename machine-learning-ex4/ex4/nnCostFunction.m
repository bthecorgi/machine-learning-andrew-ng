function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
% size(Theta1) - 25x401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
% size(Theta2): 10x26


% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;

% 25x401
Theta1_grad = zeros(size(Theta1));

% 10x26
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Let:
% m = the number of training examples
% n = the number of training features, including the initial bias unit.
% h = the number of units in the hidden layer - NOT including the bias unit
% r = the number of output classifications

% num_labels: r

% m x n
a1 = X;
a1_ones = [ones(m, 1) a1];

% size(a1)
z2 = a1_ones * Theta1';
a2 = sigmoid(z2);
a2_ones = [ones(m, 1) a2];

% contains m x num_labels predictions
H = h3 = sigmoid(a2_ones * Theta2');
% size(H)

% size(y)
% number_of_samples by num_labels

% loop
% Y = zeros(m, num_labels);
% for i = 1:m
% 	% for each sample, make y binary vector
% 	yBinaryVector = zeros(1, num_labels);
% 	yBinaryVector(y(i)) = 1;
	
% 	% store each binary vector in each row
% 	Y(i, :) = yBinaryVector;
% end	

% vectorized
Y=[1:num_labels] == y;

% size(Y)

sumOfKLabels = sum ( -Y .* log(H) - (1 - Y) .* log(1 - H) );
sumMSamples = sum( sumOfKLabels, 2 );
J = (1/m) * sumMSamples;

% Regularization
sumOfUnitsAtLayerWithTheta1 =  sum( sum( Theta1(:, 2:end).^2 ));
sumOfUnitsAtLayerWithTheta2 =  sum( sum( Theta2(:, 2:end).^2 ));
sumOfLayers = sumOfUnitsAtLayerWithTheta1 + sumOfUnitsAtLayerWithTheta2;
regularizationTerm = (lambda/(2*m)) * sumOfLayers

J = J + regularizationTerm;

% gradient

% m x r (5000 x 10)
a3 = H;

d3 = a3 - y;
% m x r
% size(d3)

for t = 1: m
	% size(Y(t, :))
	% size(a3(t, :))
	d3 = (a3(t, :) - Y(t, :)); % 1x10
	d3 = d3'; % 10x1

	% 1x25
	% size(z2(t, :)) 

	% 25x1
	% size(sigmoidGradient(z2(t, :)'))	
	
	d2 = (Theta2' * d3)  .* [1; sigmoidGradient(z2(t, :)')];

	% 25x1
	d2 = d2(2:end);

	Theta1_grad = Theta1_grad + d2 * a1_ones(t, :);
	Theta2_grad = Theta2_grad + d3 * a2_ones(t, :);
end


Theta2_grad = Theta2_grad / m;
Theta1_grad = Theta1_grad / m;

% regularized neural networks
Theta2_grad = Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
Theta1_grad = Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

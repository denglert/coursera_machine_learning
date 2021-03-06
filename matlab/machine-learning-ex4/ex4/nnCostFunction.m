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

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
K = num_labels;
         
% You need to return the following variables correctly 
Theta1_grad = zeros(size(Theta1));
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


X = [ones(m,1) X];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% ---- Cost function ---- %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% - Dimensions:
%%%   -      X:  m*n1
%%%   -      y:  m*1
%%%   -     a1:  n1*m
%%%   - Theta1: n2*n1
%%%   - Theta2: n3*(n2+1)
%%%   -     z2: (n2  )*m 
%%%   -     a2: (n2+1)*m
%%%   -     z3:     n3*m
%%%   -      h:     m*n3

J = 0;

a1 = X.';

z2 = Theta1*a1;
a2 = sigmoid(z2);
a2 = [ones(1, size(a2, 2)) ; a2];

z3 = Theta2*a2;
a3 = sigmoid(z3);

h = a3.';
cost = 0;

for i = 1:m
	for k = 1:K
		y_target = (y(i) == k);
		cost_increment = - y_target*log(h(i,k)) - (1-y_target)*log(1-h(i,k));
		cost = cost + cost_increment;
	end
end


size(Theta1);
size(Theta2);
Theta1sqr = Theta1.^2;
Theta2sqr = Theta2.^2;

cost = cost/m;
cost_regularized = lambda*(sum(sum(Theta1sqr(:,2:end))) + sum(sum(Theta2sqr(:,2:end))) )/(2*m);

J = cost + cost_regularized;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% ---- Backpropagation ---- %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));


%%% --- Looping over the samples
for i = 1:m

	% - Create y_target vector
	y_target = zeros(num_labels,1);

	%size(y_target)
	c = y(i);
	y_target(c) = 1;

	% - Gradient at layer 3
	% - delta3: n3*1
	delta3 = h(i,:)' - y_target;

	% - Gradient at layer 2
	% - delta2: (n2+1)*1
	%size(Theta2)
	%size(delta3)
	%size(y_target)
	%size(h(i,:)')
	
	size(Theta2');	
	size(delta3);
	size(sigmoidGradient(z2(:,i)));
	%size(h(i,:)')	
	%size()	
	
	delta2 = Theta2'*delta3;
	delta2 = delta2(2:end) .* sigmoidGradient(z2(:,i));

	Delta1 = Delta1 + delta2*a1(:,i)';
	Delta2 = Delta2 + delta3*a2(:,i)';

end


Theta1_grad = Delta1/m;
Theta2_grad = Delta2/m;


%%% --- Adding the regularization term
reg_term1 = zeros(size(Theta1));
reg_term2 = zeros(size(Theta2));

reg_term1(:,2:end) = Theta1(:,2:end);
reg_term2(:,2:end) = Theta2(:,2:end);

Theta1_grad = Theta1_grad + (lambda/m)*reg_term1;
Theta2_grad = Theta2_grad + (lambda/m)*reg_term2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

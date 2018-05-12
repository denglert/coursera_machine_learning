function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.


% Theta1: 25x401
% Theta2: 10x26
% X: 5000x401
% a2: 5000x26

size(X)

% - First layer
a1b = [ones(m,1) X];

% - Second layer
a2 = sigmoid(a1b*Theta1.');

n2 = size(a2, 1);
a2b = [ones(n2, 1), a2];

% - Third layer
a3 = sigmoid(a2b*Theta2.');


[argval, argmax] = max(a3, [], 2);
p = argmax;

% =========================================================================


end

function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%



Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigmas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

C_best = -1;
sigma_best = -1;
cost_best = 1e100;

for i = 1:length(Cs)

	for j = 1:length(sigmas)


		C_i = Cs(i);
		sigma_j = sigmas(j);

		model = svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_j)); 
		y_val_pred = svmPredict(model, Xval);

		cost = mean(double(y_val_pred ~= yval));

		if cost < cost_best
			cost_best = cost;
			C_best = C_i;
			sigma_best = sigma_j;
			fprintf('Found a better hyperparmeter. Cost: %f C: %f sigma: %f', cost_best, C_best, sigma_best);
		end
	
		
	end

end


%C_min = 0.01;
%C_factor = 3;
%C_nSteps = 8;
%
%sigma_min = 0.01;
%sigma_factor = 3;
%sigma_nSteps = 8;
%
%C_i = C_min;
%sigma_j = sigma_min;
%
%
%
%for i = 1:C_nSteps
%
%	for j = 1:sigma_nSteps
%
%
%		model = svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_j)); 
%		y_val_pred = svmPredict(model, Xval);
%
%		cost = mean(double(y_val_pred ~= yval));
%
%		if cost < cost_best
%			cost_best = cost;
%			C_best = C_i;
%			sigma_best = sigma_j;
%		end
%	
%		sigma_j= sigma_j*sigma_factor;
%		
%	end
%
%	C_i = C_i*C_factor;
%end



C = C_best;
sigma = sigma_best;


% =========================================================================

end

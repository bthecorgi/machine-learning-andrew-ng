function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
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

cTest = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigmaTest = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% columns: predictionErrors, c, sigma
predictionErrors = zeros(length(cTest) * length(sigmaTest), 3);

errorRow = 1;
for cT = cTest	
	for sT = sigmaTest		
		model= svmTrain(X, y, cT, @(x1, x2) gaussianKernel(x1, x2, sT)); 
		
		predictions = svmPredict(model, Xval);		
		predictionError = mean(double(predictions ~= yval));

		predictionErrors(errorRow, :) = [predictionError, cT, sT];

		errorRow = errorRow + 1;
	endfor
endfor

% sort by column 1 - the prediction error
sortedPredictionErrors = sortrows(predictionErrors, 1);

C = sortedPredictionErrors(1,2);
sigma = sortedPredictionErrors(1,3);

% =========================================================================

end

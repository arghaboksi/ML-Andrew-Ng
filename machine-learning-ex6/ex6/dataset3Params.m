function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = [0.01 0.03 0.1 0.3 1 3 10 30];     % values of C we are going to try
sigma = [0.01 0.03 0.1 0.3 1 3 10 30]; % values of sigma we are going to try

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

% In each iteration we are going to try a pair (C,sigma)
errorMatrix = zeros(length(C)*length(sigma), 3); % each row contains the triplet (C, sigma , error)
iterationCount = 0;
for i = 1:length(C)
  currentC = C(i);
  for j = 1:length(sigma)
    iterationCount += 1;
    currentSigma = sigma(j);
    model= svmTrain(X, y, currentC, @(x1, x2) gaussianKernel(x1, x2, currentSigma));    % value of theta parameters for current C and sigma
    predictions = svmPredict(model, Xval);    % predictions for cross validation set
    error = mean(double(predictions ~= yval));   % associated classification error with current C and sigma
    % printf('C = %0.4f, sigma = %0.4f, Error = %0.4f',currentC,currentSigma,error); 
    errorMatrix(iterationCount,:) = [currentC currentSigma error];  
  endfor
endfor

% Finding C and sigma for which error is minimum
[minError, index] = min(errorMatrix(:,3));
C = errorMatrix(index,1);
sigma = errorMatrix(index,2);
% =========================================================================

end

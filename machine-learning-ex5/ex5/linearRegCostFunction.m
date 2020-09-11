function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Cost Calculation

% Hypothesis predictions for the set of inputs
predictions = X*theta;
% Error in prediction for each example
error = predictions - y;
% Squared errors 
squared_errors = error .^ 2;
% Sum of squared errors
squared_error_sum = sum(squared_errors);
% Total cost without regularization
J = squared_error_sum/(2*m);
% Regularization term 
regularization_term = (sum(theta(2:size(X,2)).^2)*lambda)/(2*m);
% Total cost with regularization 
J = J + regularization_term;


% Gradient Calculation

% Gradient without regularization
grad = ((error'*X)/m)';
%Gradient with regularization
grad(2:size(X,2)) = theta(2:size(X,2))*(lambda/m) + grad(2:size(X,2));



% =========================================================================

grad = grad(:);

end
